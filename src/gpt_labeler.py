"""
GPT-based labeling for climate content classification.
Uses OpenAI API to label samples as YES/NO for climate-related content.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator, Optional

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


def load_prompt(prompt_path: str = "prompts/climate_yesno.txt") -> str:
    """Load the labeling prompt template."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def text_hash(text: str) -> str:
    """Generate a hash for deduplication."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def parse_response(response: str) -> Optional[str]:
    """Parse GPT response to extract YES/NO label."""
    response = response.strip().upper()

    if response in ('YES', 'NO'):
        return response

    # Try to extract from longer responses
    if response.startswith('YES'):
        return 'YES'
    if response.startswith('NO'):
        return 'NO'

    return None


class GPTLabeler:
    """Label samples using GPT API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1,
        chunk_size: int = 500,
        prompt_path: str = "prompts/climate_yesno.txt",
        use_chunking: bool = False
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.chunk_size = chunk_size
        self.use_chunking = use_chunking
        self.prompt_template = load_prompt(prompt_path)

        self.debug_snippet = """

IMPORTANT: After your YES/NO answer, explain your reasoning:
- QUOTE: Provide the EXACT quote/sentence from the text that proves it's climate-related (or "N/A" if NO)
- REASON: Brief explanation (1-2 sentences)

Format your response EXACTLY like this:
LABEL: [YES or NO]
QUOTE: [exact quote or N/A]
REASON: [brief explanation]"""

        self._keyword_filter = None

    @property
    def keyword_filter(self):
        """Lazy load keyword filter on first use."""
        if self._keyword_filter is None:
            from src.streaming_filters import KeywordFilter
            self._keyword_filter = KeywordFilter(keywords_path="data/weather_terms.txt")
        return self._keyword_filter

    def split_text_to_chunks(self, text: str) -> list[str]:
        """
        Split text into chunks of approximately chunk_size words.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    def _label_single_chunk(self, text: str) -> Optional[dict]:
        """Label a single chunk of text (internal method)."""
        prompt = self.prompt_template.format(text=text)
        prompt += self.debug_snippet

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0
                )

                answer = response.choices[0].message.content
                result = self._parse_debug_response(answer)

                if result is None or result.get('label') is None:
                    logger.warning(f"Could not parse response: {answer}")
                    continue

                return result

            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return None

    def label_single(self, text: str, article_id: str = '') -> list[dict]:
        """
        Label chunks with keyword prioritization for cost optimization.

        Returns list of chunk-level labels for FastText training.

        Args:
            text: Article text to split and label
            article_id: Article identifier

        Returns:
            List of dicts with fields ordered for readability:
            article_id, chunk_number, has_keywords, label, labeled, quote, reason, keyword_matches, text
        """
        if not self.use_chunking or len(text.split()) <= self.chunk_size:
            result = self._label_single_chunk(text)
            if not result:
                return []
            keyword_matches = self.keyword_filter.get_matches_with_context(text, 30)
            return [{
                'article_id': article_id,
                'chunk_number': 1,
                'has_keywords': self.keyword_filter.matches(text),
                'label': result.get('label'),
                'labeled': True,
                'quote': result.get('quote', 'N/A'),
                'reason': result.get('reason', 'N/A'),
                'keyword_matches': keyword_matches,
                'text': text
            }]

        chunks = self.split_text_to_chunks(text)
        logger.info(f"Processing {len(chunks)} chunks for article {article_id}...")

        all_chunks = []

        for i, chunk in enumerate(chunks, 1):
            has_keywords = self.keyword_filter.matches(chunk)
            chunk_data = {
                'article_id': article_id,
                'chunk_number': i,
                'has_keywords': has_keywords,
                'label': None,
                'labeled': False,
                'quote': None,
                'reason': None,
                'keyword_matches': self.keyword_filter.get_matches_with_context(chunk, 30) if has_keywords else 'No keyword matches',
                'text': chunk
            }
            all_chunks.append(chunk_data)

        # Only label chunks with keywords (cost optimization)
        chunks_with_keywords = [c for c in all_chunks if c['has_keywords']]

        for chunk_data in chunks_with_keywords:
            result = self._label_single_chunk(chunk_data['text'])
            if result and result.get('label'):
                chunk_data['label'] = result['label']
                chunk_data['labeled'] = True
                chunk_data['quote'] = result.get('quote', 'N/A')
                chunk_data['reason'] = result.get('reason', 'N/A')

        labeled_count = sum(1 for c in chunks_with_keywords if c['labeled'])
        logger.info(f"Article {article_id}: labeled {labeled_count}/{len(chunks_with_keywords)} chunks with keywords (skipped {len(all_chunks) - len(chunks_with_keywords)} without keywords)")

        return [c for c in chunks_with_keywords if c['labeled'] and c['label']]

    def _parse_debug_response(self, response: str) -> Optional[dict]:
        """Parse debug response to extract label, quote, and reason."""
        import re

        # Try to extract LABEL, QUOTE, REASON
        label_match = re.search(r'LABEL:\s*(\w+)', response, re.IGNORECASE)
        quote_match = re.search(r'QUOTE:\s*(.+?)(?=REASON:|$)', response, re.IGNORECASE | re.DOTALL)
        reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE | re.DOTALL)

        if not label_match:
            # Fallback to simple YES/NO parsing
            label = parse_response(response)
            return {'label': label, 'quote': 'N/A', 'reason': 'Could not parse debug response'} if label else None

        label = label_match.group(1).strip().upper()
        quote = quote_match.group(1).strip() if quote_match else 'N/A'
        reason = reason_match.group(1).strip() if reason_match else 'N/A'

        return {
            'label': label,
            'quote': quote,
            'reason': reason
        }

    def label_batch(
        self,
        samples: Iterator[dict],
        output_path: str,
        num_samples: int = 10000,
        resume: bool = False,
        concurrent_workers: int = 1
    ) -> dict:
        """
        Label a batch of articles and output chunk-level labels to JSONL.

        Args:
            samples: Iterator yielding dicts with 'text' and 'id'
            output_path: Path to output JSONL file (one chunk per line)
            num_samples: Number of articles to label
            resume: Ignored (always overwrites for idempotency)
            concurrent_workers: Number of parallel workers

        Returns:
            Dict with statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            'articles_processed': 0,
            'chunks_total': 0,
            'chunks_labeled': 0,
            'chunks_yes': 0,
            'chunks_no': 0,
            'articles_duplicate': 0,
            'articles_failed': 0
        }

        existing_hashes = set()
        logger.info(f"Starting chunk-level labeling (overwriting {output_path})")

        def process_article(sample: dict) -> dict:
            """Process single article into chunks (runs in worker thread)."""
            text = sample.get('text', '')
            article_id = sample.get('id', '')

            h = text_hash(text)
            if h in existing_hashes:
                return {'status': 'duplicate'}

            existing_hashes.add(h)

            chunk_labels = self.label_single(text, article_id=article_id)

            if not chunk_labels:
                return {'status': 'failed'}

            return {
                'status': 'success',
                'chunks': chunk_labels,
                'article_id': article_id
            }

        sample_list = []
        for sample in samples:
            sample_list.append(sample)
            if len(sample_list) >= num_samples:
                break

        with open(output_path, 'w', encoding='utf-8') as f:
            with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                futures = [executor.submit(process_article, s) for s in sample_list]

                pbar = tqdm(total=num_samples, desc="Labeling articles")

                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'duplicate':
                        stats['articles_duplicate'] += 1
                        pbar.update(1)
                        continue

                    if result['status'] == 'failed':
                        stats['articles_failed'] += 1
                        pbar.update(1)
                        continue

                    chunks = result['chunks']
                    stats['articles_processed'] += 1
                    stats['chunks_total'] += len(chunks)

                    for chunk in chunks:
                        stats['chunks_labeled'] += 1
                        if chunk['label'] == 'YES':
                            stats['chunks_yes'] += 1
                        else:
                            stats['chunks_no'] += 1

                        f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

                    pbar.update(1)
                    time.sleep(self.rate_limit_delay)

                pbar.close()

        logger.info(
            f"Chunk labeling complete - Articles: {stats['articles_processed']}, "
            f"Chunks labeled: {stats['chunks_labeled']} (YES: {stats['chunks_yes']}, NO: {stats['chunks_no']}), "
            f"Failed: {stats['articles_failed']}, Duplicates: {stats['articles_duplicate']}"
        )

        return stats
