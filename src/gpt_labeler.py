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
        debug_mode: bool = False,
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
        self.debug_mode = debug_mode

        # Debug snippet to append to prompt
        self.debug_snippet = """

IMPORTANT: After your YES/NO answer, explain your reasoning:
- QUOTE: Provide the EXACT quote/sentence from the text that proves it's climate-related (or "N/A" if NO)
- REASON: Brief explanation (1-2 sentences)

Format your response EXACTLY like this:
LABEL: [YES or NO]
QUOTE: [exact quote or N/A]
REASON: [brief explanation]"""

        # Lazy load keyword filter for debug mode
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

        # Append debug snippet if in debug mode
        if self.debug_mode:
            prompt += self.debug_snippet

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200 if self.debug_mode else 10,
                    temperature=0
                )

                answer = response.choices[0].message.content

                # Parse based on mode
                if self.debug_mode:
                    result = self._parse_debug_response(answer)
                else:
                    label = parse_response(answer)
                    result = {'label': label} if label else None

                if result is None or result.get('label') is None:
                    logger.warning(f"Could not parse response: {answer}")
                    continue

                return result

            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return None

    def label_single(self, text: str) -> Optional[dict]:
        """
        Label a single text sample with optional chunking.

        If use_chunking=True:
        - Splits text into chunks
        - Labels each chunk independently
        - Returns YES if ANY chunk is labeled YES
        - In debug mode: returns all chunk results

        Args:
            text: Text to label

        Returns:
            Dict with label and optionally chunk details
        """
        if not self.use_chunking or len(text.split()) <= self.chunk_size:
            return self._label_single_chunk(text)

        # Multi-chunk strategy
        chunks = self.split_text_to_chunks(text)
        logger.info(f"Processing {len(chunks)} chunks...")

        chunk_results = []
        yes_count = 0
        no_count = 0

        for i, chunk in enumerate(chunks, 1):
            logger.debug(f"Chunk {i}/{len(chunks)}: {len(chunk)} chars")
            result = self._label_single_chunk(chunk)

            if result and result.get('label'):
                label = result.get('label')

                if self.debug_mode:
                    chunk_results.append({
                        'chunk_number': i,
                        'label': label,
                        'quote': result.get('quote', 'N/A'),
                        'reason': result.get('reason', 'N/A'),
                        'keyword_matches': self.keyword_filter.get_matches_with_context(chunk, 30),
                        'text': chunk
                    })

                if label == 'YES':
                    yes_count += 1
                    if not self.debug_mode:
                        logger.info(f"Chunk {i} labeled YES - returning positive result")
                        return {'label': 'YES'}
                else:
                    no_count += 1

        # Return based on mode
        if self.debug_mode:
            final_label = 'YES' if yes_count > 0 else 'NO'
            return {
                'label': final_label,
                'yes_chunks': yes_count,
                'no_chunks': no_count,
                'chunks': chunk_results
            }
        else:
            # All chunks were NO or failed
            return {'label': 'NO'}

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
        Label a batch of samples with multi-threading.

        Args:
            samples: Iterator yielding dicts with 'text' and 'id'
            output_path: Path to output JSON file
            num_samples: Number of samples to label
            resume: Ignored (always overwrites)
            concurrent_workers: Number of parallel workers

        Returns:
            Dict with statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_processed': 0,
            'labeled_yes': 0,
            'labeled_no': 0,
            'skipped_duplicate': 0,
            'failed': 0
        }

        existing_hashes = set()
        logger.info(f"Starting fresh labeling run")

        def process_sample(sample: dict) -> Optional[dict]:
            """Process single sample (runs in worker thread)."""
            text = sample.get('text', '')
            sample_id = sample.get('id', '')

            # Dedup check
            h = text_hash(text)
            if h in existing_hashes:
                return {'status': 'duplicate'}

            existing_hashes.add(h)

            # Label with GPT
            result = self.label_single(text)

            if result is None:
                return {'status': 'failed'}

            label = result.get('label')
            if label is None:
                return {'status': 'failed'}

            # Build record
            if self.debug_mode:
                record = {
                    'id': sample_id,
                    'label': label,
                    'yes_chunks': result.get('yes_chunks', 0),
                    'no_chunks': result.get('no_chunks', 0),
                    'chunks': result.get('chunks', [])
                }
            else:
                record = {
                    'id': sample_id,
                    'label': label
                }

            return {'status': 'success', 'label': label, 'record': record}

        # Collect samples into list for parallel processing
        sample_list = []
        for sample in samples:
            sample_list.append(sample)
            if len(sample_list) >= num_samples:
                break

        all_records = []

        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(process_sample, s) for s in sample_list]

            pbar = tqdm(total=num_samples, desc="Labeling")

            for future in as_completed(futures):
                result = future.result()

                if result['status'] == 'duplicate':
                    stats['skipped_duplicate'] += 1
                    continue

                if result['status'] == 'failed':
                    stats['failed'] += 1
                    continue

                # Success
                label = result['label']
                record = result['record']

                if label == 'YES':
                    stats['labeled_yes'] += 1
                else:
                    stats['labeled_no'] += 1

                stats['total_processed'] += 1
                all_records.append(record)
                pbar.update(1)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            pbar.close()

        # Sort records by ID for consistent output
        all_records.sort(key=lambda x: x['id'])

        # Write all records as JSON array with indentation
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Labeling complete - YES: {stats['labeled_yes']}, NO: {stats['labeled_no']}, "
            f"Failed: {stats['failed']}, Duplicates: {stats['skipped_duplicate']}"
        )

        return stats
