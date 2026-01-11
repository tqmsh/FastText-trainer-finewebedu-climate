"""
GPT-based labeling for climate content classification.
Uses OpenAI API to label samples as YES/NO for climate-related content.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"


def load_prompt(prompt_path: str = "prompts/climate_yesno.txt") -> str:
    """Load the labeling prompt template."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to max_chars, keeping head and tail.
    This preserves context from both beginning and end of the document.
    """
    if len(text) <= max_chars:
        return text

    # Keep 70% from head, 30% from tail
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars - 20  # 20 chars for separator

    head = text[:head_chars]
    tail = text[-tail_chars:]

    return f"{head}\n\n[...truncated...]\n\n{tail}"


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
        max_chars: int = 2000,
        prompt_path: str = "prompts/climate_yesno.txt",
        debug_mode: bool = False
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.max_chars = max_chars
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

    def label_single(self, text: str) -> Optional[dict]:
        """Label a single text sample. Returns dict with label and optionally quote/reason."""
        # Cap at 200K chars (~50K tokens) to stay well under 128K API limit
        # Newspaper articles can be extremely long (500K+ chars)
        if len(text) > 200000:
            truncated = truncate_text(text, max_chars=200000)
        else:
            truncated = text

        prompt = self.prompt_template.format(text=truncated)

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
        resume: bool = True
    ) -> dict:
        """
        Label a batch of samples and write to JSONL file.

        Args:
            samples: Iterator yielding dicts with 'text' and 'id'
            output_path: Path to output JSONL file
            num_samples: Number of samples to label
            resume: Whether to resume from existing file

        Returns:
            Dict with statistics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing labels for resume and deduplication
        existing_hashes = set()
        existing_count = 0

        if resume and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing_hashes.add(record.get('text_hash', ''))
                        existing_count += 1
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Resuming from {existing_count} existing labels")

        # Statistics
        stats = {
            'total_processed': 0,
            'labeled_yes': 0,
            'labeled_no': 0,
            'skipped_duplicate': 0,
            'failed': 0
        }

        # Open file in append mode
        mode = 'a' if resume and existing_count > 0 else 'w'
        labels_needed = num_samples - existing_count

        if labels_needed <= 0:
            logger.info(f"Already have {existing_count} labels, target is {num_samples}")
            return stats

        logger.info(f"Need to label {labels_needed} more samples")

        with open(output_path, mode, encoding='utf-8') as f:
            pbar = tqdm(total=labels_needed, desc="Labeling")

            for sample in samples:
                if stats['labeled_yes'] + stats['labeled_no'] >= labels_needed:
                    break

                text = sample.get('text', '')
                sample_id = sample.get('id', '')

                # Check for duplicates
                h = text_hash(text)
                if h in existing_hashes:
                    stats['skipped_duplicate'] += 1
                    continue

                existing_hashes.add(h)
                stats['total_processed'] += 1

                # Label with GPT
                result = self.label_single(text)

                if result is None:
                    stats['failed'] += 1
                    continue

                # Extract label
                label = result.get('label')
                if label is None:
                    stats['failed'] += 1
                    continue

                # Update stats
                if label == 'YES':
                    stats['labeled_yes'] += 1
                else:
                    stats['labeled_no'] += 1

                # Write record with field order optimized for readability
                # Important fields first (id, label, quote, reason, keyword_matches), text last
                if self.debug_mode:
                    record = {
                        'id': sample_id,
                        'label': label,
                        'quote': result.get('quote', 'N/A'),
                        'reason': result.get('reason', 'N/A'),
                        'keyword_matches': self.keyword_filter.get_matches_with_context(text, context_chars=30),
                        'text_hash': h,
                        'model': self.model,
                        'prompt_version': PROMPT_VERSION,
                        'timestamp': datetime.utcnow().isoformat(),
                        'text': text  # Full text at the end for easy browsing
                    }
                else:
                    record = {
                        'id': sample_id,
                        'label': label,
                        'text_hash': h,
                        'model': self.model,
                        'prompt_version': PROMPT_VERSION,
                        'timestamp': datetime.utcnow().isoformat(),
                        'text': text
                    }

                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()

                pbar.update(1)

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            pbar.close()

        logger.info(
            f"Labeling complete - YES: {stats['labeled_yes']}, NO: {stats['labeled_no']}, "
            f"Failed: {stats['failed']}, Duplicates: {stats['skipped_duplicate']}"
        )

        return stats
