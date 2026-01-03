"""
Streaming filters for FineWeb dataset.
Implements English language detection and keyword filtering.
"""

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def safe_predict(model, text: str, k: int = 1) -> Tuple[tuple, tuple]:
    """
    Safely call FastText predict with NumPy 2.x compatibility.
    """
    import fasttext
    import numpy as np

    fasttext.FastText.eprint = lambda x: None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.predict(text, k=k)
        except ValueError:
            result = model.f.predict(model.f.getWords(), text, k, 0.0)
            labels = result[0]
            probs = np.asarray(result[1])
            return labels, probs


class LanguageFilter:
    """FastText-based language identification filter."""

    def __init__(self, model_path: str = "models/lid.176.bin", prob_threshold: float = 0.9):
        self.model_path = Path(model_path)
        self.prob_threshold = prob_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Language ID model not found at {self.model_path}. "
                    "Download with: huggingface-cli download facebook/fasttext-language-identification lid.176.bin --local-dir models/"
                )
            self._model = fasttext.load_model(str(self.model_path))
        return self._model

    def is_english(self, text: str) -> bool:
        """Check if text is English with confidence >= threshold."""
        # Clean text for prediction (single line, no newlines)
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return False

        try:
            predictions = safe_predict(self.model, clean_text, k=1)
            label, prob = predictions[0][0], predictions[1][0]
        except Exception as e:
            logger.debug(f"Language detection error: {e}")
            return False

        # Label format is __label__eng_Latn (HuggingFace model) or __label__en (original model)
        is_en = label in ("__label__eng_Latn", "__label__en") and prob >= self.prob_threshold
        return is_en


class KeywordFilter:
    """Keyword-based filter for climate/weather content."""

    def __init__(self, keywords_path: str = "data/weather_terms.txt"):
        self.keywords_path = Path(keywords_path)
        self._keywords: Optional[Set[str]] = None
        self._pattern: Optional[re.Pattern] = None

    @property
    def keywords(self) -> Set[str]:
        if self._keywords is None:
            self._keywords = self._load_keywords()
        return self._keywords

    @property
    def pattern(self) -> re.Pattern:
        if self._pattern is None:
            # Build regex pattern for efficient matching
            escaped = [re.escape(kw) for kw in self.keywords]
            # Use word boundaries for better matching
            pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
            self._pattern = re.compile(pattern_str, re.IGNORECASE)
        return self._pattern

    def _load_keywords(self) -> Set[str]:
        """Load keywords from file."""
        if not self.keywords_path.exists():
            raise FileNotFoundError(f"Keywords file not found at {self.keywords_path}")

        keywords = set()
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and lines with parentheses (inline comments)
                if line and not line.startswith('#') and '(' not in line:
                    keywords.add(line.lower())
        return keywords

    def matches(self, text: str) -> bool:
        """Check if text contains any climate/weather keywords."""
        return bool(self.pattern.search(text))


def iter_candidates(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "default",
    split: str = "train",
    lang_model_path: str = "models/lid.176.bin",
    keywords_path: str = "data/weather_terms.txt",
    english_prob_threshold: float = 0.9,
    max_samples: Optional[int] = None,
    log_interval: int = 10000,
    mode: str = "modern"  # "modern" or "historical"
) -> Iterator[dict]:
    """
    Stream FineWeb and yield samples passing English + keyword filters.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (e.g., 'sample-10BT')
        split: Dataset split
        lang_model_path: Path to FastText language ID model
        keywords_path: Path to keywords file (modern) or historical regex CSV
        english_prob_threshold: Minimum probability for English detection
        max_samples: Maximum samples to process (None for unlimited)
        log_interval: How often to log progress
        mode: Filter mode - "modern" uses data/weather_terms.txt,
              "historical" uses datasets/historical_climate_regex.csv

    Yields:
        Dict with 'text' and metadata for each matching sample
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset {dataset_name}/{dataset_config} in streaming mode...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True
    )

    # Initialize filters
    logger.info("Initializing language filter...")
    lang_filter = LanguageFilter(lang_model_path, english_prob_threshold)
    logger.info("Language filter initialized")
    
    logger.info("Initializing keyword filter...")
    keyword_filter = KeywordFilter(keywords_path)
    logger.info(f"Keyword filter initialized with {len(keyword_filter.keywords)} keywords")

    # Statistics
    total_processed = 0
    passed_english = 0
    passed_keywords = 0

    logger.info("Starting streaming filter...")
    logger.info("Attempting to get first sample from dataset (this may take a moment)...")
    
    # Try to get the first sample to verify the iterator works
    dataset_iter = iter(dataset)
    first_sample_processed = False
    try:
        first_sample = next(dataset_iter)
        logger.info("Successfully retrieved first sample from dataset! Processing will begin now...")
        first_sample_processed = True
        total_processed = 1
        
        # Process the first sample
        text = first_sample.get('text', '')
        if text:
            # Stage A: English filter
            if lang_filter.is_english(text):
                passed_english += 1
                # Stage B: Keyword filter
                if keyword_filter.matches(text):
                    passed_keywords += 1
                    # Yield matching sample with metadata
                    yield {
                        'text': text,
                        'id': first_sample.get('id', f'sample_{total_processed}'),
                        'url': first_sample.get('url', ''),
                        'dump': first_sample.get('dump', ''),
                        'date': first_sample.get('date', ''),
                    }
                    logger.info(f"First matching sample found!")
    except StopIteration:
        logger.error("Dataset iterator is empty!")
        return
    except Exception as e:
        logger.error(f"Error getting first sample from dataset: {e}", exc_info=True)
        raise

    try:
        for sample in dataset_iter:
            total_processed += 1

            # Log that we're continuing processing
            if total_processed == 2 and first_sample_processed:
                logger.info(f"Continuing to process samples...")

            # Heartbeat logging - show we're alive even if no matches
            heartbeat_interval = 1000
            if total_processed % heartbeat_interval == 0:
                eng_rate = passed_english / total_processed * 100 if total_processed > 0 else 0
                kw_rate = passed_keywords / total_processed * 100 if total_processed > 0 else 0
                logger.info(
                    f"[HEARTBEAT] Processed: {total_processed:,} | "
                    f"English: {passed_english:,} ({eng_rate:.2f}%) | "
                    f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
                )

            try:
                text = sample.get('text', '')
                if not text:
                    continue

                # Stage A: English filter
                if not lang_filter.is_english(text):
                    continue
                passed_english += 1

                # Stage B: Keyword filter
                if not keyword_filter.matches(text):
                    continue
                passed_keywords += 1

                # Yield matching sample with metadata
                yield {
                    'text': text,
                    'id': sample.get('id', f'sample_{total_processed}'),
                    'url': sample.get('url', ''),
                    'dump': sample.get('dump', ''),
                    'date': sample.get('date', ''),
                }

                # Log first match
                if passed_keywords == 1:
                    logger.info(f"First matching sample found after processing {total_processed:,} samples")

                # Log when we find matches (every 100 matches)
                if passed_keywords > 0 and passed_keywords % 100 == 0:
                    eng_rate = passed_english / total_processed * 100
                    kw_rate = passed_keywords / total_processed * 100
                    logger.info(
                        f"✓ Found {passed_keywords:,} matches | "
                        f"Processed: {total_processed:,} | "
                        f"English: {passed_english:,} ({eng_rate:.2f}%) | "
                        f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
                    )

                # Log progress (use smaller interval for early samples)
                early_interval = min(1000, log_interval)
                if total_processed <= 10000 and total_processed % early_interval == 0:
                    eng_rate = passed_english / total_processed * 100
                    kw_rate = passed_keywords / total_processed * 100
                    logger.info(
                        f"Processed: {total_processed:,} | "
                        f"English: {passed_english:,} ({eng_rate:.2f}%) | "
                        f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
                    )
                elif total_processed % log_interval == 0:
                    eng_rate = passed_english / total_processed * 100
                    kw_rate = passed_keywords / total_processed * 100
                    logger.info(
                        f"Processed: {total_processed:,} | "
                        f"English: {passed_english:,} ({eng_rate:.2f}%) | "
                        f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
                    )

                # Check limit
                if max_samples and passed_keywords >= max_samples:
                    logger.info(f"Reached max_samples limit: {max_samples}")
                    break

            except Exception as e:
                logger.warning(f"Error processing sample {total_processed}: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Fatal error in dataset iteration: {e}", exc_info=True)
        raise

    # Final statistics
    if total_processed > 0:
        eng_rate = passed_english / total_processed * 100
        kw_rate = passed_keywords / total_processed * 100
        logger.info(
            f"Final stats - Processed: {total_processed:,} | "
            f"English: {passed_english:,} ({eng_rate:.2f}%) | "
            f"Keywords: {passed_keywords:,} ({kw_rate:.2f}%)"
        )


def filter_csv(
    input_path: str = 'datasets/historical_regex_cleaned.csv',
    output_climate: str = 'datasets/historical_climate_regex.csv',
    keywords_path: str = 'data/weather_terms.txt',
    text_column: str = 'Text',
    limit: Optional[int] = None,
    mock: bool = False
):
    """Filter CSV by climate keywords."""
    import csv

    csv.field_size_limit(sys.maxsize)
    kf = KeywordFilter(keywords_path)

    climate_rows = []
    processed = 0

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and processed >= limit:
                break
            processed += 1
            text = row.get(text_column, '')
            matches = list(kf.pattern.findall(text))
            is_climate = bool(matches)

            if mock and is_climate:
                all_matches = set(matches)
                contexts = []
                for match in all_matches:
                    m = re.search(re.escape(match), text, re.IGNORECASE)
                    if m:
                        start = max(0, m.start() - 20)
                        end = min(len(text), m.end() + 20)
                        ctx = text[start:end].replace('\n', ' ').replace('\r', ' ')
                        ctx = re.sub(re.escape(match), f'【{match.upper()}】', ctx, flags=re.IGNORECASE)
                        contexts.append(ctx)
                climate_rows.append({
                    'line_number': processed,
                    'context': ' ... '.join(contexts)
                })
            elif is_climate:
                climate_rows.append(row)

    with open(output_climate, 'w', encoding='utf-8', newline='') as f:
        if climate_rows:
            writer = csv.DictWriter(f, fieldnames=climate_rows[0].keys())
            writer.writeheader()
            writer.writerows(climate_rows)

    logger.info(f"Processed: {processed}, CLIMATE: {len(climate_rows)}")


if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else None

    if mode == '--historical':
        filter_csv(
            input_path='datasets/historical_regex_cleaned.csv',
            output_climate='datasets/historical_climate_regex.csv',
            limit=int(sys.argv[2]) if len(sys.argv) > 2 else None,
            mock=True
        )
    elif mode == '--modern':
        filter_csv(
            input_path='datasets/modern_regex_cleaned.csv',
            output_climate='datasets/modern_climate_regex.csv',
            limit=int(sys.argv[2]) if len(sys.argv) > 2 else None,
            mock=True
        )
    else:
        print("Usage: python src/streaming_filters.py [--historical|--modern] [limit]")
        sys.exit(1)
