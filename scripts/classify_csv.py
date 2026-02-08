#!/usr/bin/env python3
"""
FastText Climate Classifier for CSV Files

Usage:
    python scripts/classify_csv.py --input /path/to/input.csv --rows 100 --mode debug --strategy 1
    python scripts/classify_csv.py --input /path/to/input.csv --rows 100 --mode prod --strategy 2
"""

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterator, Set, Tuple

import fasttext
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================
CHUNK_SIZE = 500  # Words per chunk (same as training)
KEYWORDS_PATH = "data/weather_terms.txt"
MODEL_PATH = "models/fasttext_climate.bin"
THRESHOLD = 0.5


# =============================================================================
# NumPy 2.x Compatibility Patch
# =============================================================================
_original_predict = fasttext.FastText._FastText.predict


def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
    """Patched predict that works with NumPy 2.x."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = self.f.predict(text, k, threshold, on_unicode_error)
        if result:
            probs = [float(p) for p, _ in result]
            labels = [l for _, l in result]
            return tuple(labels), probs
        else:
            return (), []


def apply_predict_patch():
    """Apply NumPy 2.x patch for inference."""
    fasttext.FastText._FastText.predict = _patched_predict


def remove_predict_patch():
    """Remove NumPy 2.x patch."""
    fasttext.FastText._FastText.predict = _original_predict


# =============================================================================
# Keyword Filter
# =============================================================================
class KeywordFilter:
    """Keyword-based filter for climate/weather content."""

    def __init__(self, keywords_path: str = KEYWORDS_PATH):
        self.keywords_path = Path(keywords_path)
        self._keywords: Set[str] = None
        self._pattern: re.Pattern = None

    @property
    def keywords(self) -> Set[str]:
        if self._keywords is None:
            self._keywords = self._load_keywords()
        return self._keywords

    @property
    def pattern(self) -> re.Pattern:
        if self._pattern is None:
            escaped = [re.escape(kw) for kw in self.keywords]
            pattern_str = r'\b(' + '|'.join(escaped) + r')\b'
            self._pattern = re.compile(pattern_str, re.IGNORECASE)
        return self._pattern

    def _load_keywords(self) -> Set[str]:
        """Load keywords from file."""
        keywords = set()
        with open(self.keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '(' not in line:
                    keywords.add(line.lower())
        return keywords

    def matches(self, text: str) -> bool:
        """Check if text contains any climate/weather keywords."""
        return bool(self.pattern.search(text))


# =============================================================================
# FastText Classifier
# =============================================================================
class FastTextClimateClassifier:
    """FastText-based climate content classifier."""

    def __init__(self, model_path: str = MODEL_PATH, threshold: float = THRESHOLD):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self._model = fasttext.load_model(str(self.model_path))
        return self._model

    def _clean_text(self, text: str) -> str:
        """Clean text for prediction."""
        return text.replace("\n", " ").strip()

    def _safe_predict(self, text: str, k: int = 2) -> Tuple[tuple, tuple]:
        """Safe predict with NumPy 2.x compatibility."""
        cleaned = self._clean_text(text)
        return self.model.predict(cleaned, k=k)

    def predict_proba(self, text: str) -> float:
        """Get probability of text being climate-related."""
        labels, probs = self._safe_predict(text, k=2)
        for i, label in enumerate(labels):
            if label == '__label__climate' and i < len(probs):
                return probs[i]
        return 0.0

    def is_climate(self, text: str) -> Tuple[bool, float]:
        """Check if text is climate-related. Returns (is_climate, prob)."""
        prob = self.predict_proba(text)
        return prob >= self.threshold, prob


# =============================================================================
# Chunking Utilities
# =============================================================================
def split_text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def prioritize_keyword_chunks_with_debug(
    text: str,
    classifier: FastTextClimateClassifier,
    keyword_filter: KeywordFilter
) -> dict:
    """
    Strategy #1: Chunk text, prioritize keyword chunks, decide based on them.
    Returns detailed debug info for strategy 1.
    Only includes chunks labeled as climate=True by FastText.
    If no climate chunks found, shows sample chunks for debugging.
    """
    chunks = split_text_to_chunks(text, CHUNK_SIZE)

    result = {
        'total_chunks': len(chunks),
        'fasttext_labeled_true_chunks': [],  # Only climate=True chunks
        'keyword_chunk_climate_count': 0,
        'keyword_chunk_other_count': 0,
        'max_climate_prob': 0.0,
        'is_climate': False
    }

    if not chunks:
        return result

    all_chunks_info = []  # Collect all chunk info for fallback debugging

    # Process all chunks
    for i, chunk in enumerate(chunks):
        has_keywords = keyword_filter.matches(chunk)
        is_climate, prob = classifier.is_climate(chunk)

        chunk_info = {
            'chunk_index': i + 1,
            'probability': prob,
            'has_keywords': has_keywords,
            'is_climate': is_climate,
            'text': chunk  # Full text
        }
        all_chunks_info.append(chunk_info)

        # Only include climate=True chunks
        if is_climate:
            result['fasttext_labeled_true_chunks'].append(chunk_info)
            if prob > result['max_climate_prob']:
                result['max_climate_prob'] = prob

        if is_climate:
            result['keyword_chunk_climate_count'] += 1
        else:
            result['keyword_chunk_other_count'] += 1

    # Fallback: if no climate chunks found, show sample chunks for debugging
    if len(result['fasttext_labeled_true_chunks']) == 0:
        result['sample_chunks_for_debug'] = all_chunks_info[:5]

    # Decision: if any keyword chunk is climate, classify as climate
    result['is_climate'] = result['max_climate_prob'] >= classifier.threshold

    return result


def prioritize_keyword_chunks(
    text: str,
    classifier: FastTextClimateClassifier,
    keyword_filter: KeywordFilter
) -> Tuple[bool, float]:
    """
    Strategy #1: Chunk text, prioritize keyword chunks, decide based on them.

    Returns (is_climate, max_prob)
    """
    debug_info = prioritize_keyword_chunks_with_debug(text, classifier, keyword_filter)
    return debug_info['is_climate'], debug_info['max_climate_prob']


def classify_whole_text(
    text: str,
    classifier: FastTextClimateClassifier
) -> Tuple[bool, float]:
    """
    Strategy #2: Just feed the whole text at once.

    Returns (is_climate, prob)
    """
    return classifier.is_climate(text)


# =============================================================================
# CSV Processing
# =============================================================================
def iter_csv_rows(csv_path: str, num_rows: int) -> Iterator[dict]:
    """Iterate over CSV rows."""
    df = pd.read_csv(csv_path, nrows=num_rows)

    # Find the text column - check common names
    text_column = None
    for col in ['Text', 'text', 'context', 'Content', 'content']:
        if col in df.columns:
            text_column = col
            break

    date_column = None
    for col in ['Date', 'date', 'timestamp', 'Time']:
        if col in df.columns:
            date_column = col
            break

    if text_column is None:
        print(f"  WARNING: No text column found. Available columns: {list(df.columns)}")

    for idx, row in df.iterrows():
        yield {
            'index': idx,
            'text': str(row.get(text_column, '')) if text_column else '',
            'date': str(row.get(date_column, '')) if date_column else '',
        }


def process_csv(
    csv_path: str,
    num_rows: int,
    strategy: int,
    classifier: FastTextClimateClassifier,
    keyword_filter: KeywordFilter
) -> Tuple[int, int, list[dict], dict]:
    """
    Process CSV and classify each row.

    Returns (climate_count, other_count, climate_rows, debug_info)
    """
    climate_count = 0
    other_count = 0
    climate_rows = []

    debug_info = {
        'input_file': csv_path,
        'num_rows_processed': num_rows,
        'strategy': strategy,
        'articles': [],
        'summary': {
            'climate_articles': 0,
            'other_articles': 0,
            'keyword_chunk_climate_total': 0,
            'keyword_chunk_other_total': 0
        }
    }

    for row in iter_csv_rows(csv_path, num_rows):
        text = row['text']
        article_idx = row['index']

        if strategy == 1:
            article_debug = prioritize_keyword_chunks_with_debug(
                text, classifier, keyword_filter
            )
            is_climate = article_debug['is_climate']
            prob = article_debug['max_climate_prob']

            # Add article info to debug
            article_debug['article_index'] = article_idx
            article_debug['date'] = row.get('date', '')
            debug_info['articles'].append(article_debug)
            debug_info['summary']['keyword_chunk_climate_total'] += article_debug['keyword_chunk_climate_count']
            debug_info['summary']['keyword_chunk_other_total'] += article_debug['keyword_chunk_other_count']
        else:
            is_climate, prob = classify_whole_text(text, classifier)
            article_debug = None

        if is_climate:
            climate_count += 1
            row['climate_prob'] = prob
            climate_rows.append(row)
        else:
            other_count += 1

    debug_info['summary']['climate_articles'] = climate_count
    debug_info['summary']['other_articles'] = other_count

    return climate_count, other_count, climate_rows, debug_info


def generate_output_filename(input_path: str) -> str:
    """Generate output filename by inserting 'fasttext' after first underscore."""
    path = Path(input_path)
    name = path.stem

    # Find first underscore and insert 'fasttext' after it
    if '_' in name:
        first_underscore = name.index('_')
        new_name = name[:first_underscore + 1] + 'fasttext_' + name[first_underscore + 1:]
    else:
        new_name = name + '_fasttext'

    return str(path.parent / (new_name + path.suffix))


def save_climate_rows(climate_rows: list[dict], output_path: str):
    """Save climate-related rows to CSV."""
    if not climate_rows:
        print("  No climate rows to save.")
        return

    df = pd.DataFrame(climate_rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(climate_rows)} climate rows to: {output_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Classify CSV rows using FastText climate classifier"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--rows', '-n',
        type=int,
        default=100,
        help='Number of rows to process (default: 100)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['debug', 'prod'],
        default='debug',
        help='Mode: debug (JSON output) or prod (save climate rows)'
    )
    parser.add_argument(
        '--strategy', '-s',
        type=int,
        choices=[1, 2],
        default=1,
        help='Strategy: 1=chunk+keyword prioritization, 2=whole text'
    )
    parser.add_argument(
        '--model',
        default=MODEL_PATH,
        help=f'FastText model path (default: {MODEL_PATH})'
    )
    parser.add_argument(
        '--keywords',
        default=KEYWORDS_PATH,
        help=f'Keywords file path (default: {KEYWORDS_PATH})'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=THRESHOLD,
        help=f'Classification threshold (default: {THRESHOLD})'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output path for debug JSON (default: debug.json)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FastText Climate Classifier")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Rows: {args.rows}")
    print(f"Mode: {args.mode}")
    print(f"Strategy: {'chunk+keyword' if args.strategy == 1 else 'whole text'}")
    print(f"Threshold: {args.threshold}")
    print("-" * 60)

    # Apply NumPy 2.x patch
    apply_predict_patch()

    try:
        # Initialize classifier and keyword filter
        classifier = FastTextClimateClassifier(
            model_path=args.model,
            threshold=args.threshold
        )
        keyword_filter = KeywordFilter(keywords_path=args.keywords)

        print(f"Loaded {len(keyword_filter.keywords)} keywords")
        print(f"Model: {args.model}")
        print()

        # Process CSV
        climate_count, other_count, climate_rows, debug_info = process_csv(
            csv_path=args.input,
            num_rows=args.rows,
            strategy=args.strategy,
            classifier=classifier,
            keyword_filter=keyword_filter
        )

        total = climate_count + other_count
        climate_pct = (climate_count / total * 100) if total > 0 else 0

        # Output results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total processed: {total}")
        print(f"Climate-related: {climate_count} ({climate_pct:.1f}%)")
        print(f"Other: {other_count} ({100-climate_pct:.1f}%)")

        # Strategy 1 specific stats
        if args.strategy == 1:
            kw_climate = debug_info['summary']['keyword_chunk_climate_total']
            kw_other = debug_info['summary']['keyword_chunk_other_total']
            print(f"Keyword chunks (climate): {kw_climate}")
            print(f"Keyword chunks (other): {kw_other}")
        print("=" * 60)

        # Debug mode: save detailed info to JSON
        if args.mode == 'debug':
            # Add timestamp and config to debug info
            debug_info['timestamp'] = datetime.now().isoformat()
            debug_info['config'] = {
                'model_path': args.model,
                'keywords_path': args.keywords,
                'threshold': args.threshold,
                'chunk_size': CHUNK_SIZE
            }

            # Generate output path
            if args.output:
                debug_output_path = args.output
            else:
                input_path = Path(args.input)
                debug_output_path = str(input_path.parent / f"{input_path.stem}_debug.json")

            with open(debug_output_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)

            print(f"\nDebug output saved to: {debug_output_path}")

        # Production mode: save climate rows
        if args.mode == 'prod' and climate_rows:
            output_path = generate_output_filename(args.input)
            save_climate_rows(climate_rows, output_path)

    finally:
        remove_predict_patch()


if __name__ == "__main__":
    main()
