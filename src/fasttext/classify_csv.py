#!/usr/bin/env python3
"""
FastText Climate Filter for CSV Files

Pipeline: Regex CSV → Keyword Filter → FastText Filter

Usage:
    # Count mode - just show distribution
    python src/fasttext/classify_csv.py -i datasets/historical_weather_keyword_regex.csv --mode count

    # Debug mode - show distribution + chunk details for sample rows
    python src/fasttext/classify_csv.py -i datasets/historical_weather_keyword_regex.csv --mode debug --sample 10

    # E2E mode - save climate rows to new CSV
    python src/fasttext/classify_csv.py -i datasets/historical_weather_keyword_regex.csv --mode e2e --threshold 0.8

    # Process multiple files
    python src/fasttext/classify_csv.py -i datasets/historical_weather_keyword_regex.csv datasets/modern_weather_keyword_regex.csv --mode count --threshold 0.8
"""

import argparse
import csv
import json
import random
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
DEFAULT_THRESHOLD = 0.5


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

    def __init__(self, model_path: str = MODEL_PATH, threshold: float = DEFAULT_THRESHOLD):
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


# =============================================================================
# CSV Processing
# =============================================================================
def get_csv_columns(csv_path: str) -> Tuple[str, str]:
    """Find text and date columns in CSV."""
    df = pd.read_csv(csv_path, nrows=1)

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

    return text_column, date_column


def _count_csv_rows(csv_path: str) -> int:
    """Count rows in CSV without loading into memory."""
    count = -1  # skip header
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in f:
            count += 1
    return max(count, 0)


def iter_csv_rows(csv_path: str, num_rows: int = None) -> Iterator[dict]:
    """Iterate over CSV rows. Uses chunked reading to avoid OOM on large files."""
    text_column, date_column = get_csv_columns(csv_path)

    if text_column is None:
        print(f"  WARNING: No text column found.")
        return

    CHUNK_ROWS = 50  # rows per pandas chunk (rows can be ~1MB each)

    if num_rows is not None:
        # Pick random row indices without loading entire file
        total = _count_csv_rows(csv_path)
        sample_n = min(num_rows, total)
        sample_indices = set(random.Random(42).sample(range(total), sample_n))
        print(f"  Sampling {sample_n} rows from {total} total (chunked read)")

        row_num = 0
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS):
            for idx, row in chunk.iterrows():
                if row_num in sample_indices:
                    yield {
                        'index': idx,
                        'text': str(row.get(text_column, '')) if text_column else '',
                        'date': str(row.get(date_column, '')) if date_column else '',
                        '_row': row
                    }
                row_num += 1
    else:
        # Stream all rows in chunks
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS):
            for idx, row in chunk.iterrows():
                yield {
                    'index': idx,
                    'text': str(row.get(text_column, '')) if text_column else '',
                    'date': str(row.get(date_column, '')) if date_column else '',
                    '_row': row
                }


# Track first row for debug (reset for each file)
_debug_file_printed = {}
_debug_json = []  # Structured debug data for JSON output


def keyword_context_snippets(text: str, pattern: re.Pattern, ctx: int = 20) -> Tuple[list, str]:
    """Find keywords and return (keyword_list, context_string).
    Context string uses the 【KEYWORD】 snippet format with surrounding chars."""
    snippets = []
    seen_keywords = set()
    for m in pattern.finditer(text):
        kw = m.group(0)
        seen_keywords.add(kw.lower())
        start = max(0, m.start() - ctx)
        end = min(len(text), m.end() + ctx)
        before = text[start:m.start()]
        after = text[m.end():end]
        snippets.append(f"{before}【{kw.upper()}】{after}")
    context_str = " ... ".join(snippets) if snippets else ""
    return sorted(seen_keywords), context_str


def classify_row(text: str, classifier: FastTextClimateClassifier, keyword_filter, file_key: str = "default") -> Tuple[bool, float, dict]:
    """Classify a single row using chunk + keyword strategy."""
    # === CRITICAL DEBUG: Print ALL chunks for first article ===
    if file_key not in _debug_file_printed and keyword_filter:
        chunks = split_text_to_chunks(text, CHUNK_SIZE)
        article_debug = {
            "file": file_key,
            "article_length_chars": len(text),
            "article_length_words": len(text.split()),
            "num_chunks": len(chunks),
            "chunks": []
        }
        print(f"\n  [DEBUG] FIRST ARTICLE - ALL CHUNKS ({file_key}):")
        print(f"  [DEBUG] Article length: {len(text)} chars, {len(text.split())} words, {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            prob = classifier.predict_proba(chunk)
            found_keywords, context_str = keyword_context_snippets(chunk, keyword_filter.pattern)

            chunk_debug = {
                "chunk": i + 1,
                "prob": round(prob, 4),
                "keywords": found_keywords,
                "keyword_text": context_str if found_keywords else None,
                "text": chunk
            }
            article_debug["chunks"].append(chunk_debug)

            if found_keywords:
                print(f"    Chunk {i+1}: prob={prob:.4f} | KEYWORDS: {found_keywords}")
                print(f"      Keyword Text: {context_str}")
            else:
                print(f"    Chunk {i+1}: prob={prob:.4f} | NO KEYWORDS")

        _debug_json.append(article_debug)
        _debug_file_printed[file_key] = True
    # =======================================================================================================

    # Now do actual classification

    chunks = split_text_to_chunks(text, CHUNK_SIZE)

    if not chunks:
        return False, 0.0, {}

    # Track all chunks for debug
    all_chunks = []
    max_prob = 0.0
    climate_chunks = []

    for i, chunk in enumerate(chunks):
        is_climate, prob = classifier.is_climate(chunk)

        chunk_info = {
            'chunk_index': i + 1,
            'probability': prob,
            'is_climate': is_climate,
            'text': chunk
        }
        all_chunks.append(chunk_info)

        # Update max_prob for ALL chunks (not just climate)
        if prob > max_prob:
            max_prob = prob

        if is_climate:
            climate_chunks.append(chunk_info)

    is_climate_final = max_prob >= classifier.threshold

    debug_info = {
        'total_chunks': len(chunks),
        'climate_chunks': len(climate_chunks),
        'max_probability': max_prob,
        'is_climate': is_climate_final,
        'all_chunks': all_chunks[:10] if len(all_chunks) > 10 else all_chunks  # Limit for output
    }

    return is_climate_final, max_prob, debug_info


def process_csv(
    csv_path: str,
    num_rows: int,
    threshold: float,
    classifier: FastTextClimateClassifier,
    keyword_filter,
    mode: str,
    sample_size: int = 10
) -> dict:
    """Process CSV and return results."""
    climate_count = 0
    other_count = 0
    climate_rows = []
    debug_articles = []

    for row in iter_csv_rows(csv_path, num_rows):
        text = row['text']
        is_climate, prob, debug_info = classify_row(text, classifier, keyword_filter, file_key=csv_path)

        if is_climate:
            climate_count += 1
            if mode == 'e2e':
                climate_rows.append(row['_row'])
        else:
            other_count += 1

        # Collect debug info for sample
        if mode == 'debug' and len(debug_articles) < sample_size:
            debug_articles.append({
                'article_index': row['index'],
                'date': row['date'],
                'probability': prob,
                **debug_info
            })

    return {
        'climate_count': climate_count,
        'other_count': other_count,
        'climate_rows': climate_rows,
        'debug_articles': debug_articles,
        'total': climate_count + other_count
    }


def process_file(csv_path: str, args) -> dict:
    """Process a single CSV file."""
    threshold = args.threshold

    # Override threshold if provided
    if args.threshold:
        threshold = args.threshold

    classifier = FastTextClimateClassifier(
        model_path=args.model,
        threshold=threshold
    )
    keyword_filter = KeywordFilter(keywords_path=args.keywords)

    print(f"\n{'='*60}")
    print(f"Processing: {csv_path}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")

    result = process_csv(
        csv_path=csv_path,
        num_rows=args.rows,
        threshold=threshold,
        classifier=classifier,
        keyword_filter=keyword_filter,
        mode=args.mode,
        sample_size=args.sample
    )

    # Output results
    total = result['total']
    climate_pct = (result['climate_count'] / total * 100) if total > 0 else 0

    print(f"\n--- Results ---")
    print(f"Total rows: {total}")
    print(f"Climate (prob >= {threshold}): {result['climate_count']} ({climate_pct:.1f}%)")
    print(f"Non-climate: {result['other_count']} ({100-climate_pct:.1f}%)")

    if args.mode == 'debug' and result['debug_articles']:
        print(f"\n--- Sample Debug Info ({len(result['debug_articles'])} articles) ---")
        for i, article in enumerate(result['debug_articles'][:3]):
            print(f"\nArticle {i+1} (index={article['article_index']}):")
            print(f"  Probability: {article['probability']:.4f}")
            print(f"  Total chunks: {article['total_chunks']}")
            print(f"  Climate chunks: {article['climate_chunks']}")

            if article['climate_chunks'] > 0:
                print(f"  Climate chunk details:")
                for chunk in article['all_chunks'][:3]:
                    if chunk['is_climate']:
                        print(f"    Chunk {chunk['chunk_index']}: prob={chunk['probability']:.4f}")

    if args.mode == 'e2e' and result['climate_rows']:
        # Generate output filename
        input_path = Path(csv_path)
        output_path = input_path.parent / f"{input_path.stem}_fasttext_{int(threshold*100)}.csv"

        # Write climate rows
        df = pd.DataFrame(result['climate_rows'])
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(result['climate_rows'])} climate rows to: {output_path}")

    return result


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FastText Climate Filter for CSV Files"
    )
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='Input CSV file path(s)'
    )
    parser.add_argument(
        '--rows', '-n',
        type=int,
        default=None,
        help='Number of rows to process (default: all)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['count', 'debug', 'e2e'],
        default='count',
        help='Mode: count=distribution only, debug=+chunk details, e2e=save climate rows'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'FastText probability threshold (default: {DEFAULT_THRESHOLD})'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=10,
        help='Number of articles to show in debug mode (default: 10)'
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

    args = parser.parse_args()

    print("=" * 60)
    print("FastText Climate Filter")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}")
    print(f"Model: {args.model}")

    # Apply NumPy 2.x patch
    apply_predict_patch()

    try:
        total_climate = 0
        total_other = 0

        for csv_path in args.input:
            result = process_file(csv_path, args)
            total_climate += result['climate_count']
            total_other += result['other_count']

        # Summary for multiple files
        if len(args.input) > 1:
            print(f"\n{'='*60}")
            print("OVERALL SUMMARY")
            print(f"{'='*60}")
            total_all = total_climate + total_other
            print(f"Total files: {len(args.input)}")
            print(f"Total climate: {total_climate} ({total_climate/total_all*100:.1f}%)")
            print(f"Total non-climate: {total_other} ({total_other/total_all*100:.1f}%)")

    finally:
        remove_predict_patch()

    # Write debug output to JSON
    if _debug_json:
        debug_path = Path("debug_output.json")
        # Load existing data or start fresh
        existing = []
        if debug_path.exists():
            with open(debug_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        # Append this run
        run_entry = {
            "timestamp": datetime.now().isoformat(),
            "threshold": args.threshold,
            "articles": _debug_json
        }
        existing.append(run_entry)
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"\n[DEBUG] Output written to {debug_path}")


if __name__ == "__main__":
    main()
