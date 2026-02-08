#!/usr/bin/env python3
"""
Extract keyword-matching rows from CSV and add highlighted context column.

Usage:
    python scripts/extract_keyword_rows.py \
      --input datasets/historical_regex.csv \
      --output datasets/historical_weather_keyword_regex.csv \
      --keywords data/weather_terms.txt
"""

import argparse
import csv
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def load_keywords(keywords_path: str) -> tuple[set, re.Pattern]:
    """Load keywords from file and build regex pattern."""
    keywords = set()
    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and lines with parentheses (inline comments)
            if line and not line.startswith('#') and '(' not in line:
                keywords.add(line.lower())

    # Build regex pattern with word boundaries
    escaped = [re.escape(kw) for kw in keywords]
    pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)

    return keywords, pattern


def highlight_keywords(text: str, pattern: re.Pattern) -> str:
    """Add 【KEYWORD】 markers around found keywords."""
    matches = set()
    for m in pattern.finditer(text):
        matches.add(m.group(0).lower())

    highlighted = text
    for kw in matches:
        # Replace with highlighted version (case-preserving)
        pattern_kw = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        highlighted = pattern_kw.sub(f'【{kw.upper()}】', highlighted)

    return highlighted


def main():
    parser = argparse.ArgumentParser(
        description='Extract keyword-matching rows and add highlighted context'
    )
    parser.add_argument('--input', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--keywords', '-k', default='data/weather_terms.txt',
                        help='Keywords file path')

    args = parser.parse_args()

    print(f"Loading keywords from {args.keywords}...")
    _, pattern = load_keywords(args.keywords)
    print(f"Pattern ready.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    matched_count = 0
    total_count = 0

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = ['context'] + reader.fieldnames  # Add context as first column

        with open(output_path, 'w', encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_count += 1
                text = row.get('Text', '')

                # Check if text contains any keyword
                if pattern.search(text):
                    matched_count += 1
                    # Add highlighted context as first column
                    context = highlight_keywords(text[:500], pattern)  # First 500 chars
                    row['context'] = context
                    writer.writerow(row)

    print(f"\nDone!")
    print(f"Total rows: {total_count}")
    print(f"Matched (keyword found): {matched_count}")
    print(f"Output: {output_path}")
    print(f"Output is strict subset of input with 'context' column added.")


if __name__ == "__main__":
    main()
