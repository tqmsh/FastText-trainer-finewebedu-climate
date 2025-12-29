"""
CSV climate filter - routes rows to climate/no_climate files.
"""
import csv
import re
import sys
from pathlib import Path

import fasttext

fasttext.FastText.eprint = lambda x: None

csv.field_size_limit(sys.maxsize)


def load_keywords(path: str) -> re.Pattern:
    keywords = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                keywords.add(line.lower())
    escaped = [re.escape(kw) for kw in keywords]
    pattern = re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    return pattern


def load_fasttext(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return fasttext.load_model(str(path))


def find_matched_keywords(text: str, pattern: re.Pattern) -> list[str]:
    matches = set()
    for m in pattern.finditer(text):
        matches.add(m.group(0).lower())
    return sorted(matches)


def fasttext_predict(model, text: str) -> tuple[str, float]:
    cleaned = text.replace('\n', ' ').strip()
    if not cleaned:
        return 'other', 0.0
    try:
        labels, probs = model.predict(cleaned, k=2)
        probs = [float(p) for p in probs]
        for i, label in enumerate(labels):
            if label == '__label__climate':
                return 'climate', probs[i]
        return 'other', probs[0] if probs else 0.0
    except Exception:
        return 'other', 0.0


def run_e2e(
    input_path: str,
    climate_path: str,
    no_climate_path: str,
    keywords_path: str,
    model_path: str,
    threshold: float = 0.5
):
    pattern = load_keywords(keywords_path)
    model = load_fasttext(model_path)

    climate_rows = []
    no_climate_rows = []

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('Text', '')
            label, prob = fasttext_predict(model, text)

            if label == 'climate' and prob >= threshold:
                climate_rows.append(row)
            else:
                no_climate_rows.append(row)

    with open(climate_path, 'w', encoding='utf-8', newline='') as f:
        if climate_rows:
            writer = csv.DictWriter(f, fieldnames=climate_rows[0].keys())
            writer.writeheader()
            writer.writerows(climate_rows)

    with open(no_climate_path, 'w', encoding='utf-8', newline='') as f:
        if no_climate_rows:
            writer = csv.DictWriter(f, fieldnames=no_climate_rows[0].keys())
            writer.writeheader()
            writer.writerows(no_climate_rows)

    print(f"E2E complete: {len(climate_rows)} climate, {len(no_climate_rows)} no_climate")


def run_mock(
    input_path: str,
    keywords_path: str,
    model_path: str,
    limit: int = 100,
    threshold: float = 0.5
):
    pattern = load_keywords(keywords_path)
    model = load_fasttext(model_path)

    print("line_number,matched_keywords,fasttext_prob,decision")

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if i > limit:
                break
            text = row.get('Text', '')
            matched = find_matched_keywords(text, pattern)
            label, prob = fasttext_predict(model, text)
            kw_str = '|'.join(matched) if matched else ''
            decision = 'CLIMATE' if (label == 'climate' and prob >= threshold) else 'NO_CLIMATE'
            print(f"{i},{kw_str},{prob:.4f},{decision}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Filter CSV by climate content')
    parser.add_argument('--mode', choices=['e2e', 'mock'], required=True)
    parser.add_argument('--input', default='datasets/historical_regex_cleaned.csv')
    parser.add_argument('--climate-output', default='datasets/historical_climate_regex.csv')
    parser.add_argument('--no-climate-output', default='datasets/historical_no_climate_regex.csv')
    parser.add_argument('--keywords', default='data/keywords.txt')
    parser.add_argument('--model', default='models/fasttext_climate.bin')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=100, help='Rows for mock mode')

    args = parser.parse_args()

    if args.mode == 'e2e':
        run_e2e(
            args.input,
            args.climate_output,
            args.no_climate_output,
            args.keywords,
            args.model,
            args.threshold
        )
    else:
        run_mock(
            args.input,
            args.keywords,
            args.model,
            args.limit,
            args.threshold
        )
