"""
CSV sampling utilities for newspaper data.
Converts CSV files to iterator format compatible with GPTLabeler.
"""

import csv
import itertools
import logging
import random
import sys
from pathlib import Path
from typing import Iterator, List

logger = logging.getLogger(__name__)

# Increase CSV field size limit for large newspaper texts
csv.field_size_limit(sys.maxsize)


def iter_csv_samples(
    csv_path: str,
    num_samples: int,
    text_column: str = 'Text',
    date_column: str = 'Date',
    seed: int = 42
) -> Iterator[dict]:
    """
    Randomly sample rows from CSV and yield as dicts.

    Args:
        csv_path: Path to CSV file
        num_samples: Number of random samples to draw
        text_column: Name of text column
        date_column: Name of date column (for ID generation)
        seed: Random seed for reproducibility

    Yields:
        Dict with 'text', 'id', 'date' fields

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If num_samples exceeds available rows
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read all rows into memory
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        # Validate columns exist
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV file: {csv_path}")

        if text_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{text_column}' not found in CSV. "
                f"Available columns: {reader.fieldnames}"
            )

        if date_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{date_column}' not found in CSV. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            rows.append(row)

    total_rows = len(rows)
    logger.info(f"Loaded {total_rows} rows from {csv_path.name}")

    # Validate num_samples
    if num_samples > total_rows:
        raise ValueError(
            f"Requested {num_samples} samples but CSV only has {total_rows} rows"
        )

    # Sample rows with reproducible randomness
    random.seed(seed)
    sampled_rows = random.sample(rows, num_samples)

    logger.info(f"Sampling {num_samples} rows from {csv_path.name}")

    # Generate IDs and yield
    csv_basename = csv_path.stem
    for idx, row in enumerate(sampled_rows):
        text = row[text_column]
        date = row.get(date_column, 'unknown')

        # Create unique ID: basename_date_idx
        # Clean date to remove slashes/spaces for ID
        clean_date = date.replace('/', '-').replace(' ', '_')
        sample_id = f"{csv_basename}_{clean_date}_{idx}"

        yield {
            'text': text,
            'id': sample_id,
            'date': date
        }


def iter_combined_samples(
    csv_paths: List[str],
    samples_per_csv: int,
    text_column: str = 'Text',
    date_column: str = 'Date',
    seed: int = 42
) -> Iterator[dict]:
    """
    Sample from multiple CSVs and combine into single iterator.

    Args:
        csv_paths: List of CSV file paths
        samples_per_csv: Number of samples to draw from each CSV
        text_column: Name of text column
        date_column: Name of date column
        seed: Random seed for reproducibility

    Yields:
        Dict with 'text', 'id', 'date', 'source' fields

    Raises:
        FileNotFoundError: If any CSV file doesn't exist
        ValueError: If samples_per_csv exceeds available rows in any CSV
    """
    iterators = []
    total_samples = len(csv_paths) * samples_per_csv

    logger.info(f"Combining samples from {len(csv_paths)} CSV files")
    logger.info(f"Total samples to yield: {total_samples}")

    for csv_path in csv_paths:
        csv_path_obj = Path(csv_path)
        source_name = csv_path_obj.stem

        # Get iterator for this CSV
        csv_iterator = iter_csv_samples(
            csv_path=csv_path,
            num_samples=samples_per_csv,
            text_column=text_column,
            date_column=date_column,
            seed=seed
        )

        # Add 'source' field to each sample
        def add_source(iterator, source):
            for sample in iterator:
                sample['source'] = source
                yield sample

        iterators.append(add_source(csv_iterator, source_name))

    # Chain all iterators together
    combined = itertools.chain(*iterators)

    logger.info(f"Combined iterator ready with {total_samples} samples")

    return combined
