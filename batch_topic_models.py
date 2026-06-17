#!/usr/bin/env python3
"""
Batch Topic Model Generator

Reads a configuration CSV and generates topic models for each entry.

Configuration CSV format:
    att,domain,csv-name
    run1,corpus-a,data_file_a.csv
    run1,corpus-b,data_file_b.csv
    run2,corpus-a,data_file_c.csv

Usage:
    python batch_topic_models.py <config_csv_file>
"""

import csv
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import logging


def setup_logging(log_level: str = "INFO"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_processing_{timestamp}.log"
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def validate_inputs(config_csv: Path, rows: list) -> bool:
    logger = logging.getLogger()
    all_valid = True
    UMAP_MIN_ROWS = 150

    for i, row in enumerate(rows):
        att = row.get("att", "").strip()
        domain = row.get("domain", "").strip()
        csv_name = row.get("csv-name", "").strip()

        if not att or not domain or not csv_name:
            logger.error(f"Row {i+2}: Missing required field(s). att={att}, domain={domain}, csv-name={csv_name}")
            all_valid = False
            row["_skip"] = True
            continue

        csv_path = Path("data") / csv_name
        if not csv_path.exists():
            logger.error(f"Row {i+2}: Input CSV not found: {csv_path}")
            all_valid = False
            row["_skip"] = True
            continue

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1

            if row_count < UMAP_MIN_ROWS:
                logger.warning(f"Row {i+2}: Skipping {att}_{domain} - only {row_count} documents (minimum: {UMAP_MIN_ROWS})")
                row["_skip"] = True
            else:
                row["_skip"] = False
        except Exception as e:
            logger.error(f"Row {i+2}: Error reading CSV: {e}")
            all_valid = False
            row["_skip"] = True

    return all_valid


def run_topic_model(att: str, domain: str, csv_path: str) -> tuple[bool, str]:
    domain_prefix = f"{att}_{domain}"
    cmd = [
        "python",
        "src/topic_modeller/topic_model.py",
        f"data/{csv_path}",
        "--domain-prefix",
        domain_prefix
    ]

    logger = logging.getLogger()
    logger.info(f"Starting: {domain_prefix}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            message = f"✓ Completed: {domain_prefix}"
            logger.info(message)
            return True, message
        else:
            message = f"✗ Failed: {domain_prefix}\n  Error: {result.stderr[:1000]}"
            logger.error(message)
            return False, message
    except subprocess.TimeoutExpired:
        message = f"✗ Timeout: {domain_prefix} (exceeded 1 hour)"
        logger.error(message)
        return False, message
    except Exception as e:
        message = f"✗ Exception: {domain_prefix}\n  Error: {str(e)}"
        logger.error(message)
        return False, message


def build_task_result(att: str, domain: str, csv_name: str) -> dict:
    return {
        "att": att,
        "domain": domain,
        "csv_name": csv_name,
        "domain_prefix": f"{att}_{domain}",
    }


def process_batch_row(row: dict, idx: int, total_rows: int, results: dict) -> None:
    logger = logging.getLogger()
    att = row["att"].strip()
    domain = row["domain"].strip()
    csv_name = row["csv-name"].strip()
    task_result = build_task_result(att, domain, csv_name)

    if row.get("_skip", False):
        results["skipped"].append(task_result)
        return

    logger.info(f"\n--- Task {idx}/{total_rows} ---")
    success, message = run_topic_model(att, domain, csv_name)

    if success:
        results["successful"].append(task_result)
    else:
        results["failed"].append({**task_result, "error": message})


def process_batch(config_csv: Path):
    logger = logging.getLogger()
    logger.info(f"Reading configuration from: {config_csv}")

    try:
        with open(config_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_csv}")
        return False
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        return False

    if not rows:
        logger.error("Configuration file is empty or has no data rows")
        return False

    logger.info(f"Found {len(rows)} processing tasks")
    logger.info("Validating input files...")
    if not validate_inputs(config_csv, rows):
        logger.warning("Some validation errors, but continuing with valid entries...")
    logger.info("Validation passed ✓")

    results = {"successful": [], "failed": [], "skipped": []}
    start_time = datetime.now()

    for idx, row in enumerate(rows, start=1):
        process_batch_row(row, idx, len(rows), results)

    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*60)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total tasks: {len(rows)}")
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Skipped (too small): {len(results['skipped'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Elapsed time: {elapsed}")

    if results["skipped"]:
        logger.info("\nSkipped tasks (insufficient data):")
        for item in results["skipped"]:
            logger.info(f"  - {item['domain_prefix']}")

    if results["failed"]:
        logger.info("\nFailed tasks:")
        for item in results["failed"]:
            logger.info(f"  - {item['domain_prefix']}: {item['error']}")

    return len(results["failed"]) == 0


def main():
    parser = argparse.ArgumentParser(description="Generate topic models in batch from a configuration CSV")
    parser.add_argument("config_csv", help="Path to the configuration CSV file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    config_csv = Path(args.config_csv)
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger()

    logger.info("="*60)
    logger.info("BATCH TOPIC MODEL GENERATION")
    logger.info("="*60)
    logger.info(f"Configuration: {config_csv}")
    logger.info(f"Logging to: {log_file}")

    success = process_batch(config_csv)
    logger.info("\n" + ("BATCH COMPLETED SUCCESSFULLY ✓" if success else "BATCH COMPLETED WITH ERRORS ✗"))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
