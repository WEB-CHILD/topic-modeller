#!/usr/bin/env python3
"""
Batch visualisation runner

Reads a configuration CSV (same format as batch_topic_models.py) and runs
visualise_topics.py for each entry where topic model outputs exist.

Usage:
    python batch_visualise.py <config_csv>
"""

import csv
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging
import shutil


def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_visualise_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return log_file


def run_visualisation_for(prefix: str, timeout: int = 1800) -> tuple[bool, str]:
    logger = logging.getLogger()
    cmd = [
        "python",
        "src/topic_modeller/visualise_topics.py",
        "--domain-prefix",
        prefix
    ]
    logger.info(f"Running visualisations for: {prefix}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            logger.info(f"✓ Visualisations completed: {prefix}")
            return True, res.stdout
        else:
            logger.error(f"✗ Visualisation failed: {prefix}\n{res.stderr[:1000]}")
            return False, res.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Visualisation timeout: {prefix} (>{timeout}s)")
        return False, "timeout"
    except Exception as e:
        logger.error(f"✗ Visualisation exception: {prefix}: {e}")
        return False, str(e)


def organize_outputs(prefix: str, att: str, domain: str) -> list:
    """Move generated visualisation files into visualisations/<att>/<domain>/."""
    logger = logging.getLogger()
    moved = []
    src_root = Path("visualisations")
    target_dir = src_root / att / domain
    target_dir.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        return moved

    for p in src_root.rglob(f"*{prefix}*"):
        if p.is_file():
            dest = target_dir / p.name
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                i = 1
                while True:
                    new_dest = target_dir / f"{stem}_{i}{suffix}"
                    if not new_dest.exists():
                        dest = new_dest
                        break
                    i += 1
            try:
                shutil.move(str(p), str(dest))
                moved.append(str(dest))
            except Exception as e:
                logger.warning(f"Failed to move {p} -> {dest}: {e}")

    for p in Path('.').glob(f"*{prefix}*"):
        if p.is_file():
            dest = target_dir / p.name
            try:
                shutil.move(str(p), str(dest))
                moved.append(str(dest))
            except Exception as e:
                logger.warning(f"Failed to move {p} -> {dest}: {e}")

    return moved


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_visualise.py <config_csv>")
        sys.exit(1)

    config_csv = Path(sys.argv[1])
    log_file = setup_logging()
    logger = logging.getLogger()

    logger.info("Batch visualisation started")
    logger.info(f"Configuration: {config_csv}")
    logger.info(f"Logging to: {log_file}")

    try:
        with open(config_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        logger.error(f"Failed to read config CSV: {e}")
        sys.exit(2)

    results = {"successful": [], "failed": [], "skipped": []}

    for idx, row in enumerate(rows, start=1):
        att = row.get('att', '').strip()
        domain = row.get('domain', '').strip()
        if not att or not domain:
            logger.warning(f"Skipping row {idx}: missing att or domain")
            results['skipped'].append((idx, row))
            continue

        prefix = f"{att}_{domain}"
        result_json = Path(f"data/{prefix}_topic_model_results.json")
        model_dir = Path(f"models/{prefix}_topic_model_bertopic")

        if not (result_json.exists() and model_dir.exists()):
            logger.warning(f"Skipping {prefix}: missing results/model")
            results['skipped'].append(prefix)
            continue

        success, msg = run_visualisation_for(prefix)
        if success:
            results['successful'].append(prefix)
            moved = organize_outputs(prefix, att, domain)
            if moved:
                logger.info(f"Moved {len(moved)} files to visualisations/{att}/{domain}/")
        else:
            results['failed'].append({"prefix": prefix, "error": str(msg)})

    logger.info("\n" + "="*60)
    logger.info("BATCH VISUALISATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total entries: {len(rows)}")
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results['failed']:
        logger.info("\nFailed items:")
        for it in results['failed']:
            logger.info(f" - {it['prefix']}: {it['error']}")

    logger.info("Batch visualisation finished")
    sys.exit(0 if not results['failed'] else 1)


if __name__ == '__main__':
    main()
