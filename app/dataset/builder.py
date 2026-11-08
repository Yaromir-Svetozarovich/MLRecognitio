"""Dataset builder: assemble crops, labels.csv, and dataset_manifest.json.

Reads extracted crops and labels, writes a structured dataset directory
with train/val/test splits and a manifest for reproducibility.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.errors import AnswerKeyMismatchError
from app.core.schemas import AnswerKey


def build_dataset(
    crops_dir: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    template_version: str,
    photo_id: str,
    answer_key: AnswerKey | None = None,
    split: str = "train",
) -> dict[str, Any]:
    """Build a dataset from extracted crops and labels.

    Creates the output directory structure:
        output_dir/
          crops/
            <photo_id>_<field_id>.png
          labels.csv          (field_id, crop_filename, label, photo_id, split)
          dataset_manifest.json

    Args:
        crops_dir: directory containing extracted crop PNGs.
        labels_path: path to labels CSV (field_id,label).
        output_dir: where to write the dataset.
        template_version: template version string for the manifest.
        photo_id: unique photo identifier.
        answer_key: optional AnswerKey for cross-checking labels.
        split: train/val/test label for this batch.

    Returns:
        The dataset manifest dict.
    """
    crops_dir = Path(crops_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_out = output_dir / "crops"
    crops_out.mkdir(parents=True, exist_ok=True)

    # Read labels from CSV if it exists
    labels = {}
    labels_file = Path(labels_path)
    if labels_file.is_file():
        with open(labels_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["field_id"]] = row.get("label", "")

    # Cross-check with AnswerKey if provided
    if answer_key is not None:
        if answer_key.template_version != template_version:
            raise AnswerKeyMismatchError(
                details={"answer_key_version": answer_key.template_version, "template_version": template_version}
            )
        for fid in answer_key.answers:
            if fid not in labels:
                labels[fid] = answer_key.answers[fid]

    # Copy crops and build labels.csv
    labels_csv = output_dir / "labels.csv"
    with open(labels_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field_id", "crop_filename", "label", "photo_id", "split"])
        for crop_file in sorted(crops_dir.glob("*.png")):
            # field_id is the stem of the crop filename (e.g. "q1_answer" from "q1_answer.png")
            field_id = crop_file.stem
            label = labels.get(field_id, "")
            dest = crops_out / f"{photo_id}_{crop_file.name}"
            if not dest.exists():
                dest.write_bytes(crop_file.read_bytes())
            writer.writerow([field_id, dest.name, label, photo_id, split])

    # Build manifest
    manifest = {
        "template_version": template_version,
        "photo_id": photo_id,
        "created_at": datetime.now().isoformat(),
        "total_crops": len(list(crops_out.glob("*.png"))),
        "split": split,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest