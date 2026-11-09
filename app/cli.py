"""CLI adapter (thin): calls domain services, prints JSON/text.

Entry point: python -m app.cli <command> [options]

Commands:
  align          — normalize a blank photo (ЛР2)
  extract        — extract ROI crops from an aligned image (ЛР3)
  build-dataset  — assemble crops, labels, manifest (ЛР3)
  train          — train CNN classifier (ЛР4)

Exit codes per SKILL v4 §14:
  0 — success
  2 — domain error
  3 — I/O error
  4 — training/export error
No traceback is ever printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app.core.config import AppConfig
from app.core.errors import DomainError, CropOutOfBoundsError
from app.core.schemas import TemplateSpec
from app.core.storage import LocalStorage
from app.normalizer.service import NormalizerService
from app.roi.extractor import extract_crops
from app.roi.spec_loader import load_template_spec
from app.dataset.builder import build_dataset
from app.dataset.splits import split_photo_ids
from app.training.train import main as train_main


def _load_template_spec(template_arg, config):
    """Load TemplateSpec from a spec.json file path or template_id directory."""
    if not template_arg:
        return None
    p = Path(template_arg)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    spec_path = config.templates_dir / template_arg / "spec.json"
    if spec_path.is_file():
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    return None


def _print_json(obj):
    """Print a JSON object to stdout (compact, ensure_ascii=False)."""
    print(json.dumps(obj, ensure_ascii=False))


# ---- align subcommand ----

def cmd_align(args):
    """Handle the `align` subcommand."""
    config = AppConfig.from_env()
    storage = LocalStorage(config.data_dir)
    service = NormalizerService(config, storage)
    template_spec = _load_template_spec(args.template, config)
    corner_mode = args.corner_mode if args.corner_mode else None

    try:
        output_image, edges_image, result = service.align_file(
            args.input,
            template_spec=template_spec,
            corner_mode=corner_mode,
            save_aligned_path=args.out,
            save_edges_path=args.edges,
        )
    except DomainError as e:
        if args.json:
            _print_json({"status": "error", "code": e.code, "message": e.message})
        else:
            print(f"ERROR [{e.code}] {e.message}", file=sys.stderr)
        return 2
    except (OSError, IOError) as e:
        if args.json:
            _print_json({"status": "error", "code": "IO_ERROR", "message": str(e)})
        else:
            print(f"IO ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        if args.json:
            _print_json({"status": "error", "code": "INTERNAL_ERROR", "message": str(e)})
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 3

    out_obj = {
        "status": "ok",
        "output": str(args.out),
        "edges": str(args.edges),
        "marker_ids": result.marker_ids,
        "output_size": list(result.output_size),
        "corner_mode": result.corner_mode,
        "warnings": result.warnings,
    }
    if args.json:
        _print_json(out_obj)
    else:
        print(f"OK: {args.out} ({result.output_size[0]}x{result.output_size[1]}, mode={result.corner_mode})")
    return 0


# ---- extract subcommand ----

def cmd_extract(args):
    """Handle the `extract` subcommand: extract ROI crops from aligned image."""
    config = AppConfig.from_env()
    template_spec = _load_template_spec(args.template, config)
    if template_spec is None:
        print("ERROR: TemplateSpec required for extraction", file=sys.stderr)
        return 2

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        print(f"IO ERROR: cannot read {args.input}", file=sys.stderr)
        return 3

    try:
        crops = extract_crops(image, template_spec)
    except DomainError as e:
        if args.json:
            _print_json({"status": "error", "code": e.code, "message": e.message})
        else:
            print(f"ERROR [{e.code}] {e.message}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        if args.json:
            _print_json({"status": "error", "code": "INTERNAL_ERROR", "message": str(e)})
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for crop in crops:
        filename = f"{crop.field_id}.png"
        cv2.imwrite(str(out_dir / filename), crop.crop)

    if args.json:
        _print_json({
            "status": "ok",
            "output_dir": str(out_dir),
            "num_crops": len(crops),
            "field_ids": [c.field_id for c in crops],
        })
    else:
        print(f"OK: extracted {len(crops)} crops to {out_dir}")
    return 0


# ---- build-dataset subcommand ----

def cmd_build_dataset(args):
    """Handle the `build-dataset` subcommand: assemble crops + labels + manifest."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = build_dataset(
            crops_dir=args.crops_dir,
            labels_path=args.labels,
            output_dir=out_dir,
            template_version=args.template_version,
            photo_id=args.photo_id,
            split=args.split,
        )
    except DomainError as e:
        if args.json:
            _print_json({"status": "error", "code": e.code, "message": e.message})
        else:
            print(f"ERROR [{e.code}] {e.message}", file=sys.stderr)
        return 2
    except (OSError, IOError) as e:
        if args.json:
            _print_json({"status": "error", "code": "IO_ERROR", "message": str(e)})
        else:
            print(f"IO ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        if args.json:
            _print_json({"status": "error", "code": "INTERNAL_ERROR", "message": str(e)})
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 3

    if args.json:
        _print_json({"status": "ok", "manifest": manifest})
    else:
        print(f"OK: dataset built in {out_dir} ({manifest['total_crops']} crops)")
    return 0


def cmd_train(args):
    """Handle the `train` subcommand: train CNN classifier (ЛР4)."""
    argv = [
        "--dataset-dir", args.dataset_dir,
        "--output-dir", args.output_dir,
        "--alphabet", args.alphabet,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--input-height", str(args.input_height),
        "--input-width", str(args.input_width),
    ]
    if args.json:
        argv.append("--json")
    return train_main(argv)


def main(argv=None):
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    # align
    p_align = sub.add_parser("align", help="Normalize a blank photo")
    p_align.add_argument("input", help="Input image path")
    p_align.add_argument("--template", default=None, help="template_id or path to spec.json")
    p_align.add_argument("--out", default="aligned_blank.png", help="Output aligned image")
    p_align.add_argument("--edges", default="edges.png", help="Output edges image")
    p_align.add_argument(
        "--corner-mode",
        default=None,
        choices=["centers", "centers_to_layout", "outer_corners"],
    )
    p_align.add_argument("--json", action="store_true", help="JSON output")
    p_align.set_defaults(func=cmd_align)

    # extract
    p_extract = sub.add_parser("extract", help="Extract ROI crops from aligned image")
    p_extract.add_argument("input", help="Aligned image path")
    p_extract.add_argument("--template", required=True, help="template_id or path to spec.json")
    p_extract.add_argument("--out", default="crops", help="Output directory for crops")
    p_extract.add_argument("--json", action="store_true", help="JSON output")
    p_extract.set_defaults(func=cmd_extract)

    # build-dataset
    p_build = sub.add_parser("build-dataset", help="Build dataset from crops + labels")
    p_build.add_argument("--crops-dir", required=True, help="Directory with extracted crops")
    p_build.add_argument("--labels", required=True, help="Path to labels CSV")
    p_build.add_argument("--out", required=True, help="Output dataset directory")
    p_build.add_argument("--template-version", required=True, help="Template version string")
    p_build.add_argument("--photo-id", required=True, help="Photo identifier")
    p_build.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split label")
    p_build.add_argument("--json", action="store_true", help="JSON output")
    p_build.set_defaults(func=cmd_build_dataset)

    # train
    p_train = sub.add_parser("train", help="Train CNN classifier (ЛР4)")
    p_train.add_argument("--dataset-dir", required=True, help="Dataset directory (crops/, labels.csv, split_proof.json)")
    p_train.add_argument("--output-dir", required=True, help="Output directory for model and artifacts")
    p_train.add_argument("--alphabet", default="alphabet.json", help="Path to alphabet.json")
    p_train.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    p_train.add_argument("--input-height", type=int, default=64, help="Input image height")
    p_train.add_argument("--input-width", type=int, default=128, help="Input image width")
    p_train.add_argument("--json", action="store_true", help="JSON output")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())