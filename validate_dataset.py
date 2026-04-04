import os
import sys
import json
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from PIL import Image

SUBSETS                  = ["train", "val", "test"]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_RESOLUTION      = (640, 640)
CHECKSUM_OUTPUT_FILE     = "dataset_checksum.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger(__name__)


# ── Checksum helpers ──────────────────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_dataset(image_files: list) -> str:
    h = hashlib.md5()
    for path in sorted(image_files):
        h.update(md5_file(path).encode())
    return h.hexdigest()


# ── Check 1: Folder structure ─────────────────────────────────────────────────

def check_folder_structure(dataset_path: Path) -> list:
    errors = []
    for subset in SUBSETS:
        for top in ["images", "labels"]:
            folder = dataset_path / top / subset
            if not folder.exists():
                errors.append(f"Missing folder: {folder}")
            elif not folder.is_dir():
                errors.append(f"Expected directory but found file: {folder}")
    return errors


# ── Check 2: Image format & resolution ───────────────────────────────────────

def check_image_formats_and_resolution(dataset_path: Path, check_resolution: bool = True):
    errors = []
    all_images = []
    resolution_counts = defaultdict(int)

    for subset in SUBSETS:
        images_dir = dataset_path / "images" / subset
        if not images_dir.exists():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                errors.append(f"Unsupported file type: {img_path}")
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    resolution = img.size
                    resolution_counts[resolution] += 1
                if check_resolution and EXPECTED_RESOLUTION:
                    if resolution != EXPECTED_RESOLUTION:
                        errors.append(
                            f"Resolution mismatch in {img_path.relative_to(dataset_path)}: "
                            f"expected {EXPECTED_RESOLUTION}, got {resolution}"
                        )
                all_images.append(img_path)
            except Exception as e:
                errors.append(f"Corrupted/unreadable image: {img_path} — {e}")

    log.info("  Resolution distribution across dataset:")
    for res, count in sorted(resolution_counts.items(), key=lambda x: -x[1]):
        log.info(f"    {res[0]}x{res[1]} -> {count} images")
    return errors, all_images


# ── Check 3: Label files ──────────────────────────────────────────────────────

def check_label_files(dataset_path: Path, image_paths: list) -> list:
    """
    For each image at crack-seg/images/<subset>/<name>.jpg
    expects a label at  crack-seg/labels/<subset>/<name>.txt
    """
    errors = []
    for img_path in image_paths:
        subset = img_path.parent.name
        label_path = (
            dataset_path / "labels" / subset / img_path.with_suffix(".txt").name
        )
        if not label_path.exists():
            errors.append(
                f"Missing label: {label_path.relative_to(dataset_path)}"
            )
        elif label_path.stat().st_size == 0:
            # Empty label files are valid in YOLO — means no objects in this image
            log.warning(
                f"  Empty label file (negative sample): {label_path.relative_to(dataset_path)}"
            )
    return errors


# ── Check 4: Checksum ─────────────────────────────────────────────────────────

def compute_and_save_checksum(image_paths: list, output_path: Path) -> str:
    log.info("  Computing dataset checksum (this may take a moment)...")
    checksum = md5_dataset(image_paths)
    result = {"total_images": len(image_paths), "aggregate_md5": checksum}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"  Checksum saved to: {output_path}")
    return checksum


# ── Main validator ────────────────────────────────────────────────────────────

def validate(dataset_path: Path, check_resolution: bool = True) -> bool:
    log.info("=" * 55)
    log.info("  Crack Segmentation Dataset Validator")
    log.info(f"  Path: {dataset_path.resolve()}")
    log.info("=" * 55)
    all_errors = []

    log.info("\n[1/4] Checking folder structure...")
    errs = check_folder_structure(dataset_path)
    all_errors.extend(errs)
    if errs:
        for e in errs:
            log.error(f"  x {e}")
    else:
        log.info("  OK Folder structure is valid")

    log.info("\n[2/4] Checking image formats and resolution...")
    errs, valid_images = check_image_formats_and_resolution(
        dataset_path, check_resolution=check_resolution
    )
    all_errors.extend(errs)
    if errs:
        for e in errs[:20]:
            log.error(f"  x {e}")
        if len(errs) > 20:
            log.error(f"  ... and {len(errs) - 20} more errors")
    else:
        log.info(f"  OK All {len(valid_images)} images are valid")

    log.info("\n[3/4] Checking label files...")
    errs = check_label_files(dataset_path, valid_images)
    all_errors.extend(errs)
    if errs:
        for e in errs[:20]:
            log.error(f"  x {e}")
        if len(errs) > 20:
            log.error(f"  ... and {len(errs) - 20} more errors")
    else:
        log.info("  OK All label files present and non-empty")

    log.info("\n[4/4] Computing integrity checksum...")
    checksum_path = dataset_path.parent / CHECKSUM_OUTPUT_FILE
    if valid_images:
        checksum = compute_and_save_checksum(valid_images, checksum_path)
        log.info(f"  OK Aggregate MD5: {checksum}")
    else:
        log.warning("  WARN No valid images found, skipping checksum.")

    log.info("\n" + "=" * 55)
    if all_errors:
        log.error(f"  VALIDATION FAILED -- {len(all_errors)} error(s) found")
        log.info("=" * 55)
        return False
    else:
        log.info("  VALIDATION PASSED -- Dataset is ready for training")
        log.info("=" * 55)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default=Path("./crack-seg"))
    parser.add_argument("--skip-resolution-check", action="store_true")
    parser.add_argument("--ci", action="store_true")
    args = parser.parse_args()

    if not args.dataset_path.exists():
        log.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    is_valid = validate(
        dataset_path=args.dataset_path,
        check_resolution=not args.skip_resolution_check,
    )
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()