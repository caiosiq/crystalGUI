#!/usr/bin/env python3
"""Organize crystallization microscopy datasets for the Outputs tab.

- Extracts 06032026-Crystallization.zip into clean folder names
- Skips wrong / bad sampling folders
- Splits insulin images into x10/ and x20/ subfolders when filenames indicate magnification
- Renames legacy dataset_* upload folders to their meaningful subfolder names
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "dataset_uploads"
ZIP_PATH = BASE / "06032026-Crystallization.zip"

SKIP_PATH_FRAGMENTS = (
    "wrong",
    "need better",
)

# zip source prefix -> destination folder name (under dataset_uploads)
INSULIN_SOURCES = {
    "1. Reference Condition - 20 C, 2.5 mgmL, 400 rpm/": "reference_condition_20C_2.5mgmL_400rpm",
    "2. Concentration - 1.25 and 5 mgmL/1.25 mgmL at 20C_35%RPM/06082026 - Correct one/": "insulin_1.25mgmL_20C_35pctRPM",
    "2. Concentration - 1.25 and 5 mgmL/5 mgmL at 20C_35%RPM/06082026 - correct one/": "insulin_5mgmL_20C_35pctRPM",
    "3. RPM - 200 and 800 rpm/200 rpm - low RPM _2.5mgmL_20C/": "insulin_200rpm_2.5mgmL_20C",
    "3. RPM - 200 and 800 rpm/800 rpm - high RPM_2.5mgmL_20C/": "insulin_800rpm_2.5mgmL_20C",
    "4. Temperature - 5 and 40 C/5C/": "insulin_5C_2.5mgmL_400rpm",
    "4. Temperature - 5 and 40 C/40C/": "insulin_40C_2.5mgmL_400rpm",
    "5. 200 rpm and 5C/": "insulin_200rpm_5C",
}

LEGACY_RENAMES = {
    "dataset_2026_06_23_13_01_40_904cf1fa": "Continuous_PmAb_1_mg-mL",
    "dataset_2026_06_24_12_16_17_454be66b": "Continuous_PmAb_2_mg-mL",
    "dataset_2026_06_24_12_34_27_064df4f6": "Continuous_PmAb_5_mg-mL",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _should_skip(path: str) -> bool:
    low = path.lower()
    return any(frag in low for frag in SKIP_PATH_FRAGMENTS)


def _mag_subfolder(filename: str) -> str | None:
    low = filename.lower()
    if re.search(r"(?<!\d)20x", low):
        return "x20"
    if re.search(r"(?<!\d)10x", low):
        return "x10"
    return None


def _extract_insulin_from_zip() -> list[str]:
    if not ZIP_PATH.exists():
        print(f"Zip not found: {ZIP_PATH}")
        return []

    created: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for src_prefix, dest_name in INSULIN_SOURCES.items():
            dest_root = BASE / dest_name
            if dest_root.exists():
                shutil.rmtree(dest_root)
            dest_root.mkdir(parents=True, exist_ok=True)

            count = 0
            for member in zf.namelist():
                if not member.startswith(src_prefix):
                    continue
                if member.endswith("/") or _should_skip(member):
                    continue
                fname = member.split("/")[-1]
                if Path(fname).suffix.lower() not in IMAGE_EXTS:
                    continue
                if fname.lower().startswith("image_"):
                    continue

                mag = _mag_subfolder(fname)
                out_dir = dest_root / mag if mag else dest_root
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / fname
                with zf.open(member) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1

            readme = dest_root / "README.txt"
            readme.write_text(
                f"Condition: {dest_name}\n"
                f"Source zip prefix: {src_prefix}\n"
                f"Images: {count}\n"
                f"Layout: x10/ and x20/ subfolders where filenames include magnification.\n"
                f"Outputs tab: select x20 for 100 µm scale calibration comparisons.\n",
                encoding="utf-8",
            )
            created.append(f"{dest_name} ({count} images)")
            print(f"  {dest_name}: {count} images")

    return created


def _rename_legacy_uploads() -> list[str]:
    renamed: list[str] = []
    for old_name, new_name in LEGACY_RENAMES.items():
        old_path = BASE / old_name
        if not old_path.exists():
            print(f"  skip (missing): {old_name}")
            continue

        new_path = BASE / new_name
        if new_path.exists():
            print(f"  skip (target exists): {new_name}")
            continue

        # If the upload wrapper contains a single subfolder with the target name, hoist it.
        children = [p for p in old_path.iterdir() if p.name != "README.txt"]
        if len(children) == 1 and children[0].is_dir() and children[0].name == new_name:
            children[0].rename(new_path)
            old_path.rmdir()
        else:
            old_path.rename(new_path)

        readme = new_path / "README.txt"
        if not readme.exists():
            readme.write_text(
                f"Dataset: {new_name}\n"
                f"Renamed from legacy upload folder: {old_name}\n",
                encoding="utf-8",
            )
        renamed.append(new_name)
        print(f"  {old_name} -> {new_name}")

    return renamed


def main() -> None:
    print(f"Organizing datasets under {BASE}\n")

    print("Renaming legacy PmAb uploads...")
    renamed = _rename_legacy_uploads()

    print("\nExtracting insulin conditions from zip (skipping wrong/bad folders)...")
    extracted = _extract_insulin_from_zip()

    # Master index for the user
    index_path = BASE / "DATASETS_INDEX.txt"
    lines = [
        "Crystallization datasets for Outputs tab",
        "=" * 40,
        "",
        "Insulin (from 06032026-Crystallization.zip):",
    ]
    for item in extracted:
        lines.append(f"  - {item}")
    lines += ["", "PmAb continuous (renamed uploads):"]
    for name in renamed:
        lines.append(f"  - {name}")
    lines += [
        "",
        "Excel sheet mapping for compare_psd.py:",
        "  reference_condition_20C_2.5mgmL_400rpm -> Reference-2.5mgmL, 20C, 400RPM",
        "  insulin_5C_2.5mgmL_400rpm                  -> 5C",
        "  insulin_40C_2.5mgmL_400rpm                 -> 40C",
        "  insulin_200rpm_2.5mgmL_20C                 -> 200RPM",
        "  insulin_800rpm_2.5mgmL_20C                 -> 800RPM",
        "  insulin_1.25mgmL_20C_35pctRPM              -> 1.25mgmL",
        "  insulin_5mgmL_20C_35pctRPM                  -> 5mgmL",
        "",
        "Excluded from zip:",
        "  - 06042026 - wrong HPLC dilution",
        "  - 06042026 - need better sampling",
        "",
        "Tip: In Outputs, pick the x20 subfolder for each insulin condition.",
    ]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {index_path}")


if __name__ == "__main__":
    main()
