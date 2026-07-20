#!/usr/bin/env python3
"""
Download model weights for the ObjectDetection module.

Usage:
    python download_models.py --dest assets --custom-dest custom-models

  --dest        Directory to store the generic YOLO weight (created if absent).
  --custom-dest Directory to store the IPcam custom model weights (created if absent).
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import zipfile

CUSTOM_MODELS_URL = (
    "https://codeproject-ai-bunny.b-cdn.net/server/assets/models/"
    "custom-models-yolo5-pt.zip"
)

# The one model we require for a successful install
REQUIRED_MODEL = "ipcam-combined.pt"


def _make_opener():
    """Build a urllib opener that follows HTTP redirects."""
    handler = urllib.request.HTTPRedirectHandler()
    return urllib.request.build_opener(handler)


def download_generic_model(dest_dir: str) -> None:
    """Trigger Ultralytics to download yolo26n.pt into dest_dir (best-effort).

    This requires ultralytics/torch which are installed by install.sh/install.bat.
    If they are not yet available the download is skipped non-fatally — the weight
    will be auto-downloaded by Ultralytics at first runtime use.
    """
    os.makedirs(dest_dir, exist_ok=True)
    target = os.path.join(dest_dir, "yolo26n.pt")
    if os.path.isfile(target):
        print(f"[download_models] yolo26n.pt already present in {dest_dir}, skipping.")
        return

    print("[download_models] Fetching yolo26n.pt via Ultralytics…")
    # Ultralytics downloads to the cwd / YOLO_HOME by default; we change to
    # dest_dir so the weight lands there, then verify.
    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError:
            print(
                "[download_models] NOTE: ultralytics not yet installed — yolo26n.pt will be "
                "auto-downloaded at first runtime use. Skipping pre-download."
            )
            return
        YOLO("yolo26n.pt")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[download_models] NOTE: yolo26n.pt pre-download failed ({exc}) — "
            "it will be auto-downloaded at first runtime use."
        )
        return
    finally:
        os.chdir(old_cwd)

    if not os.path.isfile(target):
        print(f"[download_models] WARNING: yolo26n.pt not found at {target} after download.")
    else:
        print(f"[download_models] yolo26n.pt saved to {target}.")


def download_custom_models(custom_dest_dir: str) -> bool:
    """
    Download and unzip the IPcam custom models zip into custom_dest_dir.
    Skips files that already exist.
    Returns True iff ipcam-combined.pt is present after the operation.
    """
    os.makedirs(custom_dest_dir, exist_ok=True)
    required_path = os.path.join(custom_dest_dir, REQUIRED_MODEL)

    if os.path.isfile(required_path):
        print(
            f"[download_models] {REQUIRED_MODEL} already present in {custom_dest_dir}, skipping download."
        )
        return True

    zip_path = os.path.join(custom_dest_dir, "custom-models-yolo5-pt.zip")

    if not os.path.isfile(zip_path):
        print(f"[download_models] Downloading custom models from {CUSTOM_MODELS_URL} …")
        opener = _make_opener()
        try:
            with opener.open(CUSTOM_MODELS_URL) as resp, open(zip_path, "wb") as out:
                total = 0
                while True:
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
            print(f"[download_models] Downloaded {total / 1e6:.1f} MB → {zip_path}")
        except (urllib.error.URLError, OSError) as exc:
            print(f"[download_models] ERROR downloading custom models: {exc}", file=sys.stderr)
            return False
    else:
        print(f"[download_models] Zip already present at {zip_path}, skipping download.")

    print(f"[download_models] Extracting {zip_path} → {custom_dest_dir} …")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                dest_file = os.path.join(custom_dest_dir, member.filename)
                if os.path.isfile(dest_file):
                    print(f"[download_models]   skip (exists): {member.filename}")
                    continue
                zf.extract(member, custom_dest_dir)
                print(f"[download_models]   extracted: {member.filename}")
    except zipfile.BadZipFile as exc:
        print(f"[download_models] ERROR: bad zip file: {exc}", file=sys.stderr)
        # Remove corrupt zip so next run re-downloads
        os.remove(zip_path)
        return False

    if os.path.isfile(required_path):
        print(f"[download_models] {REQUIRED_MODEL} present — custom models OK.")
        return True
    else:
        print(
            f"[download_models] ERROR: {REQUIRED_MODEL} not found after extraction.",
            file=sys.stderr,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ObjectDetection model weights.")
    parser.add_argument(
        "--dest",
        default="assets",
        help="Directory to store the generic YOLO weight (default: assets).",
    )
    parser.add_argument(
        "--custom-dest",
        default="custom-models",
        help="Directory to store IPcam custom model weights (default: custom-models).",
    )
    args = parser.parse_args()

    # Required: IPcam custom models (stdlib-only — safe before pip installs)
    ok = download_custom_models(args.custom_dest)
    # Best-effort: generic YOLO26 weight (needs ultralytics/torch)
    download_generic_model(args.dest)

    if not ok:
        print(
            "[download_models] FAILED: required model ipcam-combined.pt is missing.",
            file=sys.stderr,
        )
        return 1

    print("[download_models] All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
