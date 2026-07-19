# modules/ImprovedFaceProcessing/intelligencelayer/download_models.py
#
# Model sources:
#   Detector (accurate):  SCRFD-10G ONNX — public-data/insightface on HuggingFace
#                         (individual file, 16.9 MB)
#   Detector (fast):      SCRFD-2.5G ONNX — inside buffalo_m.zip from
#                         vladmandic/insightface-faceanalysis on HuggingFace;
#                         download_tier() fetches the zip and extracts the .onnx.
#   Recognizer (accurate): AdaFace IR-101 WebFace12M .pt —
#                          minchul/cvlface_adaface_ir101_webface12m on HuggingFace
#   Recognizer (fast):    AdaFace IR-50 MS1MV2 .pt —
#                          minchul/cvlface_adaface_ir50_ms1mv2 on HuggingFace
#
# sha256 values are left empty; checksum check is skipped when sha256 == "".

import os
import hashlib
import urllib.request
import zipfile
import tempfile

TIER_MODELS = {
    "accurate": {
        "detector": {
            "file": "det_10g.onnx",
            "url":  "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
            "sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        },
        "recognizer": {
            "file": "adaface_ir101_webface12m.pt",
            "url":  "https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/main/pretrained_model/model.pt",
            "sha256": "e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758",
        },
    },
    "fast": {
        "detector": {
            # det_2.5g.onnx is not available as a standalone file;
            # it lives inside buffalo_m.zip.  download_tier() handles the
            # zip-fetch-and-extract step transparently.
            "file": "det_2.5g.onnx",
            "url":  "https://huggingface.co/vladmandic/insightface-faceanalysis/resolve/main/buffalo_m.zip",
            "sha256": "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
        },
        "recognizer": {
            "file": "adaface_ir50_ms1mv2.pt",
            "url":  "https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2/resolve/main/pretrained_model/model.pt",
            "sha256": "b9401a04c1f3e782ac4faa36a255619ef25cf2474321d395412b27a67b47cf7d",
        },
    },
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: str) -> None:
    """Download *url* to *dest*, following redirects (HuggingFace LFS)."""
    # HuggingFace resolve URLs redirect to the actual LFS blob.
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    with opener.open(url) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _download_direct(spec: dict, dest: str) -> None:
    """Download a single-file model (ONNX or .pt) if not already present."""
    if not os.path.exists(dest):
        print(f"Downloading {spec['file']} …")
        _fetch(spec["url"], dest)
        print(f"  saved → {dest}")


def _download_from_zip(spec: dict, dest: str) -> None:
    """
    Download a zip archive and extract the target .onnx from it.
    Used for det_2.5g.onnx which ships only inside buffalo_m.zip.
    """
    if os.path.exists(dest):
        return
    target_name = spec["file"]          # e.g. "det_2.5g.onnx"
    zip_url     = spec["url"]
    print(f"Downloading zip for {target_name} ({zip_url}) …")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _fetch(zip_url, tmp_path)
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Find the entry whose basename matches our target file
            matches = [n for n in zf.namelist() if os.path.basename(n) == target_name]
            if not matches:
                raise FileNotFoundError(
                    f"{target_name} not found inside zip downloaded from {zip_url}. "
                    f"Available entries: {zf.namelist()}"
                )
            with zf.open(matches[0]) as src, open(dest, "wb") as out:
                out.write(src.read())
        print(f"  extracted {target_name} → {dest}")
    finally:
        os.unlink(tmp_path)


def download_tier(tier: str, dest_dir: str) -> list:
    """
    Download all models for *tier* ('accurate' or 'fast') into *dest_dir*.

    Returns a list of absolute paths to the downloaded files.
    Checksum verification is skipped when sha256 == "" (not yet computed).
    """
    if tier not in TIER_MODELS:
        raise ValueError(f"Unknown tier {tier!r}; valid values: {list(TIER_MODELS)}")

    os.makedirs(dest_dir, exist_ok=True)
    out_paths = []

    for role in ("detector", "recognizer"):
        spec = TIER_MODELS[tier][role]
        dest = os.path.join(dest_dir, spec["file"])

        url = spec["url"]
        if url.endswith(".zip") and not spec["file"].endswith(".zip"):
            # The source is a zip archive; extract the target file from it.
            _download_from_zip(spec, dest)
        else:
            _download_direct(spec, dest)

        # Checksum — skipped when sha256 is empty (not yet populated)
        expected = spec.get("sha256", "")
        if expected and not expected.startswith("<"):
            actual = _sha256(dest)
            assert actual == expected, (
                f"SHA-256 mismatch for {spec['file']}: "
                f"expected {expected}, got {actual}"
            )

        out_paths.append(dest)

    return out_paths


if __name__ == "__main__":
    import sys
    dest = sys.argv[1] if len(sys.argv) > 1 else "assets"
    for t in ("accurate", "fast"):
        print(f"\n=== Tier: {t} ===")
        paths = download_tier(t, dest)
        for p in paths:
            size_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"  {p}  ({size_mb:.1f} MB)")
