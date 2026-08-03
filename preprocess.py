import io
import re
import hashlib
from pathlib import Path
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC_DIR = Path("data/main")
DST_DIR = Path("data/main_processed")
MAX_SIDE = 1024
QUALITY = 90
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def sanitize_stem(stem: str) -> str:
    """Normalize filename stem: lowercase, safe chars only, collapse repeats."""
    stem = stem.strip().lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "image"


def short_hash(data: bytes, length: int = 8) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def preprocess_image_bytes(raw: bytes, max_side=MAX_SIDE, quality=QUALITY) -> bytes:
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)

    if getattr(img, "is_animated", False):
        img.seek(0)

    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    longer_side = max(width, height)
    if longer_side > max_side:
        scale = max_side / longer_side
        new_size = (round(width * scale), round(height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def process_directory(src_dir: Path = SRC_DIR, dst_dir: Path = DST_DIR):
    dst_dir.mkdir(parents=True, exist_ok=True)

    used_names = {p.name for p in dst_dir.glob("*.jpg")}
    n_ok, n_skip, n_err = 0, 0, 0

    for src_path in sorted(src_dir.iterdir()):
        if not src_path.is_file() or src_path.suffix.lower() not in VALID_EXTS:
            continue

        stem = sanitize_stem(src_path.stem)
        out_name = f"{stem}.jpg"
        out_path = dst_dir / out_name

        try:
            raw = src_path.read_bytes()
        except Exception as e:
            print(f"[ERROR] could not read {src_path.name}: {e}")
            n_err += 1
            continue

        # If a file with this name already exists, check whether it's the same
        # source (skip) or a genuine collision (disambiguate with hash).
        if out_name in used_names:
            marker_path = dst_dir / f".{out_name}.srchash"
            src_hash = short_hash(raw)
            if marker_path.exists() and marker_path.read_text().strip() == src_hash:
                n_skip += 1
                continue  # already processed this exact file, skip
            else:
                out_name = f"{stem}_{src_hash}.jpg"
                out_path = dst_dir / out_name

        try:
            jpeg_bytes = preprocess_image_bytes(raw)
        except Exception as e:
            print(f"[ERROR] could not process {src_path.name}: {e}")
            n_err += 1
            continue

        out_path.write_bytes(jpeg_bytes)
        (dst_dir / f".{out_name}.srchash").write_text(short_hash(raw))
        used_names.add(out_name)
        n_ok += 1
        print(f"[OK] {src_path.name} -> {out_name}")

    print(f"\nDone. Processed: {n_ok}, Skipped (already processed): {n_skip}, Errors: {n_err}")


if __name__ == "__main__":
    process_directory()