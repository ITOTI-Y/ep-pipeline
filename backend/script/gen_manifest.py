"""Generate and verify manifest for backend/data/ and backend/output/ directories."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

SAMPLE_THRESHOLD = 10 * 1024 * 1024  # 10MB
SAMPLE_SIZE = 1024 * 1024  # 1MB per sample chunk

TRACKED_DIRS = [Path("backend/data"), Path("backend/output")]
MANIFEST_PATH = Path("data.manifest.json")


def _sampled_hash(path: Path, file_size: int) -> str:
    h = hashlib.sha256()
    h.update(file_size.to_bytes(8, "big"))
    with open(path, "rb") as f:
        h.update(f.read(SAMPLE_SIZE))
        mid = max(0, file_size // 2 - SAMPLE_SIZE // 2)
        f.seek(mid)
        h.update(f.read(SAMPLE_SIZE))
        f.seek(max(0, file_size - SAMPLE_SIZE))
        h.update(f.read(SAMPLE_SIZE))
    return h.hexdigest()


def _full_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hash(path: Path) -> str:
    size = path.stat().st_size
    if size >= SAMPLE_THRESHOLD:
        return _sampled_hash(path, size)
    return _full_hash(path)


def generate(dirs: list[Path] = TRACKED_DIRS) -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(f for f in d.rglob("*") if f.is_file()):
            stat = f.stat()
            manifest[str(f.relative_to("."))] = {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "sha256": file_hash(f),
            }
    return manifest


def save(manifest: dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load(path: Path = MANIFEST_PATH) -> dict[str, dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check(manifest_path: Path = MANIFEST_PATH) -> bool:
    old = load(manifest_path)
    if old is None:
        print("No existing manifest found. Run without --check first.")
        return False

    new = generate()
    old_keys = set(old)
    new_keys = set(new)

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    common = old_keys & new_keys
    modified = sorted(k for k in common if old[k]["sha256"] != new[k]["sha256"])

    if not added and not removed and not modified:
        print(f"OK: {len(common)} files unchanged.")
        return True

    if added:
        print(f"\n  Added ({len(added)}):")
        for p in added:
            print(f"    + {p}")
    if removed:
        print(f"\n  Removed ({len(removed)}):")
        for p in removed:
            print(f"    - {p}")
    if modified:
        print(f"\n  Modified ({len(modified)}):")
        for p in modified:
            print(f"    ~ {p}")

    print(
        f"\nSummary: {len(added)} added, {len(removed)} removed, {len(modified)} modified"
    )
    return False
