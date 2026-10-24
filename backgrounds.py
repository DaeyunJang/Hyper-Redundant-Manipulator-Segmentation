"""Download or expand real Endoscapes backgrounds while preserving official splits."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

URL = "https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
SOURCE = "https://github.com/CAMMA-public/Endoscapes"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SIZE = 6285499749
ETAG = '"445795b0639cc3fce825ed197fa3949c-750"'
LICENSE_TEXT_URL = "https://raw.githubusercontent.com/CAMMA-public/Endoscapes/main/LICENSE"
COUNTS = {"train": 20, "val": 5, "test": 5}
CITATIONS = [
    {"title": "The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark", "authors": "Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Guido Costamagna, Didier Mutter, Jacques Marescaux, Bernard Dallemagne, Nicolas Padoy", "year": 2023, "url": "https://arxiv.org/abs/2312.12429"},
    {"title": "Latent Graph Representations for Critical View of Safety Assessment", "authors": "Aditya Murali et al.", "year": 2023, "doi": "10.1109/TMI.2023.3333034", "url": "https://arxiv.org/abs/2212.04155"},
]


def get_range(start: int, length: int) -> bytes:
    """Bound every response; never consume a server response that ignores Range."""
    if length <= 0 or length > 40_000_000:
        raise ValueError(f"Unsafe byte range size: {length}")
    end = min(start + length, SIZE) - 1
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=30) as response:
                expected = f"bytes {start}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Server ignored or changed Range; refusing full-archive download.")
                if response.headers.get("ETag") != ETAG:
                    raise RuntimeError("Official source archive changed; review provenance before updating.")
                result = response.read(end - start + 2)
                if len(result) != end - start + 1:
                    raise OSError("Truncated or oversized HTTP range response")
                return result
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def read(self, n=-1):
        n = min(SIZE - self.position, n if n >= 0 else SIZE - self.position)
        if n <= 0:
            return b""
        print(f"Reading archive index range: {self.position:,} + {n:,} bytes", flush=True)
        if n > 1_048_576:
            requests = [(start, min(1_048_576, self.position + n - start))
                        for start in range(self.position, self.position + n, 1_048_576)]
            with ThreadPoolExecutor(max_workers=6) as pool:
                data = b"".join(pool.map(lambda pair: get_range(*pair), requests))
        else:
            data = get_range(self.position, n)
        self.position += n
        return data


def source_index(destination):
    cache = destination / ".cache/endoscapes-index.json"
    if cache.is_file():
        index = json.loads(cache.read_text())
        if index.get("etag") == ETAG and index.get("url") == URL:
            return index["members"]
    members = []
    with zipfile.ZipFile(RemoteZip()) as archive:
        for member in archive.infolist():
            match = re.fullmatch(r"endoscapes/(train|val|test)/(\d+)_(\d+)\.jpg", member.filename)
            if match and member.file_size >= 30_000:
                split, video, frame = match[1], int(match[2]), int(match[3])
                members.append({
                    "path": f"{split}/endoscapes_{video:03d}_{frame:06d}.jpg",
                    "origin_path": member.filename, "video_id": video,
                    "frame_index": frame, "group_id": f"endoscapes_video_{video:03d}",
                    "official_split": split, "source_id": "endoscapes2023",
                    "header_offset": member.header_offset, "compressed_size": member.compress_size,
                    "compression": member.compress_type, "size_bytes": member.file_size,
                    "crc32": member.CRC,
                })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"url": URL, "etag": ETAG, "members": members}))
    return members


def choose_members(index, targets, existing, excluded):
    selected = []
    used = {entry["origin_path"] for entry in existing} | excluded
    for split, target in targets.items():
        by_video = {}
        chosen = {}
        for entry in existing:
            if entry["official_split"] == split:
                chosen.setdefault(entry["video_id"], []).append(entry["frame_index"])
        for entry in index:
            if entry["official_split"] == split and entry["origin_path"] not in used:
                by_video.setdefault(entry["video_id"], []).append(entry)
        needed = max(0, target - sum(map(len, chosen.values())))
        for _ in range(needed):
            if not by_video:
                raise RuntimeError(f"Insufficient usable source frames for {split}={target}")
            # Cover unused cases first; spread extra samples across cases deterministically.
            video = min(by_video, key=lambda v: (len(chosen.get(v, [])), hashlib.sha256(str(v).encode()).hexdigest()))
            frames = by_video[video]
            previous = chosen.get(video, [])
            if previous:
                entry = max(frames, key=lambda e: (min(abs(e["frame_index"] - f) for f in previous), -e["frame_index"]))
            else:
                frames.sort(key=lambda e: e["frame_index"])
                entry = frames[len(frames) // 2]
            selected.append(entry)
            chosen.setdefault(video, []).append(entry["frame_index"])
            frames.remove(entry)
            if not frames:
                del by_video[video]
    return selected


class RejectedFrame(ValueError):
    pass


def image_quality(data):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RejectedFrame("Image decode failed")
    mean, std = float(image.mean()), float(image.std())
    visible_fraction = float((image.max(axis=2) > 15).mean())
    if mean < 8 or std < 8 or visible_fraction < 0.2:
        raise RejectedFrame(f"Blank or near-black image: mean={mean:.2f}, std={std:.2f}, visible={visible_fraction:.3f}")
    return {"pixel_mean": round(mean, 3), "pixel_std": round(std, 3), "visible_fraction": round(visible_fraction, 4)}


def jpeg_dimensions(data):
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Expected original JPEG image")
    offset = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            raise ValueError("Invalid JPEG marker")
        while data[offset] == 0xFF:
            offset += 1
        marker = data[offset]
        offset += 1
        length = struct.unpack_from(">H", data, offset)[0]
        if marker in sof:
            height, width = struct.unpack_from(">HH", data, offset + 3)
            return width, height
        offset += length
    raise ValueError("JPEG dimensions were not found")


def fetch_member(entry, destination):
    path = destination / entry["path"]
    if path.is_file() and entry.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
        return {**entry, "quality": image_quality(path.read_bytes())}
    raw = get_range(entry["header_offset"], entry["compressed_size"] + 65536)
    if raw[:4] != b"PK\x03\x04":
        raise RuntimeError("Invalid ZIP local header")
    filename_len, extra_len = struct.unpack_from("<HH", raw, 26)
    offset = 30 + filename_len + extra_len
    filename = raw[30:30 + filename_len].decode("utf-8")
    if filename != entry["origin_path"]:
        raise RuntimeError("Unexpected archive member filename")
    compressed = raw[offset:offset + entry["compressed_size"]]
    if len(compressed) != entry["compressed_size"]:
        raise RuntimeError("ZIP local extra field exceeded bounded response")
    if entry["compression"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif entry["compression"] == zipfile.ZIP_STORED:
        data = compressed
    else:
        raise RuntimeError("Unsupported ZIP compression")
    if len(data) != entry["size_bytes"] or zlib.crc32(data) != entry["crc32"]:
        raise RuntimeError("ZIP CRC/length verification failed")
    quality = image_quality(data)
    width, height = jpeg_dimensions(data)
    digest = hashlib.sha256(data).hexdigest()
    if entry.get("sha256") not in (None, digest):
        raise RuntimeError("Saved source SHA-256 does not match downloaded bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jpg.part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return {**entry, "sha256": digest, "width": width, "height": height, "modified": False, "quality": quality}


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def preview(destination, entries):
    import cv2
    import numpy as np
    # Representative montage: one frame per video, mixed across official splits.
    unique = {entry["group_id"]: entry for entry in entries}
    ordered = sorted(unique.values(), key=lambda e: e["video_id"])
    count = min(60, len(ordered))
    samples = [ordered[round(i * (len(ordered) - 1) / max(1, count - 1))] for i in range(count)]
    columns, width, height = 6, 240, 160
    canvas = np.zeros((((count + columns - 1) // columns) * height, columns * width, 3), np.uint8)
    for i, entry in enumerate(samples):
        frame = cv2.imread(str(destination / entry["path"]))
        thumb = cv2.resize(frame, (width, 135), interpolation=cv2.INTER_AREA)
        row, col = divmod(i, columns)
        canvas[row * height:row * height + 135, col * width:(col + 1) * width] = thumb
        label = f"{entry['official_split']}  case {entry['video_id']} / {entry['frame_index']}"
        cv2.putText(canvas, label, (col * width + 4, row * height + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(destination / "preview.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError("Failed to write background preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "datasets/backgrounds")
    for split, count in COUNTS.items():
        parser.add_argument(f"--{split}", type=int, default=count, help="Minimum frame count; existing images are always retained")
    args = parser.parse_args()
    targets = {split: getattr(args, split) for split in COUNTS}
    if any(count < 1 for count in targets.values()):
        parser.error("Every split count must be at least 1")
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "sources.json"
    original = manifest_path.read_bytes() if manifest_path.is_file() else None
    if original:
        manifest = json.loads(original)
        if manifest.get("source_url") != URL or manifest.get("archive_etag") != ETAG:
            raise RuntimeError("Existing manifest does not describe the expected official archive")
    else:
        manifest = {
            "schema_version": 2, "dataset": "Endoscapes2023 laparoscopic cholecystectomy frames",
            "source_url": URL, "project_url": SOURCE, "archive_etag": ETAG,
            "archive_size_bytes": SIZE, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": "CC BY-NC-SA 4.0", "license_url": LICENSE_URL,
            "license_source_url": LICENSE_TEXT_URL, "citations": CITATIONS, "files": [],
            "limitations": ["Real surgery backgrounds; original instruments and black endoscope borders remain.",
                            "Only laparoscopic cholecystectomy is represented; this is not coverage of other operations or anatomy.",
                            "Composites are synthetic, not validation on real surgical HRM footage.",
                            "Keep official video splits separate and preserve attribution and CC BY-NC-SA terms."],
        }
    existing = manifest["files"]
    counts = Counter(entry["official_split"] for entry in existing)
    expanding = any(counts[split] < target for split, target in targets.items())
    if original and (expanding or manifest.get("schema_version", 1) < 2):
        history = destination / "manifests" / f"sources-{hashlib.sha256(original).hexdigest()[:16]}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        if not history.exists():
            history.write_bytes(original)
        manifest.setdefault("history", []).append(str(history.relative_to(destination)))
    manifest["schema_version"] = 2
    manifest["dataset"] = "Endoscapes2023 original laparoscopic cholecystectomy frames"
    manifest["selection"] = "Preserve all existing frames. Fill unused official video groups first, then prefer temporally distant frames within the least-sampled groups. Reject tiny JPEGs and decoded blank/near-black frames."
    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest["files"] = list(pool.map(lambda e: fetch_member(e, destination), existing))
    index = source_index(destination) if expanding else []
    excluded = {entry["origin_path"] for entry in manifest.get("excluded_source_frames", [])}
    while True:
        planned = choose_members(index, targets, manifest["files"], excluded)
        if not planned:
            break
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_member, entry, destination): entry for entry in planned}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    manifest["files"].append(future.result())
                    if len(manifest["files"]) % 25 == 0:
                        print(f"Verified {len(manifest['files'])} background frames", flush=True)
                except RejectedFrame as error:
                    excluded.add(entry["origin_path"])
                    manifest.setdefault("excluded_source_frames", []).append({"origin_path": entry["origin_path"], "reason": str(error)})
                    print(f"Excluded {entry['origin_path']}: {error}", flush=True)
                finally:
                    write_manifest(manifest_path, manifest)
    manifest["files"].sort(key=lambda e: (e["official_split"], e["video_id"], e["frame_index"]))
    groups = {}
    for entry in manifest["files"]:
        entry.setdefault("source_id", "endoscapes2023")
        group, split = entry["group_id"], entry["official_split"]
        if group in groups and groups[group] != split:
            raise RuntimeError("A source video crosses background splits")
        groups[group] = split
    license_path = destination / "LICENSE-CC-BY-NC-SA-4.0.txt"
    if not license_path.is_file():
        with urllib.request.urlopen(LICENSE_TEXT_URL, timeout=30) as response:
            text = response.read(100_000)
        if b"Attribution-NonCommercial-ShareAlike 4.0 International" not in text:
            raise RuntimeError("Unexpected official dataset license text")
        license_path.write_bytes(text)
    manifest["license_sha256"] = hashlib.sha256(license_path.read_bytes()).hexdigest()
    manifest["counts"] = dict(Counter(e["official_split"] for e in manifest["files"]))
    manifest["distinct_videos"] = len(groups)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    preview(destination, manifest["files"])
    print(f"Ready: {len(manifest['files'])} original frames, {len(groups)} source videos; {manifest['counts']}")


if __name__ == "__main__":
    main()
