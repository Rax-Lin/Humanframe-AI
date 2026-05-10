#!/usr/bin/env python3
"""Download SmugMug portraits for Level-4 training data.

Pipeline:
1) Query SmugMug ImageSearch API.
2) Download raw images into data/raw/level4_excellent.
3) Run YOLO person filtering via portraiq/utils/pose_utils.py.
4) Copy passing images into data/processed/level4_excellent.
5) Save annotations to data/annotations/level4_smugmug.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError


API_URL = "https://api.smugmug.com/api/v2/image!search"
USER_AGENT = "Humanframe-AI/1.0 (portrait aesthetics research)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and build Level-4 SmugMug dataset")
    parser.add_argument("--max_images", type=int, default=5000, help="Maximum total annotated images")
    parser.add_argument(
        "--per_keyword",
        type=int,
        default=1500,
        help="Maximum number of API candidates per keyword",
    )
    parser.add_argument("--person_conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--resume", action="store_true", help="Resume from existing annotation JSON")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--save_every", type=int, default=200, help="Save annotation every N newly kept images")
    parser.add_argument("--sleep_ms", type=int, default=0, help="Sleep milliseconds between requests")
    parser.add_argument("--min_side", type=int, default=384, help="Minimum accepted width/height")
    parser.add_argument("--person_model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument(
        "--scope",
        type=str,
        default="/api/v2/user/cmac",
        help="SmugMug search scope URI (e.g. /api/v2/user/<nickname>)",
    )
    parser.add_argument(
        "--discover_scopes",
        action="store_true",
        help="Discover additional public user scopes via user!search",
    )
    parser.add_argument(
        "--max_scopes",
        type=int,
        default=30,
        help="Maximum discovered scopes to use when --discover_scopes is enabled",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data root (default: <repo>/portraiq/data)",
    )
    return parser.parse_args()


def repo_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "portraiq" / "data"


def ensure_dirs(data_root: Path) -> Dict[str, Path]:
    raw_dir = data_root / "raw" / "level4_excellent"
    proc_dir = data_root / "processed" / "level4_excellent"
    ann_dir = data_root / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": proc_dir, "annotations": ann_dir}


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _download_binary(url: str, dst: Path, timeout: int) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dst.write_bytes(resp.content)
        return True
    except Exception:
        return False


def _extract_items(payload: Dict) -> List[Dict]:
    # Handle common SmugMug response shapes robustly.
    if not isinstance(payload, dict):
        return []
    response = payload.get("Response")
    if isinstance(response, dict):
        for key in ("Image", "Images", "AlbumImage", "AlbumImages"):
            value = response.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if isinstance(response.get("Node"), dict):
            node = response["Node"]
            for key in ("Image", "Images"):
                value = node.get(key)
                if isinstance(value, list):
                    return [x for x in value if isinstance(x, dict)]
    for key in ("Image", "Images"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def _extract_largest_uri(item: Dict) -> Optional[str]:
    largest = item.get("LargestImage")
    if isinstance(largest, dict):
        uri = largest.get("Uri") or largest.get("uri")
        if isinstance(uri, str) and uri.strip():
            return uri.strip()

    uris = item.get("Uris")
    if isinstance(uris, dict):
        largest = uris.get("LargestImage")
        if isinstance(largest, dict):
            uri = largest.get("Uri") or largest.get("uri")
            if isinstance(uri, str) and uri.strip():
                return uri.strip()
    return None


def _largest_image_url(item: Dict) -> Optional[str]:
    largest = item.get("LargestImage")
    if isinstance(largest, dict):
        url = largest.get("Url") or largest.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()

    # Fallbacks for variant field names occasionally seen in API payloads.
    for key in (
        "LargestImageUrl",
        "LargestImageURL",
        "ImageUrl",
        "ImageURL",
        "OriginalUrl",
        "OriginalURL",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _iter_smugmug_candidates(
    api_key: str,
    scope: str,
    per_keyword: int,
    timeout: int,
) -> Iterable[Tuple[str, str]]:
    keywords = ["portrait", "professional portrait", "street portrait", "environmental portrait"]
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}

    for kw in keywords:
        start = 1
        seen_for_kw = 0
        while seen_for_kw < per_keyword:
            params = {
                "APIKey": api_key,
                "Text": kw,
                "Scope": scope,
                "count": 100,
                "start": start,
            }
            try:
                resp = requests.get(API_URL, headers=headers, params=params, timeout=timeout)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                extra = ""
                try:
                    if "resp" in locals():
                        body = resp.text[:400].replace("\n", " ")
                        extra = f" | body={body}"
                except Exception:
                    pass
                print(f"SmugMug search failed for '{kw}' start {start}: {exc}{extra}")
                break

            items = _extract_items(payload)
            if not items:
                break

            emitted_this_page = 0
            for item in items:
                image_key = (
                    item.get("ImageKey")
                    or item.get("ImageKeyString")
                    or item.get("Key")
                    or item.get("Uri")
                )
                largest_uri = _extract_largest_uri(item)
                if not image_key or not largest_uri:
                    continue
                seen_for_kw += 1
                emitted_this_page += 1
                yield str(image_key), str(largest_uri)
                if seen_for_kw >= per_keyword:
                    break

            if emitted_this_page == 0:
                break

            start += len(items)


def _discover_user_scopes(api_key: str, timeout: int, max_scopes: int) -> List[str]:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    terms = ["portrait", "professional portrait", "street portrait", "environmental portrait"]
    scopes: List[str] = []
    seen = set()

    for term in terms:
        start = 1
        while len(scopes) < max_scopes:
            params = {"APIKey": api_key, "q": term, "count": 100, "start": start}
            try:
                r = requests.get(
                    "https://api.smugmug.com/api/v2/user!search",
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )
                r.raise_for_status()
                payload = r.json()
            except Exception:
                break

            users = payload.get("Response", {}).get("User", [])
            if not isinstance(users, list) or not users:
                break

            added_this_page = 0
            for user in users:
                if not isinstance(user, dict):
                    continue
                uri = user.get("Uri")
                if not isinstance(uri, str) or not uri.startswith("/api/v2/user/"):
                    continue
                if uri in seen:
                    continue
                seen.add(uri)
                scopes.append(uri)
                added_this_page += 1
                if len(scopes) >= max_scopes:
                    break

            if added_this_page == 0:
                break
            start += len(users)
    return scopes[:max_scopes]


def main() -> None:
    args = parse_args()
    api_key = os.getenv("SMUGMUG_API_KEY")
    if not api_key:
        raise SystemExit("SMUGMUG_API_KEY is required.")

    data_root = Path(args.data_root) if args.data_root else repo_data_root()
    paths = ensure_dirs(data_root)
    output_json = paths["annotations"] / "level4_smugmug.json"

    annotations: List[Dict] = []
    annotated_ids = set()
    if args.resume and output_json.exists():
        prev = load_json(output_json)
        if isinstance(prev, list):
            annotations = prev
            for item in prev:
                if isinstance(item, dict):
                    image_id = str(item.get("image_id", "")).strip()
                    if image_id:
                        annotated_ids.add(image_id)

    # Import project detector only after path setup.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    portraiq_root = repo_root / "portraiq"
    if str(portraiq_root) not in sys.path:
        sys.path.insert(0, str(portraiq_root))

    from utils.pose_utils import PersonDetector

    detector = PersonDetector(model_name=args.person_model, use_yolo=True)
    if detector.model is None:
        raise SystemExit("YOLO detector not available. Install ultralytics or fix YOLO model path.")

    scopes = [str(args.scope)]
    if args.discover_scopes:
        discovered = _discover_user_scopes(
            api_key=str(api_key),
            timeout=int(args.timeout),
            max_scopes=max(1, int(args.max_scopes)),
        )
        for s in discovered:
            if s not in scopes:
                scopes.append(s)
        print(f"DEBUG | discovered_scopes={len(discovered)} total_scopes={len(scopes)}")

    new_downloaded = 0
    new_kept = 0
    retrieved_from_api = 0

    for scope in scopes:
        candidates = _iter_smugmug_candidates(
            api_key=str(api_key),
            scope=scope,
            per_keyword=int(args.per_keyword),
            timeout=int(args.timeout),
        )

        for ext_id, largest_uri in candidates:
            retrieved_from_api += 1
            if len(annotations) >= int(args.max_images):
                break

            image_id = f"smugmug_{ext_id}"
            raw_name = f"{image_id}.jpg"
            raw_path = paths["raw"] / raw_name
            proc_path = paths["processed"] / raw_name

            if image_id in annotated_ids:
                continue

            largest_url = _largest_image_url({"LargestImage": {"Uri": largest_uri}})
            if largest_url is None:
                # Resolve URI form (/api/v2/image/...!largestimage) to URL payload.
                try:
                    endpoint = f"https://api.smugmug.com{largest_uri}"
                    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
                    r = requests.get(endpoint, headers=headers, params={"APIKey": api_key}, timeout=int(args.timeout))
                    r.raise_for_status()
                    payload = r.json()
                    largest_url = (
                        payload.get("Response", {})
                        .get("LargestImage", {})
                        .get("Url")
                    )
                except Exception:
                    largest_url = None

            if not largest_url:
                continue

            if not raw_path.exists():
                ok = _download_binary(largest_url, raw_path, timeout=int(args.timeout))
                if not ok:
                    continue
                new_downloaded += 1

            try:
                with Image.open(raw_path) as im:
                    image = im.convert("RGB")
            except (UnidentifiedImageError, OSError):
                raw_path.unlink(missing_ok=True)
                continue

            w, h = image.size
            if min(w, h) < int(args.min_side):
                continue

            det = detector.detect_largest_person(image_rgb=np.array(image))
            if det.source != "yolo" or float(det.confidence) < float(args.person_conf):
                continue

            shutil.copy2(raw_path, proc_path)
            annotations.append(
                {
                    "image_id": image_id,
                    "filename": f"level4_excellent/{raw_name}",
                    "category": "portrait",
                    "score": 9.5,
                    "score_std": 0.0,
                    "annotator": "smugmug_fixed",
                    "split": "train",
                }
            )
            annotated_ids.add(image_id)
            new_kept += 1

            if args.save_every > 0 and new_kept % int(args.save_every) == 0:
                dump_json(output_json, annotations)

            if args.sleep_ms > 0:
                time.sleep(float(args.sleep_ms) / 1000.0)
        if len(annotations) >= int(args.max_images):
            break

    dump_json(output_json, annotations)
    print(f"DEBUG | source=smugmug | candidate_urls_retrieved_before_yolo={retrieved_from_api}")
    print("SmugMug download complete")
    print(f"Downloaded raw images (new): {new_downloaded}")
    print(f"Processed/kept images (new): {new_kept}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Output annotations: {output_json}")


if __name__ == "__main__":
    main()
