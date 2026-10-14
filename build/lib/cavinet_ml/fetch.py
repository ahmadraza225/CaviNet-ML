"""`make fetch-model`: put a model bundle at the given path.

1. If a URL is given (MODEL_URL in .env, a GitHub Release asset), download it, check the
   optional SHA-256, and verify it loads as a bundle.
2. Otherwise, or if the download fails, build the demo bundle (is_demo = true).

A model card (`model_card.json`, the bundle without weights) is written beside it.
"""

import json
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from cavinet_ml.demo.build import build_demo_bundle
from cavinet_ml.model.bundle import BundleError, file_sha256, load_bundle, metadata

DOWNLOAD_TIMEOUT_SECONDS = 60


def write_model_card(bundle_path: Path) -> Path:
    info = metadata(load_bundle(bundle_path, mmap=True))
    info["file_sha256"] = file_sha256(bundle_path)
    card = bundle_path.with_name("model_card.json")
    card.write_text(json.dumps(info, indent=2, default=str) + "\n", encoding="utf-8")
    return card


def download(url: str, destination: Path, sha256: str | None = None) -> None:
    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(url, headers={"User-Agent": "cavinet-fetch-model"})
    with (
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        temporary.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    try:
        if sha256 and file_sha256(temporary) != sha256.lower():
            raise BundleError("the downloaded file does not match MODEL_SHA256")
        load_bundle(temporary, mmap=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)


def fetch_model(
    out: Path | str,
    *,
    url: str | None = None,
    sha256: str | None = None,
    force: bool = False,
    build_demo: Callable[..., object] = build_demo_bundle,
    log: Callable[[str], None] = print,
) -> str:
    """Returns "kept", "downloaded" or "demo"."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        try:
            load_bundle(out, mmap=True)
            log(f"A valid model is already installed at {out}.")
            return "kept"
        except BundleError:
            log(f"{out} is not a valid model bundle; replacing it.")
    if url:
        log(f"Downloading the model from {url} …")
        try:
            download(url, out, sha256)
            write_model_card(out)
            log(f"Model downloaded to {out}.")
            return "downloaded"
        except Exception as error:  # noqa: BLE001 - any failure falls back to the demo
            log(f"Download failed ({type(error).__name__}: {error}); building the demo model.")
    else:
        log("No MODEL_URL is set; building the demo model (synthetic data, NOT FOR CLINICAL USE).")
    build_demo(out, log=log)
    write_model_card(out)
    return "demo"
