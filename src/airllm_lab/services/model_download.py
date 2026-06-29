"""ModelDownloader: fetch model weights reliably via curl.

``huggingface_hub``'s downloader hangs on large shards on this Windows host
(no bytes, no error), while plain HTTP works fine. So we list the repo's
files via the (small) Hub API and fetch each one with ``curl`` into a local
model directory; the model is then loaded from that directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import list_repo_files

HF_BASE = "https://huggingface.co"
_SKIP_SUFFIXES = (".gitattributes",)


class ModelDownloader:
    """Downloads a model's files to a local directory using curl."""

    def __init__(self, hf_token: str | None = None) -> None:
        """Store the optional Hugging Face token used for downloads."""
        self._token = hf_token

    def download(self, model_id: str, dest_root: str | Path) -> Path:
        """Fetch every repo file of ``model_id`` into ``dest_root/<name>``."""
        dest = Path(dest_root) / model_id.split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        for filename in list_repo_files(model_id, token=self._token):
            if filename.endswith(_SKIP_SUFFIXES):
                continue
            self._fetch_file(model_id, filename, dest)
        return dest

    def _fetch_file(self, model_id: str, filename: str, dest: Path) -> None:
        """Download a single repo file with curl (resumable, with retries)."""
        target = dest / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_BASE}/{model_id}/resolve/main/{filename}"
        cmd = ["curl.exe", "-fSL", "--retry", "5", "--retry-delay", "3", "-C", "-"]
        if self._token:
            cmd += ["-H", f"Authorization: Bearer {self._token}"]
        cmd += ["-o", str(target), url]
        subprocess.run(cmd, check=True)
