"""Secret access: load ``.env`` and read tokens via the environment only.

Secrets are never hardcoded; they are read through ``os.environ`` after
loading the (gitignored) ``.env`` file.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv


def get_hf_token() -> str | None:
    """Return ``HF_TOKEN`` from the environment, or ``None`` if unset/blank.

    Searches for ``.env`` from the current working directory upward (the repo
    root when commands are run from there), since the installed package lives
    in ``site-packages`` and cannot locate the repo via its own path.
    """
    load_dotenv(find_dotenv(usecwd=True), override=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    return token or None
