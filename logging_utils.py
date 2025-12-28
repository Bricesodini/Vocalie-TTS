"""Utilities to control verbosity for terminal logs and progress bars."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import Iterator


_QUIET_MODE = True


def set_verbosity(verbose: bool) -> None:
    global _QUIET_MODE
    _QUIET_MODE = not verbose

    if verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        os.environ.pop("TQDM_DISABLE", None)
        warnings.filterwarnings("default")
    else:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TQDM_DISABLE"] = "1"
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")

    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)

    logging.getLogger("chatterbox_tts").setLevel(logging.INFO)
    logging.getLogger("chatterbox.models.t3.t3").setLevel(logging.ERROR if not verbose else logging.INFO)
    logging.getLogger("chatterbox.models.t3").setLevel(logging.ERROR if not verbose else logging.INFO)
    logging.getLogger("chatterbox").setLevel(logging.ERROR if not verbose else logging.INFO)
    logging.getLogger("tqdm").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING if not verbose else logging.INFO)


def is_verbose() -> bool:
    return not _QUIET_MODE


@contextlib.contextmanager
def verbosity_context(verbose: bool) -> Iterator[None]:
    previous = _QUIET_MODE
    set_verbosity(verbose)
    try:
        yield
    finally:
        set_verbosity(not previous)
