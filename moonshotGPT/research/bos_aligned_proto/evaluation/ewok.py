#!/usr/bin/env python3
"""Research-local EWoK evaluation shim for the BOS prototype.

This keeps BOS prototype imports under `research.bos_aligned_proto.evaluation`
without forking the shared language-model EWoK implementation.
"""

import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROTO_ROOT = os.path.dirname(_THIS_DIR)
_RESEARCH_ROOT = os.path.dirname(_PROTO_ROOT)
_REPO_ROOT = os.path.dirname(_RESEARCH_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from evaluation.ewok import *  # noqa: F401,F403
