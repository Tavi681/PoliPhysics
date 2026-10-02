"""Shared pytest fixtures / path setup.

Adds the read-only ``ref/`` directory to ``sys.path`` so the reference modules
can be imported by their bare names (``import riemann``, ``import offc``), as they
were written. ``ref/tables.py`` and the other driver scripts must NOT be imported
because they run print-heavy code at import time; only ``riemann`` and ``offc``
expose reusable functions.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

REF_DIR = Path(__file__).resolve().parent.parent / "ref"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))
