"""Make the ``ballooning`` package importable when running pytest from this dir."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
