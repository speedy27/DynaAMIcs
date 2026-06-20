"""Pytest bootstrap: make the repo root importable so tests can import the
`examples` namespace package (only `eb_jepa` is installed as a distribution)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
