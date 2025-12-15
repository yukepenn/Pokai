#!/usr/bin/env python3
"""Single recommended CLI entrypoint for vgc-lab."""

import sys
from pathlib import Path

# Add src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.cli import main

if __name__ == "__main__":
    main()
