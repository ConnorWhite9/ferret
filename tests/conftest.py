"""
Pytest configuration: ensure the project root is on sys.path so that
top-level packages (eval, policy, train, etc.) can be imported without
installing the package.
"""

import sys
from pathlib import Path

# Insert the project root (parent of this file's directory) at the front of sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
