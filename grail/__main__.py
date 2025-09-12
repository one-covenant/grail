"""Module entry point to support `python -m grail`.

This delegates to the Typer CLI defined in `grail.cli:main`.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
