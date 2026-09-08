"""Entrypoint for python -m llm_reliability.cli."""

import sys

from llm_reliability.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
