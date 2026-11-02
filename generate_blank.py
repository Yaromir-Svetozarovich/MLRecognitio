#!/usr/bin/env python3
"""Backward-compatible entry point. Delegates to app.cli."""

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())