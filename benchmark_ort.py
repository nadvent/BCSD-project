#!/usr/bin/env python3
"""Convenience root wrapper for src/python/benchmark_ort.py"""
import sys
from pathlib import Path

# Add src/python to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))

from benchmark_ort import main

if __name__ == "__main__":
    main()
