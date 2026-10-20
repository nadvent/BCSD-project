#!/usr/bin/env python3
"""Convenience root wrapper for src/python/benchmark_torch_compile.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "python"))

from benchmark_torch_compile import main

if __name__ == "__main__":
    main()
