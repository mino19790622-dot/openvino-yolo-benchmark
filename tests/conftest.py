# tests/conftest.py
"""Shared test setup for the benchmark suite.

``yolo_utils.py`` is deliberately torch-free *and* OpenVINO-free, so the
pre-processing / NMS / decoding maths can be exercised on any machine without
installing the inference stack. Only numpy and OpenCV are needed.
"""

import os
import sys

# allow running pytest from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
