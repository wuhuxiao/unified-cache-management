# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import ctypes
import math
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: test_cpp_python_bridge.py <cpp-writer-so> <ucmmetrics-dir>"
        )

    writer_path = Path(sys.argv[1])
    ucmmetrics_dir = Path(sys.argv[2])
    sys.path.insert(0, str(ucmmetrics_dir))

    import ucmmetrics

    ucmmetrics.set_up(100)
    ucmmetrics.create_stats("cpp_bridge_counter", "counter")
    ucmmetrics.create_stats("cpp_bridge_gauge", "gauge")
    ucmmetrics.create_stats("cpp_bridge_histogram", "histogram")

    writer = ctypes.CDLL(str(writer_path))
    writer.WriteMetricsFromCpp.argtypes = []
    writer.WriteMetricsFromCpp.restype = None
    writer.WriteMetricsFromCpp()

    counters, gauges, histograms = ucmmetrics.get_all_stats_and_clear()

    assert math.isclose(counters["cpp_bridge_counter"], 5.5)
    assert math.isclose(gauges["cpp_bridge_gauge"], 9.0)
    assert histograms["cpp_bridge_histogram"] == [1.25, 2.5]


if __name__ == "__main__":
    main()
