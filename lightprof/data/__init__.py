"""Data loading and preprocessing for LightPROF.

This package provides dataset loaders for WebQSP and CWQ benchmarks,
as well as utilities for loading and processing Freebase knowledge graphs.

Modules:
    freebase: Freebase Knowledge Graph loader with BFS retrieval.
    webqsp: WebQSP dataset loader (4,737 questions, 2-hop).
    cwq: ComplexWebQuestions dataset loader (34,689 questions, 4-hop).
"""

from __future__ import annotations

from lightprof.data.cwq import CWQDataset, CWQSample
from lightprof.data.freebase import FreebaseKG
from lightprof.data.webqsp import WebQSPDataset, WebQSPSample

__all__ = [
    "CWQDataset",
    "CWQSample",
    "FreebaseKG",
    "WebQSPDataset",
    "WebQSPSample",
]
