"""Offline verification runner for the Active Learning / RAG triage pipeline.

Runs the dependency-free ``unittest`` suite that covers:

* rambling Whisper-style STT typos -> cleaned clinical query + OOD + one flag;
* flag -> ticket -> doctor confirmation -> ``DOCTOR_CONFIRMED``;
* RAG training -> ``target_audience='tiep_don'`` + canonical ``age_group`` in the
  training report and retrieval with those metadata filters;
* ICD-10 description sanity, multi-specialty routing and flag de-duplication.

No PostgreSQL, Groq or filesystem writes are required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(PROJECT_ROOT / "tests"),
        pattern="test_*.py",
        top_level_dir=str(PROJECT_ROOT / "tests"),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("ALL_TRIAGE_PIPELINE_VERIFICATIONS_PASSED")
        return 0
    print("TRIAGE_PIPELINE_VERIFICATIONS_FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
