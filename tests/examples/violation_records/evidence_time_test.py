from __future__ import annotations

import os
import time
import unittest
from datetime import datetime
from datetime import timezone

from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.evidence_fingerprint import (
    evidence_fingerprint,
)
from src.evidence import EVIDENCE_METADATA


class EvidenceTimeTest(unittest.TestCase):
    def test_naive_times_have_the_same_utc_meaning_across_process_timezones(
        self,
    ):
        original = os.environ.get('TZ')
        try:
            results = []
            for zone in ('UTC', 'Asia/Taipei'):
                os.environ['TZ'] = zone
                time.tzset()
                evidence = ViolationEvidence(
                    site='Site',
                    site_id=1,
                    stream_name='Cam',
                    image_file=None,
                    detection_time=datetime(2026, 11, 3, 12),
                )
                results.append(evidence_fingerprint(evidence, 'a' * 64))
                self.assertEqual(
                    evidence.detection_time,
                    datetime(2026, 11, 3, 12, tzinfo=timezone.utc),
                )
            self.assertEqual(*results)
        finally:
            if original is None:
                os.environ.pop('TZ', None)
            else:
                os.environ['TZ'] = original
            time.tzset()

    def test_http_and_outbox_metadata_normalize_explicit_offsets_together(
        self,
    ):
        data = {
            'site': 'Site',
            'stream_name': 'Cam',
            'detection_time': '2026-11-03T20:00:00+08:00',
        }
        parsed = EVIDENCE_METADATA.validate_python(data)
        self.assertEqual(
            parsed.detection_time,
            datetime(2026, 11, 3, 12, tzinfo=timezone.utc),
        )
        reparsed = EVIDENCE_METADATA.validate_python(
            {**data, 'detection_time': '2026-11-03T12:00:00'},
        )
        self.assertEqual(parsed.detection_time, reparsed.detection_time)
