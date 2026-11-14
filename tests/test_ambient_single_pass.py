"""Tests for the unified-transcript single-pass ambient synthesis."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services import ambient
from app.services.clinical_client import AvailableDrug

TRANSCRIPT = (
    "Bác sĩ: Chào anh, hôm nay anh khó chịu ở đâu? "
    "Bệnh nhân: Tôi đau họng và sốt hai ngày nay."
)


class SinglePassAmbientSynthesisTests(unittest.TestCase):
    def test_extracts_contract_fields_from_unified_transcript(self) -> None:
        drug = AvailableDrug(
            id=7,
            code="PARA500",
            name="Paracetamol",
            concentration="500mg",
            dosage_form="Viên nén",
            stock_quantity=100,
        )
        payload = {
            "chief_complaints": ["Đau họng", "Sốt"],
            "symptoms": ["Đau họng 2 ngày", "Sốt 38.5 độ"],
            "has_red_flags": True,
            "red_flag_alert": "Cần khai thác khó thở",
            "recommended_inquiry": "Anh có khó thở không?",
            "matched_patterns": ["sốt"],
            "soap_note": {
                "subjective": "Bệnh nhân đau họng, sốt 2 ngày.",
                "objective": "Chưa có thông tin",
                "assessment": "Viêm họng cấp",
                "plan": "Theo dõi, tái khám nếu nặng.",
            },
            "diagnoses": [
                {
                    "icd10Code": "J06.9",
                    "diseaseName": "Viêm họng cấp",
                    "isPrimary": True,
                    "aiConfidence": 85,
                }
            ],
            "prescription_draft": [
                {
                    "drugId": 7,
                    "drugName": "Paracetamol",
                    "routeFrequency": "Uống 2 lần/ngày",
                    "duration": "5 ngày",
                    "quantity": 10,
                }
            ],
        }

        with patch.object(ambient, "complete_json", return_value=payload) as mocked:
            draft, red_flags, complaints, symptoms = (
                ambient.single_pass_ambient_synthesis(
                    TRANSCRIPT,
                    "[HỒ SƠ LÂM SÀNG] test",
                    "",
                    [drug],
                    "test-model",
                    None,
                )
            )

        user_prompt = mocked.call_args.args[1]
        self.assertIn(TRANSCRIPT, user_prompt)
        self.assertEqual(complaints, ["Đau họng", "Sốt"])
        self.assertEqual(symptoms, ["Đau họng 2 ngày", "Sốt 38.5 độ"])
        self.assertTrue(red_flags.has_red_flags)
        self.assertEqual(red_flags.recommended_inquiry, "Anh có khó thở không?")
        self.assertEqual(draft.soap_note.subjective, "Bệnh nhân đau họng, sốt 2 ngày.")
        self.assertEqual([d.icd10Code for d in draft.diagnoses], ["J06.9"])
        self.assertEqual([p.drugId for p in draft.prescription_draft], [7])


if __name__ == "__main__":
    unittest.main()
