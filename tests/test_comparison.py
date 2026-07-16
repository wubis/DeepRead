import unittest

from deepread.comparison import compare_results


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_aligns_questions_and_reports_delta(self):
        baseline = {
            "rows": [
                {"question_id": "q1", "status": "ok", "answer_f1": 0.0},
                {"question_id": "q2", "status": "ok", "answer_f1": 0.5},
                {"question_id": "q3", "status": "error", "answer_f1": 1.0},
            ]
        }
        candidate = {
            "rows": [
                {"question_id": "q1", "status": "ok", "answer_f1": 1.0},
                {"question_id": "q2", "status": "ok", "answer_f1": 0.5},
                {"question_id": "q4", "status": "ok", "answer_f1": 1.0},
            ]
        }

        result = compare_results(
            baseline,
            candidate,
            ["answer_f1"],
            iterations=200,
            seed=17,
        )

        metric = result["metrics"]["answer_f1"]
        self.assertEqual(result["paired_questions"], 2)
        self.assertEqual(metric["pairs"], 2)
        self.assertEqual(metric["mean_delta"], 0.5)
        self.assertEqual(metric["improvement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
