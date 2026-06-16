from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from followup_refs import resolve_followup_query


class ResolveFollowupQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = [
            {
                "user": "What scholarships exist?",
                "bot": "Some answer",
                "followups": [
                    "What deadlines should I watch?",
                    "How do I compare award letters?",
                    "Where can I find more scholarships?",
                ],
            }
        ]

    def test_resolves_q_prefix(self) -> None:
        self.assertEqual(resolve_followup_query("Q1", self.memory), "What deadlines should I watch?")
        self.assertEqual(resolve_followup_query("q2", self.memory), "How do I compare award letters?")

    def test_resolves_question_phrase(self) -> None:
        self.assertEqual(resolve_followup_query("answer question 1", self.memory), "What deadlines should I watch?")
        self.assertEqual(resolve_followup_query("question 3", self.memory), "Where can I find more scholarships?")

    def test_leaves_non_matches_unchanged(self) -> None:
        self.assertEqual(resolve_followup_query("Tell me more about grants", self.memory), "Tell me more about grants")


if __name__ == "__main__":
    unittest.main()