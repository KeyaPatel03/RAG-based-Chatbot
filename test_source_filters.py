import unittest

from source_filters import filter_sources_for_query, should_omit_sources_section


class SourceFilterTests(unittest.TestCase):
    def test_filters_sources_to_named_university(self) -> None:
        query = "What are the UCLA transfer admission requirements?"
        sources = [
            "https://admission.ucla.edu/apply/transfer",
            "https://admission.lmu.edu/apply",
            "https://studentaid.gov/apply-for-aid/fafsa/filling-out",
        ]

        self.assertEqual(
            filter_sources_for_query(query, sources),
            ["https://admission.ucla.edu/apply/transfer"],
        )

    def test_keeps_general_sources_when_query_is_not_university_specific(self) -> None:
        query = "How do I apply for financial aid?"
        sources = [
            "https://admission.ucla.edu/apply/transfer",
            "https://studentaid.gov/apply-for-aid/fafsa/filling-out",
        ]

        self.assertEqual(filter_sources_for_query(query, sources), sources)

    def test_omits_sources_when_university_query_has_no_match(self) -> None:
        query = "What are the UCLA transfer admission requirements?"
        sources = [
            "https://admission.lmu.edu/apply",
            "https://studentaid.gov/apply-for-aid/fafsa/filling-out",
        ]

        self.assertTrue(should_omit_sources_section(query, sources))

    def test_lowercase_university_name_is_detected(self) -> None:
        query = "what are the santa clara university transfer deadlines?"
        sources = [
            "https://admission.ucla.edu/apply/transfer",
            "https://www.westmont.edu/liberal-studies",
        ]

        self.assertTrue(should_omit_sources_section(query, sources))

    def test_matches_university_acronym_from_phrase(self) -> None:
        query = "What deadlines does University of California Los Angeles have?"
        sources = [
            "https://admission.ucla.edu/apply",
            "https://admission.lmu.edu/apply",
        ]

        self.assertEqual(
            filter_sources_for_query(query, sources),
            ["https://admission.ucla.edu/apply"],
        )

    def test_detects_short_university_acronym_query(self) -> None:
        query = "what're courses offered by SCU?"
        sources = [
            "https://catalog.ucsd.edu/curric/VIS-ug.html",
            "https://catalog.ucsd.edu/curric/CGS-ug.html",
            "https://www.westmont.edu/liberal-studies",
        ]

        self.assertTrue(should_omit_sources_section(query, sources))


if __name__ == "__main__":
    unittest.main()