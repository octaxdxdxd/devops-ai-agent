"""Tests for assistant response markdown normalization."""

from __future__ import annotations

import unittest

from src.ui.chat import _format_for_markdown


class ChatFormattingTests(unittest.TestCase):
    def test_multiline_assistant_text_is_not_wrapped_in_code_block(self) -> None:
        content = "This is a summary.\nIt has two lines."
        formatted = _format_for_markdown(content)
        self.assertEqual(formatted, content)
        self.assertNotIn("```text", formatted)

    def test_common_section_labels_are_bolded_for_readability(self) -> None:
        content = (
            "Issue Summary: GitLab is degraded\n"
            "Severity: P2\n"
            "Confidence Score: 80\n"
            "Recommended Action: Restart the webservice deployment."
        )
        formatted = _format_for_markdown(content)
        self.assertIn("**Issue Summary:** GitLab is degraded", formatted)
        self.assertIn("**Severity:** P2", formatted)
        self.assertIn("**Confidence Score:** 80", formatted)
        self.assertIn("**Recommended Action:** Restart the webservice deployment.", formatted)


if __name__ == "__main__":
    unittest.main()
