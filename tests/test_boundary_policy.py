from __future__ import annotations

import unittest

from starter.src.config import (
    BOUNDARY_BROAD_REASK_ATTRIBUTE,
    BOUNDARY_BROAD_REASK_MESSAGE,
    should_broad_reask_after_boundary,
)


class BoundaryPolicyTest(unittest.TestCase):
    def test_initial_boundary_decline_requests_broad_other_follow_up(self) -> None:
        self.assertTrue(
            should_broad_reask_after_boundary(
                is_no_preference=True,
                attributes_asked=["feature"],
            )
        )
        self.assertEqual(BOUNDARY_BROAD_REASK_ATTRIBUTE, "other")
        self.assertTrue(BOUNDARY_BROAD_REASK_MESSAGE)

    def test_later_declines_do_not_repeatedly_force_broad_follow_up(self) -> None:
        self.assertFalse(
            should_broad_reask_after_boundary(
                is_no_preference=True,
                attributes_asked=["feature", "material"],
            )
        )

    def test_non_boundary_response_does_not_change_question_selection(self) -> None:
        self.assertFalse(
            should_broad_reask_after_boundary(
                is_no_preference=False,
                attributes_asked=["feature"],
            )
        )


if __name__ == "__main__":
    unittest.main()
