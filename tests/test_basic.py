"""Basic tests to verify the test configuration is working correctly."""

import unittest


def test_simple():
    """A simple test function that will always pass."""
    assert True


class TestBasic(unittest.TestCase):
    """A simple test class with basic tests."""

    def test_assertTrue(self):
        """Verify that assertTrue works."""
        self.assertTrue(True)

    def test_assertEqual(self):
        """Verify that assertEqual works."""
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main() 