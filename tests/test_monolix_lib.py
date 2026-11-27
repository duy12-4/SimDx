"""
tests/test_monolix_lib.py - Kiểm tra tính toàn vẹn và tính đúng đắn của Monolix Model Library.
"""
import unittest
from monolix_library import MONOLIX_LIBRARY

class TestMonolixLibrary(unittest.TestCase):
    def test_library_not_empty(self):
        self.assertGreater(len(MONOLIX_LIBRARY), 20)

    def test_model_fields(self):
        required_fields = ["id", "category", "admin", "delay", "absorption", "distrib", "elim", "param", "params", "defaults", "ranges", "ode", "solution"]
        for m in MONOLIX_LIBRARY:
            for f in required_fields:
                self.assertIn(f, m, f"Model {m.get('id')} missing field {f}")
            self.assertEqual(len(m["params"]), len(m["defaults"]), f"Length mismatch params vs defaults in {m['id']}")
            self.assertEqual(len(m["params"]), len(m["ranges"]), f"Length mismatch params vs ranges in {m['id']}")

    def test_categories_present(self):
        cats = set(m["category"] for m in MONOLIX_LIBRARY)
        self.assertIn("PK", cats)
        self.assertIn("PD", cats)
        self.assertIn("PKPD", cats)
        self.assertIn("TMDD", cats)

if __name__ == "__main__":
    unittest.main()
