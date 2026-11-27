"""
tests/test_data_io.py - Tests for data_io.py and Gate G4.
G4: Loader reproduces exactly: 565 rows / 83 subjects / 287 doses / 278 observations. Otherwise fail loudly.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_io import load_dataset, export_monolix


class TestDataIO(unittest.TestCase):

    def setUp(self):
        self.excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dataset_chính_thức.xlsx"))

    def test_g4_exact_counts(self):
        """
        Gate G4: Loader reproduces exactly: 565 rows / 83 subjects / 287 doses / 278 observations.
        """
        self.assertTrue(os.path.exists(self.excel_path), f"File {self.excel_path} must exist.")
        data = load_dataset(self.excel_path, tinf=1.0, enforce_g4=True)
        
        self.assertEqual(len(data["clean_df"]), 565, f"Expected 565 rows, got {len(data['clean_df'])}")
        self.assertEqual(data["N"], 83, f"Expected 83 subjects, got {data['N']}")
        
        n_doses = sum(len(d) for d in data["sub_doses"].values())
        self.assertEqual(n_doses, 287, f"Expected 287 doses, got {n_doses}")
        
        self.assertEqual(data["nobs"], 278, f"Expected 278 observations, got {data['nobs']}")
        print(f"G4 PASS: 565 rows / 83 subjects / 287 doses / 278 observations strictly verified.")

    def test_g4_fails_loudly_on_mismatch(self):
        """
        Ensure load_dataset with enforce_g4=True raises ValueError loudly if any count mismatches.
        """
        # Create a dummy dataframe with wrong row count
        dummy_df = pd.DataFrame({
            "ID": [1, 1], "Time": [0.0, 1.0], "Amount": [1000.0, "."], "DV": [".", 20.0]
        })
        dummy_path = "/tmp/dummy_test_mismatch.csv"
        dummy_df.to_csv(dummy_path, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_dataset(dummy_path, enforce_g4=True)
            self.assertIn("Gate G4 Verification Failed loudly", str(ctx.exception))
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_duplicate_id_time_raises_error(self):
        """
        Verify that duplicate (ID, Time) raises ValueError.
        """
        dummy_df = pd.DataFrame({
            "ID": [1, 1], "Time": [0.0, 0.0], "Amount": [1000.0, "."], "DV": [".", 20.0]
        })
        dummy_path = "/tmp/dummy_test_dup.csv"
        dummy_df.to_csv(dummy_path, index=False)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_dataset(dummy_path, enforce_g4=False)
            self.assertIn("duplicate (ID, Time)", str(ctx.exception))
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    def test_outlier_flagging_without_modification(self):
        """
        Check that outliers (ID 76, 68, 10) are flagged and NOT dropped or modified.
        """
        data = load_dataset(self.excel_path, tinf=1.0)
        outliers = data["outliers"]
        outlier_ids = [o["ID"] for o in outliers]
        self.assertIn(76, outlier_ids)
        self.assertIn(68, outlier_ids)
        self.assertIn(10, outlier_ids)
        
        # Check that ID 76 DV=124.9 is still in y matrix
        id76_idx = np.where(data["ids"] == 76)[0][0]
        id76_y = data["y"][id76_idx, data["mask"][id76_idx]]
        self.assertTrue(any(abs(v - 124.9) < 1e-4 for v in id76_y), "ID 76 DV=124.9 must be retained")

    def test_export_monolix(self):
        """
        Check export_monolix produces valid CSV and mapping documentation.
        """
        data = load_dataset(self.excel_path, tinf=1.0)
        out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "validation", "export_monolix.csv"))
        csv_path, map_path = export_monolix(data, out_csv)
        
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(map_path))
        
        m_df = pd.read_csv(csv_path)
        self.assertEqual(len(m_df), 565)
        self.assertEqual(m_df["ID"].nunique(), 83)
        self.assertEqual((m_df["EVID"] == 1).sum(), 287)
        self.assertEqual((m_df["EVID"] == 0).sum(), 278)
        self.assertIn("TINFUSION", m_df.columns)
        print(f"Monolix Export PASS: {csv_path} with {len(m_df)} rows and correct EVID.")


if __name__ == "__main__":
    unittest.main()
