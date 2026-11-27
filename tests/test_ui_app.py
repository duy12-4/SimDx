"""
tests/test_ui_app.py - Automated UI & Component Verification for SimDx app.py

Uses Streamlit's official AppTest framework to verify:
1. Initial page load and dataset loading (Dataset_chính_thức.xlsx)
2. All 5 tabs render without exceptions
3. Running SAEM fit populates results and renders Diagnostics, Individual Fits, VPC, Covariates without errors
"""
import os
import sys
import unittest
from streamlit.testing.v1 import AppTest

class TestSimDxUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

    def test_app_initial_load(self):
        """Kiểm tra khởi động ban đầu của app.py."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"App raised exceptions on startup: {at.exception}")

        # Kiểm tra tiêu đề và logo
        all_texts = [str(t.value) for t in list(at.title) + list(at.markdown)]
        self.assertTrue(any("SimDx" in t for t in all_texts))

        # Kiểm tra metric thẻ dữ liệu lâm sàng
        metrics_labels = [m.label for m in at.metric]
        self.assertIn("Bệnh nhân", metrics_labels)
        self.assertIn("Tổng số liều", metrics_labels)
        self.assertIn("Mẫu nồng độ (DV)", metrics_labels)

        # Kiểm tra giá trị metric
        for m in at.metric:
            if m.label == "Bệnh nhân":
                self.assertEqual(str(m.value), "83")
            elif m.label == "Tổng số liều":
                self.assertEqual(str(m.value), "287")
            elif m.label == "Mẫu nồng độ (DV)":
                self.assertEqual(str(m.value), "278")

    def test_app_full_fit_and_all_tabs(self):
        """Kiểm tra toàn bộ 5 tab sau khi chạy SAEM."""
        at = AppTest.from_file(self.app_path, default_timeout=60)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Tìm button 'Bắt đầu ước lượng SAEM'
        run_buttons = [b for b in at.button if "Bắt đầu ước lượng SAEM" in b.label]
        self.assertTrue(len(run_buttons) > 0, "Button 'Bắt đầu ước lượng SAEM' not found")

        # Click và chạy SAEM
        run_buttons[0].click().run()

        self.assertEqual(len(at.exception), 0, f"Exceptions after running SAEM: {at.exception}")
        self.assertIn("res", at.session_state)

if __name__ == "__main__":
    unittest.main()
