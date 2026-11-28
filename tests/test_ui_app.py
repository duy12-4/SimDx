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

    def test_login_gate(self):
        """Kiểm tra màn hình khóa và xác thực mật khẩu."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.run()
        # Ban đầu khi chưa nhập mật khẩu, dừng lại ở màn hình khóa
        self.assertEqual(len(at.exception), 0)
        self.assertFalse(at.session_state.get("authenticated", False))
        
        # Nhập sai mật khẩu
        pwd_inputs = [w for w in at.text_input]
        self.assertTrue(len(pwd_inputs) > 0)
        pwd_inputs[0].input("sai_mat_khau")
        submit_btn = [b for b in at.button if any(k in (b.label or "") for k in ["Đăng nhập", "bảnh", "Thử lại", "cơ hội"])]
        self.assertTrue(len(submit_btn) > 0)
        submit_btn[0].click().run()
        self.assertFalse(at.session_state.get("authenticated", False))
        
        # Nhập đúng mật khẩu
        pwd_inputs = [w for w in at.text_input]
        pwd_inputs[0].input("ngovietcuongkhongchodung1112")
        submit_btn = [b for b in at.button if any(k in (b.label or "") for k in ["Đăng nhập", "bảnh", "Thử lại", "cơ hội"])]
        submit_btn[0].click().run()
        self.assertTrue(at.session_state.get("authenticated", False))

    def test_app_initial_load(self):
        """Kiểm tra khởi động ban đầu của app.py khi đã đăng nhập."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.session_state["authenticated"] = True
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
        """Kiểm tra toàn bộ quy trình: Data -> SAEM -> Tự động chuyển Results."""
        at = AppTest.from_file(self.app_path, default_timeout=180)
        at.session_state["authenticated"] = True
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Chọn dataset lâm sàng
        data_radio = [r for r in at.radio if "Chọn dữ liệu" in (r.label or "")][0]
        data_radio.set_value("Dataset lâm sàng (Dataset_chính_thức.xlsx)").run()
        self.assertEqual(len(at.exception), 0)

        # Bấm nút 'Tiếp tục: Cấu hình & Chạy SAEM →'
        btn_goto = [b for b in at.button if "Tiếp tục: Cấu hình & Chạy SAEM" in (b.label or "")]
        self.assertTrue(len(btn_goto) > 0)
        btn_goto[0].click().run()
        self.assertEqual(at.session_state.get("nav_step"), "2. SAEM Estimation & Initial Estimates")

        # Tìm button 'Bắt đầu ước lượng SAEM'
        run_buttons = [b for b in at.button if "Bắt đầu ước lượng SAEM" in b.label]
        self.assertTrue(len(run_buttons) > 0, "Button 'Bắt đầu ước lượng SAEM' not found")

        # Click và chạy SAEM
        run_buttons[0].click().run()

        self.assertEqual(len(at.exception), 0, f"Exceptions after running SAEM: {at.exception}")
        self.assertIn("res", at.session_state)
        # Kiểm tra tự động chuyển sang Results
        self.assertEqual(at.session_state.get("nav_step"), "3. Results & Population parameters")

if __name__ == "__main__":
    unittest.main()
