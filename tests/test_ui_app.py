"""
tests/test_ui_app.py - Automated UI & Component Verification for SimDx app.py

Uses Streamlit's official AppTest framework to verify:
1. Initial page load and dataset loading (Dataset_chính_thức.xlsx)
2. Workflow gating: Model selection required before unlocking initial estimates & SAEM execution
3. Running SAEM fit populates results and automatically transitions to Results tab
"""
import os
import sys
import unittest
from streamlit.testing.v1 import AppTest

class TestSimDxUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        cls.cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".simdx_session_cache.pkl"))
        if os.path.exists(cls.cache_path):
            os.remove(cls.cache_path)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "cache_path") and os.path.exists(cls.cache_path):
            os.remove(cls.cache_path)

    def setUp(self):
        if hasattr(self, "cache_path") and os.path.exists(self.cache_path):
            os.remove(self.cache_path)

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
        """Kiểm tra khởi động ban đầu của app.py: data ban đầu trắng, sau khi nạp thì hiển thị metric đầy đủ."""
        at = AppTest.from_file(self.app_path, default_timeout=30)
        at.session_state["authenticated"] = True
        at.run()
        self.assertEqual(len(at.exception), 0, f"App raised exceptions on startup: {at.exception}")

        # Kiểm tra tiêu đề và logo
        all_texts = [str(t.value) for t in list(at.title) + list(at.markdown)]
        self.assertTrue(any("SimDx" in t for t in all_texts))

        # Ban đầu chưa nạp dữ liệu, hiển thị hướng dẫn
        load_btn = [b for b in at.button if "Nạp Dataset chuẩn lâm sàng" in (b.label or "")]
        self.assertTrue(len(load_btn) > 0, "Nút nạp dataset mẫu không tìm thấy")
        load_btn[0].click().run()
        self.assertEqual(len(at.exception), 0)

        # Kiểm tra metric thẻ dữ liệu lâm sàng sau khi nạp
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
        """Kiểm tra toàn bộ quy trình: Data -> Chọn & Xác nhận Model -> SAEM -> Tự động chuyển Results."""
        at = AppTest.from_file(self.app_path, default_timeout=180)
        at.session_state["authenticated"] = True
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Nạp dataset lâm sàng chuẩn
        load_btn = [b for b in at.button if "Nạp Dataset chuẩn lâm sàng" in (b.label or "")]
        self.assertTrue(len(load_btn) > 0)
        load_btn[0].click().run()
        self.assertEqual(len(at.exception), 0)

        # Bấm nút 'Tiếp tục: Cấu hình & Chạy SAEM →'
        btn_goto = [b for b in at.button if "Tiếp tục: Cấu hình & Chạy SAEM" in (b.label or "")]
        self.assertTrue(len(btn_goto) > 0)
        btn_goto[0].click().run()
        self.assertEqual(at.session_state.get("nav_step"), "2. SAEM Estimation & Initial Estimates")

        # Ban đầu chưa xác nhận mô hình: nút chạy SAEM KHÔNG ĐƯỢC PHÉP xuất hiện
        run_buttons_before = [b for b in at.button if "Bắt đầu ước lượng SAEM" in (b.label or "")]
        self.assertEqual(len(run_buttons_before), 0, "Nút chạy SAEM không được xuất hiện trước khi xác nhận mô hình!")

        # Tìm nút xác nhận mô hình
        btn_confirm = [b for b in at.button if "Xác nhận chọn mô hình này" in (b.label or "")]
        self.assertTrue(len(btn_confirm) > 0, "Nút 'Xác nhận chọn mô hình này' phải tồn tại")
        btn_confirm[0].click().run()
        self.assertEqual(len(at.exception), 0)

        # Sau khi xác nhận mô hình: Nút chạy SAEM và Auto-estimate xuất hiện
        run_buttons = [b for b in at.button if "Bắt đầu ước lượng SAEM" in (b.label or "")]
        self.assertTrue(len(run_buttons) > 0, "Button 'Bắt đầu ước lượng SAEM' not found after confirmation")

        # Test Auto-estimate button
        btn_auto = [b for b in at.button if "Tự động ước lượng Vd, Cl" in (b.label or "")]
        self.assertTrue(len(btn_auto) > 0, "Button 'Tự động ước lượng Vd, Cl' not found")
        btn_auto[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Exception on Auto-estimate: {at.exception}")

        # Tìm lại button 'Bắt đầu ước lượng SAEM' sau khi auto-estimate rerun
        run_buttons = [b for b in at.button if "Bắt đầu ước lượng SAEM" in (b.label or "")]
        self.assertTrue(len(run_buttons) > 0)

        # Click và chạy SAEM
        run_buttons[0].click().run()

        self.assertEqual(len(at.exception), 0, f"Exceptions after running SAEM: {at.exception}")
        self.assertIn("res", at.session_state)
        # Kiểm tra tự động chuyển sang Results
        self.assertEqual(at.session_state.get("nav_step"), "3. Results & Population parameters")

    def test_session_persistence_across_reloads(self):
        """Kiểm tra tính năng lưu trữ dữ liệu & kết quả SAEM cho toàn bộ phiên làm việc."""
        at = AppTest.from_file(self.app_path, default_timeout=180)
        at.session_state["authenticated"] = True
        at.run()
        
        # Nạp dữ liệu
        load_btn = [b for b in at.button if "Nạp Dataset chuẩn lâm sàng" in (b.label or "")][0]
        load_btn.click().run()
        self.assertIsNotNone(at.session_state.get("clinical_data"))

        # Khởi động một phiên AppTest mới (mô phỏng người dùng F5 hoặc mở lại trình duyệt)
        at2 = AppTest.from_file(self.app_path, default_timeout=30)
        at2.session_state["authenticated"] = True
        at2.run()

        # Dữ liệu từ phiên trước phải được tự động khôi phục từ cache!
        self.assertIsNotNone(at2.session_state.get("clinical_data"), "Dữ liệu phải được tự động duy trì và khôi phục trong phiên!")
        self.assertEqual(at2.session_state.get("clinical_data")["N"], 83)

    def test_tab_navigation_preserves_data_and_res(self):
        """Kiểm tra khi chuyển qua lại các tab (Tab 4, 5, 6, 1, 2) dữ liệu và kết quả SAEM không bị mất."""
        at = AppTest.from_file(self.app_path, default_timeout=180)
        at.session_state["authenticated"] = True
        at.run()

        # Nạp data
        load_btn = [b for b in at.button if "Nạp Dataset chuẩn lâm sàng" in (b.label or "")][0]
        load_btn.click().run()

        # Sang tab 2
        btn_goto = [b for b in at.button if "Tiếp tục: Cấu hình & Chạy SAEM" in (b.label or "")][0]
        btn_goto.click().run()

        # Xác nhận mô hình và chạy SAEM
        btn_confirm = [b for b in at.button if "Xác nhận chọn mô hình này" in (b.label or "")][0]
        btn_confirm.click().run()
        run_btn = [b for b in at.button if "Bắt đầu ước lượng SAEM" in (b.label or "")][0]
        run_btn.click().run()

        self.assertIsNotNone(at.session_state.get("res"), "SAEM fit phải có kết quả")

        # Chuyển tuần tự qua tất cả các tab: Tab 3, Tab 4 (GOF), Tab 5 (EBE), Tab 6 (Covariates), Tab 7 (Equations)
        tabs_to_test = [
            "3. Results & Population parameters",
            "4. Diagnostic Plots (GOF)",
            "5. Individual fits (EBE)",
            "6. Covariates & Simulx",
            "7. Model Equations (Phương trình PopPK)",
            "1. Data",
            "2. SAEM Estimation & Initial Estimates",
            "7. Model Equations (Phương trình PopPK)"
        ]

        for tab_name in tabs_to_test:
            nav_radio = [r for r in at.sidebar.radio if "Chọn bước phân tích" in (r.label or "")][0]
            nav_radio.set_value(tab_name).run()
            self.assertEqual(len(at.exception), 0, f"Exception khi vào tab {tab_name}: {at.exception}")
            self.assertIsNotNone(at.session_state.get("clinical_data"), f"clinical_data bị mất khi vào tab {tab_name}")
            self.assertIsNotNone(at.session_state.get("res"), f"res bị mất khi vào tab {tab_name}")
            
            # Khẳng định không có cảnh báo 'Vui lòng chạy SAEM ở bước 2 trước'
            warnings_text = [w.value for w in at.warning]
            self.assertFalse(any("chạy SAEM ở bước 2 trước" in str(w) for w in warnings_text),
                             f"Cảnh báo yêu cầu chạy lại SAEM xuất hiện tại {tab_name}: {warnings_text}")

        # Thử bấm nút 'Đặt lại phiên làm việc' ở sidebar - không được ném StreamlitWidgetAlreadyInstantiatedError
        reset_btn = [b for b in at.sidebar.button if "Đặt lại phiên làm việc" in (b.label or "")][0]
        reset_btn.click().run()
        self.assertEqual(len(at.exception), 0, f"Exception khi bấm Đặt lại phiên làm việc: {at.exception}")
        self.assertIsNone(at.session_state.get("clinical_data"), "Dữ liệu phải được xóa sau khi đặt lại phiên")
        self.assertEqual(at.session_state.get("nav_step"), "1. Data")

if __name__ == "__main__":
    unittest.main()
