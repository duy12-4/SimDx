# 💊 SimDx — Nonlinear Mixed-Effects Pharmacometrics Modeling & Simulation

Nền tảng NLME mã nguồn mở chuẩn hoá theo phương pháp luận Monolix, phục vụ mô hình hoá Dược động học quần thể (Population PK) trên dữ liệu lâm sàng thực tế với truyền tĩnh mạch lặp lại, mẫu thưa, và các hiệp biến chức năng thận biến thiên theo thời gian.

---

## 🚀 Tính năng chính

1. **Bộ giải cấu trúc biến cố (Event-driven Structural Solver)**:
   - Nghiệm giải tích chính xác từng đoạn (piecewise exact) cho truyền tĩnh mạch lặp lại (multi-dose IV infusion).
   - Xử lý truyền chồng lấn (overlapping infusions) bằng cách cộng dồn tốc độ truyền $R = \sum \text{Dose}/t_{inf}$.
   - Tự động bắt cặp biến cố hiệp biến thời gian (Cockcroft-Gault, Creatinine, eGFR) với quy tắc LOCF (Last Observation Carried Forward).
   - Đạt sai số tương đối tối đa $< 10^{-12}$ so với giải tích chuẩn và solver ODE `solve_ivp(Radau, rtol=1e-10)`.

2. **Thuật toán SAEM (Stochastic Approximation Expectation-Maximization)**:
   - 2 pha hội tụ: Pha 1 ($\gamma = 1$) với simulated annealing làm mượt không gian mẫu; Pha 2 ($\gamma = 1 / (k - K_1)$) hội tụ tiệm cận về MLE.
   - E-step: MCMC 3-kernel thích nghi (random walk, multidimensional Gaussian, independent proposal).
   - 4 mô hình sai số thặng dư chuẩn Monolix:
     - Constant: $g = a$
     - Proportional: $g = b \cdot f$
     - Combined 1: $g = a + b \cdot f$
     - Combined 2: $g = \sqrt{a^2 + b^2 f^2}$
   - Đánh giá hàm hợp lý cận biên $-2LL$ bằng Student-$t_5$ Importance Sampling kèm sai số Monte Carlo ($MC\text{-}SE$).
   - Tiêu chuẩn thông tin: $AIC$, $BIC$, $BICc$ (sử dụng $N$ bệnh nhân cho tham số hiệu ứng ngẫu nhiên và $n_{obs}$ cho tham số cố định/sai số).
   - Sai số chuẩn $SE$, $RSE\%$ dựa trên ma trận Fisher tuyến tính hoá (Linearised FIM).

3. **Xử lý Dữ liệu Lâm sàng (`data_io.py`)**:
   - Tự động nạp file `Dataset_chính_thức.xlsx`, giải mã ký tự khuyết thiếu `.`
   - Kiểm tra tính toàn vẹn 565 dòng / 83 bệnh nhân / 287 liều / 278 điểm đo nồng độ DV.
   - Gắn cờ (flag) ngoại lai theo tiêu chí lâm sàng (ví dụ ID 76, ID 68, ID 10) mà không làm biến dạng dữ liệu gốc.
   - Xuất dữ liệu chuẩn Monolix (`validation/export_monolix.csv`) kèm bảng ánh xạ chi tiết.

4. **Chẩn đoán mô hình & VPC**:
   - Đồ thị Goodness of Fit: DV vs PRED, DV vs IPRED, IWRES vs PRED/Time, NPDE phân phối chuẩn.
   - Đường cong nồng độ cá thể trên lưới mịn từng giờ (fine grid) với đánh dấu thời điểm truyền dịch.
   - pcVPC (Prediction-Corrected Visual Predictive Check) mô phỏng lại toàn bộ thiết kế lâm sàng thực tế.

---

## 🛠️ Lỗi Xác Định Bệnh Nhân (`subject_ids`) & Bản Sửa Lỗi

### 1. Cơ chế phát sinh lỗi
Trong hàm `predict()` của `pk_core.py`, việc xác định chỉ số bệnh nhân trước đây dùng `subj_idx = m % N`:
- Biểu thức này chỉ đúng khi ma trận $\phi$ chứa đầy đủ $N$ bệnh nhân theo thứ tự (Full-$N$ batch path: E-step của SAEM, `compute_fim`, VPC).
- Khi các hàm gọi truyền ma trận $\phi$ cho một cá thể duy nhất ($M = 1$ trong `compute_ebe`, $M = n_{is}$ trong `loglik_is`, hoặc $M = K$ trong NPDE), biểu thức `0 % N = 0` dẫn đến **luôn luôn lấy dữ liệu liều và hiệp biến thời gian của bệnh nhân ID 1**.
- Điều này gây sai lệch nghiêm trọng trên mô hình hiệp biến thời gian $CL \sim \text{CLCr\_CG}$: IPRED bị lệch $28\% - 120\%$, kéo theo $-2LL$, $\Delta\text{OFV}$, $SE$, và Shrinkage bị biến dạng. Mô hình Base Model không có hiệp biến thì không bị ảnh hưởng.
- Đồng thời tồn tại các giá trị ngầm định (silent defaults) `52.8415` và `0.75` che giấu việc thiếu dữ liệu đầu vào.

### 2. Giải pháp khắc phục
1. **Target State A**: Thêm tham số `subject_ids` tường minh vào `predict()` và `loglik_rows()`. Mọi vị trí gọi đều truyền rõ danh tính bệnh nhân. Nếu có hiệp biến thời gian mà `subject_ids` bị thiếu khi $M \neq N$, hệ thống lập tức ném ngoại lệ `ValueError`. Đường gọi Full-$N$ giữ nguyên 100% kết quả tính toán.
2. **Target State B**: Loại bỏ triệt để các silent defaults `52.8415` và `0.75`. Nếu thiếu `ref_medians` hoặc thiếu hệ số $\beta$, hệ thống ném `KeyError` / `ValueError`.

### 3. Kết quả thẩm định sau sửa lỗi
- **F0 (Pre-fix failing proof)**: Đã chứng minh lỗi trên code cũ: F1 FAIL (sai số $516\% \ge 10^{-8}$), F3 FAIL (lệch loglik $3887.96 \ge 10^{-9}$).
- **F1 (Single-subject vs independent solve_ivp)**: Max relative error $= 5.95 \times 10^{-15} < 10^{-8}$ (**PASS**). Khớp tuyệt đối với batch call (diff $= 0.0$).
- **F2 (IPRED vs reference solver across all 83 subjects)**: Max relative error $= 7.58 \times 10^{-15} < 10^{-8}$ (**PASS**).
- **F3 (Deterministic consistency)**: Lệch `loglik_rows` giữa single-subject và Full-$N$ $= 0.0 < 10^{-9}$ (**PASS**).
- **F4 (Hồi quy G1–G4 & Base Model)**: Toàn bộ test cũ PASS; Base Model $-2LL = 2015.26 \pm 0.01$ (**PASS**).
- **F5 (Khớp lại $CL \sim \text{CLCr\_CG}$)**: $-2LL = 1937.33$, $\Delta\text{OFV} = -77.93$ (trùng khớp chính xác tuyệt đối với kết quả đối chứng độc lập 1937.33, sai lệch $0.00 < 1.0$).
- **F6 (Covariate Simulation-Estimation 15 reps)**: Tỷ lệ không hội tụ $0.0\%$, median bias $\beta_{\text{CLCr}} = 5.56\% < 10\%$, bias $\theta_{CL} = 2.54\%$, bias $\theta_V = 2.26\%$, SE ratios trong khoảng $[0.97, 1.06] \in [0.75, 1.33]$ (**PASS**).

---

## 📁 Cấu trúc Dự án

```
.
├── Dataset_chính_thức.xlsx       # Bộ dữ liệu lâm sàng gốc (không sửa đổi)
├── pk_core.py                     # Lõi giải tích PK, SAEM, FIM, -2LL IS, Diagnostics, VPC
├── data_io.py                     # Module nạp, tiền xử lý, kiểm tra dữ liệu và xuất Monolix
├── app.py                         # Giao diện web tương tác SimDx (Streamlit)
├── test_pk_core.py                # Test hồi quy lõi PK cơ bản
├── tests/
│   ├── test_solver.py             # Gate G1 & G2: Kiểm thử bộ giải 1-Cpt Infusion
│   ├── test_data_io.py            # Gate G4: Kiểm thử nạp dữ liệu lâm sàng
│   ├── test_per_subject.py        # Gates F1–F3: Kiểm thử định danh bệnh nhân & ODE solver độc lập
│   └── test_ui_app.py             # Kiểm thử tự động hóa giao diện Streamlit (AppTest)
└── validation/
    ├── run_simulation_estimation.py     # Gate G5: Mô phỏng - Ước lượng Base Model (30 reps)
    ├── run_simulation_estimation_cov.py # Gate F6: Mô phỏng - Ước lượng Covariate Model (15 reps)
    ├── run_multistart.py                # Gate G6: Đa khởi tạo 0.5x - 2.0x
    ├── run_sensitivity_tinf.py          # Gate G8: Độ nhạy thời gian truyền tinf
    ├── run_covariate_analysis.py        # Gate F5: Phân tích 5 mô hình hiệp biến lâm sàng
    ├── run_saemix.R                     # Script đối chứng độc lập saemix (R)
    ├── run_nlmixr2.R                    # Script đối chứng độc lập nlmixr2 (R)
    ├── export_monolix.csv               # Dữ liệu xuất định dạng Monolix
    └── export_monolix_mapping.md        # Tài liệu quy tắc ánh xạ Monolix
```

---

## 🛠️ Hướng dẫn Cài đặt & Chạy

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10+:
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Giao diện Web (SimDx UI)
```bash
streamlit run app.py
```
Mở trình duyệt tại: `http://localhost:8502`

### 3. Chạy Toàn Bộ Kiểm Thử Đảm Bảo Độ Đúng Đắn

- **Bộ kiểm thử cá thể & sửa lỗi (Gates F1, F2, F3)**:
  ```bash
  python3 -m unittest tests/test_per_subject.py
  ```
- **Kiểm thử hồi quy (Gates G1, G2, G3, G4)**:
  ```bash
  python3 -m unittest tests/test_solver.py tests/test_data_io.py test_pk_core.py
  ```
- **Mô phỏng - Ước lượng mô hình hiệp biến (Gate F6, 15 replicates)**:
  ```bash
  python3 validation/run_simulation_estimation_cov.py 15
  ```
- **Chạy phân tích hiệp biến lâm sàng (Gate F5)**:
  ```bash
  python3 validation/run_covariate_analysis.py
  ```

---

## 📊 Kết Quả Thẩm Định Kỹ Thuật (Acceptance Gates)

### Bảng 1: Các Cổng Thẩm Định Sửa Lỗi (Gates F0 – F6)

| Cổng | Mô tả kiểm định | Ngưỡng chấp thuận | Kết quả thực đo | Trạng thái |
|---|---|---|---|:---:|
| **F0** | Kiểm chứng lỗi trước khi sửa trên code gốc | Bắt buộc FAIL | F1 error = $516\%$, F3 loglik diff = $3887.96$ | **PASS (Đã chứng minh có bug)** |
| **F1** | Single-subject predict vs independent `solve_ivp` ($\ge 10$ BN) | Max rel error $< 10^{-8}$ | **Max rel error: $5.95 \times 10^{-15}$**; batch diff = $0.0$ | **PASS** |
| **F2** | IPRED vs reference solver trên toàn bộ 83 bệnh nhân | Max rel error $< 10^{-8}$ | **Max rel error: $7.58 \times 10^{-15}$** | **PASS** |
| **F3** | Tính nhất quán tiền định single-subject vs Full-$N$ | Abs diff $< 10^{-9}$ | **Abs diff: $0.0000 \times 10^0$** | **PASS** |
| **F4** | Hồi quy G1–G4 & Base Model $-2LL$ | G1–G4 PASS; Base $-2LL = 2015.26 \pm 0.01$ | **8/8 unit tests PASS; Base $-2LL = 2015.2562$** | **PASS** |
| **F5** | Khớp lại $CL \sim \text{CLCr\_CG}$ (seed=42) | Đối chiếu với tham chiếu 1937.33 (sai lệch $\le 1.0$) | **$-2LL = 1937.33$ ($\Delta = 0.00$), $\Delta\text{OFV} = -77.93$** | **PASS** |
| **F6** | Covariate Simulation-Estimation (15 reps) | Bias $\beta < 10\%$, Bias $\theta < 10\%$, SE ratio $[0.75, 1.33]$ | **Bias $\beta = 5.56\%$, Bias $\theta \le 2.54\%$, Ratio $0.98 - 1.06$, $0\%$ non-conv** | **PASS** |

### Bảng 2: Các Cổng Thẩm Định Nền Tảng (Gates G1 – G8)

| Tiêu chuẩn (Gate) | Ngưỡng chấp thuận | Kết quả thực đo | Trạng thái |
|---|---|---|:---:|
| **G1: Solver vs solve_ivp (250 kịch bản)** | Max relative error $< 10^{-6}$ | $2.01 \times 10^{-13}$ | **PASS** |
| **G2: Closed-form check (single & steady-state)** | Max diff $< 10^{-9}$ | $8.88 \times 10^{-15}$ / $7.11 \times 10^{-15}$ | **PASS** |
| **G3: Existing test_pk_core.py** | All tests pass | 2/2 tests PASS | **PASS** |
| **G4: Data loader verification** | 565 rows / 83 IDs / 287 doses / 278 obs | 565 / 83 / 287 / 278 | **PASS** |
| **G5: Simulation-estimation (30 reps, Base Model)** | Bias $\theta < 10\%$, $\omega < 25\%$, SE ratio $[0.75, 1.33]$ | $\text{Bias}_{\theta} \le 2.74\%$, $\text{Bias}_{\omega} \le 10.25\%$, Ratio $1.02 - 1.15$, $0\%$ non-conv | **PASS** |
| **G6: Multi-start (4 runs 0.5x - 2.0x)** | $\theta$ dev $< 5\%$, $\Delta(-2LL) \le 2.0$ | $\text{Dev}_{\theta} \le 0.97\%$, $\Delta(-2LL) = 1.35$ | **PASS** |
| **G7: Independent reference (R scripts)** | Export Monolix & write R scripts | `export_monolix.csv`, `run_saemix.R`, `run_nlmixr2.R` | **Scripts written, NOT executed** (Rscript có sẵn tại `/usr/local/bin/Rscript`, nhưng máy chủ chưa cài package `saemix` và `nlmixr2`) |
| **G8: Sensitivity analysis ($t_{inf}$)** | Fit $t_{inf} \in \{0.5, 1.0, 2.0\}$ h | Đo đạc $\theta, \omega, -2LL$ trên 3 kịch bản thời gian truyền | **PASS** |

---

## 📈 Kết Quả Phân Tích Hiệp Biến Lâm Sàng (Sau Sửa Lỗi)

| Mô hình | $-2LL$ (IS) | $\Delta\text{OFV}$ | AIC | $\Delta\text{AIC}$ | BICc | $\Delta\text{BICc}$ | Ước lượng Hệ số Hiệp biến $\beta$ ($SE$, Wald $p$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **M1: Base Model** (không hiệp biến) | 2015.26 | 0.00 | 2027.26 | 0.00 | 2044.19 | 0.00 | — |
| **M2: $CL \sim \text{CLCr\_CG}$** (thay đổi theo thời gian) | **1937.33** | **-77.93** | **1951.33** | **-75.93** | **1970.68** | **-73.51** | $\beta_{\text{CLCr\_CG}} = 0.8443$ ($SE=0.0776$, $p < 10^{-15}$) |
| **M3: $V \sim \text{TBW}$** | 2011.33 | -3.93 | 2025.33 | -1.93 | 2044.68 | +0.49 | $\beta_{\text{TBW}} = 0.5837$ ($SE=0.2409$, $p = 0.0154$) |
| **M4: Sepsis trên $CL$ và $V$** | 2014.62 | -0.64 | 2030.62 | +3.36 | 2052.39 | +8.20 | $\beta_{CL} = -0.1466$ ($p=0.487$); $\beta_V = 0.0268$ ($p=0.847$) |
| **M5: $CL \sim \text{Age}$** (không kèm CLCr_CG) | 1994.47 | -20.78 | 2008.47 | -18.78 | 2027.82 | -16.36 | $\beta_{\text{Age}} = -1.0469$ ($SE=0.2105$, $p < 10^{-6}$) |
