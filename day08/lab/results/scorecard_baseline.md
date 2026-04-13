# Scorecard: baseline_dense
Generated: 2026-04-13 18:08

## Summary

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 3.10/5 |
| Relevance | 4.60/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.10/5 |

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| gq01 | SLA | 5 | 5 | 5 | 5 | Every claim in the answer is explicitly supported by the con |
| gq02 | Cross-Document | 3 | 5 | 5 | 5 | The answer is mostly grounded in the context, but it introdu |
| gq03 | Refund | 2 | 5 | 5 | 4 | Several claims in the answer, such as the activation of the  |
| gq04 | Refund | 5 | 5 | 5 | 5 | Every claim in the answer is explicitly supported by the con |
| gq05 | Access Control | 2 | 5 | 5 | 1 | Several claims about Admin Access for contractors and the ap |
| gq06 | Cross-Document | 3 | 5 | 5 | 5 | The answer includes claims about logging temporary permissio |
| gq07 | Insufficient Context | 1 | 1 | None | 2 | The answer does not reference any information from the retri |
| gq08 | HR Policy | 3 | 5 | 5 | 4 | The answer is mostly grounded in the context but introduces  |
| gq09 | IT Helpdesk | 2 | 5 | 5 | 5 | Several claims about password change frequency and reminders |
| gq10 | Refund | 5 | 5 | 5 | 5 | Every claim in the answer is explicitly supported by the con |

## Answers

### [gq01] SLA xử lý ticket P1 đã thay đổi như thế nào so với phiên bản trước?
**Answer:** **Quy định**: SLA xử lý ticket P1

**Điều kiện áp dụng**: Áp dụng cho các ticket được phân loại là P1 (ưu tiên cao nhất).

**Quy trình / Bước thực hiện**:
1. **Phản hồi ban đầu**: 15 phút kể từ khi ticket được tạo.
2. **Xử lý và khắc phục**: Thời gian xử lý đã được cập nhật từ 6 giờ xuống còn 4 giờ.
3. **Escalation**: Tự động escalate lên Senior Engineer nếu không có phản hồi trong 10 phút.
4. **Thông báo stakeholder**: Ngay khi nhận ticket, cập nhật mỗi 30 phút cho đến khi ticket được giải quyết.

**Lịch sử thay đổi**:
- **Phiên bản 2026.1 (2026-01-15)**: Cập nhật SLA P1 resolution từ 6 giờ xuống 4 giờ.
- **Phiên bản 2025.3 (2025-09-01)**: Thêm quy trình escalation tự động.
- **Phiên bản 2025.1 (2025-03-01)**: Phiên bản đầu tiên.

**[source | Phần 5]**

**Expected:** Phiên bản hiện tại (v2026.1, có hiệu lực từ 2026-01-15) đã cập nhật SLA P1 resolution từ 6 giờ xuống còn 4 giờ. Phiên bản trước (v2025.3) không thay đổi SLA này.

---

### [gq02] Khi làm việc remote, tôi phải dùng VPN và được kết nối trên tối đa bao nhiêu thiết bị?
**Answer:** **Quy định**: Khi làm việc remote, nhân viên phải sử dụng VPN.

**Điều kiện áp dụng**:
- Kết nối VPN là bắt buộc khi làm việc với hệ thống nội bộ.
- Mỗi tài khoản được kết nối VPN trên tối đa **2 thiết bị cùng lúc**.

**Quy trình / Bước thực hiện**:
1. Tải phần mềm VPN Cisco AnyConnect từ [đường dẫn tải](https://vpn.company.internal/download).
2. Kết nối VPN trên thiết bị cần thiết.

Thông tin này được trích từ tài liệu **[support/helpdesk-faq.md | Section 2]**.

**Expected:** Khi làm việc remote và truy cập hệ thống nội bộ, VPN là bắt buộc (theo HR Leave Policy). Phần mềm VPN là Cisco AnyConnect và mỗi tài khoản được kết nối VPN trên tối đa 2 thiết bị cùng lúc (theo IT Helpdesk FAQ).

---

### [gq03] Đơn hàng mua trong chương trình Flash Sale và đã kích hoạt sản phẩm có được hoàn tiền không?
**Answer:** **Quy định**: Đơn hàng mua trong chương trình Flash Sale và đã kích hoạt sản phẩm không được hoàn tiền.

**Điều kiện áp dụng**:
- Đơn hàng đã áp dụng mã giảm giá đặc biệt theo chương trình khuyến mãi Flash Sale.
- Sản phẩm đã được kích hoạt hoặc đăng ký tài khoản.

**Quy trình / Bước thực hiện**:
- Không có quy trình hoàn tiền cho các đơn hàng thuộc trường hợp trên.

Thông tin này được trích từ tài liệu hoàn tiền **[policy/refund-v4.pdf | Điều 3: Điều kiện áp dụng và ngoại lệ]**.

**Expected:** Không được hoàn tiền. Đơn hàng rơi vào hai ngoại lệ cùng lúc: (1) đơn hàng áp dụng mã giảm giá chương trình Flash Sale và (2) sản phẩm đã được kích hoạt — cả hai đều thuộc danh sách ngoại lệ không được hoàn tiền theo Điều 3 của chính sách.

---

### [gq04] Nếu chọn nhận store credit thay vì hoàn tiền, tôi được bao nhiêu phần trăm so với số tiền gốc?
**Answer:** **Quy định**: Hoàn tiền qua store credit

**Điều kiện áp dụng**: Khách hàng có thể chọn nhận store credit thay thế cho hoàn tiền với giá trị 110% so với số tiền hoàn.

**Quy trình / Bước thực hiện**:
1. Khách hàng gửi yêu cầu qua hệ thống ticket nội bộ với category "Refund Request".
2. CS Agent xem xét trong vòng 1 ngày làm việc và xác nhận điều kiện đủ điều kiện.
3. Nếu đủ điều kiện, chuyển yêu cầu sang Finance Team để xử lý hoàn tiền.

Thông tin này được trích từ tài liệu **[policy/refund-v4.pdf | Điều 5]**.

**Expected:** Khách hàng có thể chọn nhận store credit (credit nội bộ) thay thế với giá trị 110% so với số tiền hoàn gốc. Đây là tùy chọn, không phải bắt buộc.

---

### [gq05] Contractor từ bên ngoài công ty có thể được cấp quyền Admin Access không? Nếu có, cần bao nhiêu ngày và có yêu cầu đặc biệt gì?
**Answer:** **Quy định**: Contractor từ bên ngoài công ty không được cấp quyền Admin Access (Level 4).

**Điều kiện áp dụng**: 
- Level 4 — Admin Access chỉ áp dụng cho các vị trí như DevOps, SRE, IT Admin, và yêu cầu phê duyệt từ IT Manager và CISO.

**Quy trình / Bước thực hiện**: 
- Không có quy trình nào cho việc cấp quyền Admin Access cho contractor, vì họ không nằm trong danh sách các đối tượng đủ điều kiện.

Nếu bạn cần thêm thông tin hoặc có câu hỏi khác, vui lòng cho tôi biết!

**Expected:** Có. Theo Access Control SOP, quy trình áp dụng cho tất cả nhân viên, contractor và third-party vendor. Admin Access (Level 4) cần phê duyệt từ IT Manager và CISO, thời gian xử lý 5 ngày làm việc, và có yêu cầu thêm: training bắt buộc về security policy.

---

### [gq06] Lúc 2 giờ sáng xảy ra sự cố P1, on-call engineer cần cấp quyền tạm thời cho một engineer xử lý incident. Quy trình cụ thể như thế nào và quyền này tồn tại bao lâu?
**Answer:** **Quy định**: Cấp quyền tạm thời trong sự cố P1.

**Điều kiện áp dụng**: Khi xảy ra sự cố P1, on-call engineer cần cấp quyền tạm thời cho một engineer để xử lý incident.

**Quy trình / Bước thực hiện**:
1. On-call IT Admin có thể cấp quyền tạm thời (tối đa 24 giờ) sau khi được Tech Lead phê duyệt bằng lời.
2. Sau 24 giờ, quyền tạm thời sẽ bị thu hồi tự động hoặc cần có ticket chính thức để duy trì quyền.
3. Mọi quyền tạm thời phải được ghi log vào hệ thống Security Audit.

Quyền tạm thời này tồn tại tối đa 24 giờ. **[it/access-control-sop.md | Section 4]**.

**Expected:** On-call IT Admin có thể cấp quyền tạm thời (tối đa 24 giờ) sau khi được Tech Lead phê duyệt bằng lời. Sau 24 giờ, phải có ticket chính thức hoặc quyền bị thu hồi tự động. Mọi quyền tạm thời phải được ghi log vào hệ thống Security Audit. Liên hệ on-call qua hotline ext. 9999 (24/7) theo SLA P1.

---

### [gq07] Công ty sẽ phạt bao nhiêu nếu team IT vi phạm cam kết SLA P1?
**Answer:** Tôi không tìm thấy thông tin này trong tài liệu nội bộ hiện có.

**Expected:** Tài liệu hiện có không quy định mức phạt hay hình thức xử lý khi vi phạm SLA P1. Tài liệu sla-p1-2026.pdf chỉ mô tả quy trình xử lý và SLA target, không có điều khoản về penalty.

---

### [gq08] Nhân viên phải báo trước bao nhiêu ngày để xin nghỉ phép năm? Con số này có giống với số ngày cần giấy tờ khi nghỉ ốm không?
**Answer:** **Quy định**: Nhân viên phải thông báo trước khi xin nghỉ phép năm và nghỉ ốm.

**Điều kiện áp dụng**:
- **Nghỉ phép năm**: Nhân viên cần gửi yêu cầu ít nhất **3 ngày làm việc** trước ngày nghỉ.
- **Nghỉ ốm**: Nhân viên phải thông báo cho Line Manager trước **9:00 sáng** ngày nghỉ. Nếu nghỉ trên **3 ngày liên tiếp**, cần có giấy tờ y tế.

**Quy trình / Bước thực hiện**:
1. **Nghỉ phép năm**:
   - Gửi yêu cầu qua hệ thống HR Portal.
   - Line Manager phê duyệt trong vòng 1 ngày làm việc.
2. **Nghỉ ốm**:
   - Thông báo cho Line Manager trước 9:00 sáng.
   - Cung cấp giấy tờ y tế nếu nghỉ trên 3 ngày liên tiếp.

Tóm lại, số ngày thông báo trước cho nghỉ phép năm là **3 ngày làm việc**, trong khi nghỉ ốm cần thông báo trước **9:00 sáng** ngày nghỉ. **[hr/leave-policy-2026.pdf | Phần 2]**.

**Expected:** Nghỉ phép năm cần báo trước ít nhất 3 ngày làm việc qua HR Portal. Nghỉ ốm cũng liên quan đến mốc 3 ngày: nếu nghỉ TRÊN 3 ngày liên tiếp thì cần giấy tờ y tế từ bệnh viện. Hai con số '3 ngày' này áp dụng cho hai tình huống khác nhau.

---

### [gq09] Mật khẩu tài khoản công ty cần đổi định kỳ không? Nếu có, hệ thống sẽ nhắc nhở trước bao nhiêu ngày và đổi qua đâu?
**Answer:** **Quy định**: Mật khẩu tài khoản công ty cần thay đổi định kỳ.

**Điều kiện áp dụng**: Mật khẩu phải được thay đổi mỗi 90 ngày.

**Quy trình / Bước thực hiện**:
1. Hệ thống sẽ nhắc nhở người dùng 7 ngày trước khi mật khẩu hết hạn.
2. Để thay đổi mật khẩu, người dùng có thể truy cập vào portal SSO tại https://sso.company.internal/reset hoặc liên hệ Helpdesk qua ext. 9000.

**[source | Section 1]**

**Expected:** Có. Mật khẩu phải được thay đổi mỗi 90 ngày. Hệ thống sẽ nhắc nhở 7 ngày trước khi hết hạn. Để đổi mật khẩu, truy cập https://sso.company.internal/reset hoặc liên hệ Helpdesk qua ext. 9000.

---

### [gq10] Chính sách hoàn tiền hiện tại áp dụng cho đơn hàng đặt trước ngày 01/02/2026 không?
**Answer:** **Quy định**: Chính sách hoàn tiền hiện tại áp dụng cho đơn hàng đặt trước ngày 01/02/2026 không.

**Điều kiện áp dụng**: 
- Chính sách hoàn tiền phiên bản 4 áp dụng cho tất cả các đơn hàng được đặt trên hệ thống nội bộ kể từ ngày 01/02/2026.
- Các đơn hàng đặt trước ngày 01/02/2026 sẽ áp dụng theo chính sách hoàn tiền phiên bản 3.

**Quy trình / Bước thực hiện**: 
- Đối với đơn hàng đặt trước ngày 01/02/2026, nhân viên cần tham khảo chính sách hoàn tiền phiên bản 3 để biết thêm chi tiết về quy trình và điều kiện hoàn tiền.

**[source | Điều 1]**

**Expected:** Không. Chính sách hoàn tiền phiên bản 4 có hiệu lực từ ngày 01/02/2026 và chỉ áp dụng cho các đơn hàng được đặt kể từ ngày đó. Các đơn hàng đặt trước ngày này sẽ áp dụng theo chính sách hoàn tiền phiên bản 3.

---

