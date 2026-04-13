# Scorecard: variant_hybrid_rerank
Generated: 2026-04-13 18:11

## Summary

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 3.40/5 |
| Relevance | 4.50/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.00/5 |

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| gq01 | SLA | 5 | 5 | 5 | 5 | Every claim in the answer is explicitly supported by the con |
| gq02 | Cross-Document | 2 | 5 | 5 | 5 | Several claims, such as the maximum number of devices allowe |
| gq03 | Refund | 2 | 5 | 5 | 4 | Several claims in the answer, such as the activation of prod |
| gq04 | Refund | 5 | 5 | 5 | 5 | Every claim in the answer is explicitly supported by the con |
| gq05 | Access Control | 1 | 5 | 5 | 1 | Answer is largely hallucinated and not grounded in the conte |
| gq06 | Cross-Document | 4 | 5 | 5 | 5 | Answer is almost fully grounded, but the detail about loggin |
| gq07 | Insufficient Context | 5 | 1 | None | 2 | Every claim in the answer is explicitly supported by the con |
| gq08 | HR Policy | 3 | 5 | 5 | 4 | The answer is mostly grounded in the context, but it introdu |
| gq09 | IT Helpdesk | 2 | 5 | 5 | 5 | Several claims about password change frequency and reminders |
| gq10 | Refund | 5 | 4 | 5 | 4 | Every claim in the answer is explicitly supported by the con |

## Answers

### [gq01] SLA xử lý ticket P1 đã thay đổi như thế nào so với phiên bản trước?
**Answer:** **Quy định**: SLA xử lý ticket P1

**Điều kiện áp dụng**: Áp dụng cho các ticket được phân loại là P1.

**Quy trình / Bước thực hiện**:
1. **Phản hồi ban đầu**: 15 phút kể từ khi ticket được tạo.
2. **Xử lý và khắc phục**: 4 giờ (đã giảm từ 6 giờ trong phiên bản trước).
3. **Escalation**: Tự động escalate lên Senior Engineer nếu không có phản hồi trong 10 phút (quy trình này được thêm vào trong phiên bản trước).
4. **Thông báo stakeholder**: Ngay khi nhận ticket, update mỗi 30 phút cho đến khi resolve.

**Lịch sử thay đổi**:
- **v2026.1 (2026-01-15)**: Cập nhật SLA P1 resolution từ 6 giờ xuống 4 giờ.
- **v2025.3 (2025-09-01)**: Thêm quy trình escalation tự động.
- **v2025.1 (2025-03-01)**: Phiên bản đầu tiên.

Thông tin này được trích từ tài liệu SLA P1 [support/sla-p1-2026.pdf | Phần 5].

**Expected:** Phiên bản hiện tại (v2026.1, có hiệu lực từ 2026-01-15) đã cập nhật SLA P1 resolution từ 6 giờ xuống còn 4 giờ. Phiên bản trước (v2025.3) không thay đổi SLA này.

---

### [gq02] Khi làm việc remote, tôi phải dùng VPN và được kết nối trên tối đa bao nhiêu thiết bị?
**Answer:** **Quy định**: Kết nối VPN khi làm việc remote

**Điều kiện áp dụng**:
- Nhân viên phải sử dụng VPN khi làm việc với hệ thống nội bộ.

**Quy trình / Bước thực hiện**:
1. Sử dụng phần mềm **Cisco AnyConnect** để kết nối VPN.
2. Mỗi tài khoản được phép kết nối trên tối đa **2 thiết bị** cùng lúc.

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

Thông tin này được nêu rõ trong tài liệu chính sách hoàn tiền **[policy/refund-v4.pdf | Điều 3]**.

**Expected:** Không được hoàn tiền. Đơn hàng rơi vào hai ngoại lệ cùng lúc: (1) đơn hàng áp dụng mã giảm giá chương trình Flash Sale và (2) sản phẩm đã được kích hoạt — cả hai đều thuộc danh sách ngoại lệ không được hoàn tiền theo Điều 3 của chính sách.

---

### [gq04] Nếu chọn nhận store credit thay vì hoàn tiền, tôi được bao nhiêu phần trăm so với số tiền gốc?
**Answer:** **Quy định**: Hoàn tiền qua store credit

**Điều kiện áp dụng**: 
- Khách hàng có thể chọn nhận store credit thay thế với giá trị **110%** so với số tiền hoàn.

**Quy trình / Bước thực hiện**:
1. Khách hàng gửi yêu cầu qua hệ thống ticket nội bộ với category "Refund Request".
2. CS Agent xem xét trong vòng 1 ngày làm việc và xác nhận điều kiện đủ điều kiện.
3. Nếu đủ điều kiện, chuyển yêu cầu sang Finance Team để xử lý hoàn tiền.
4. Finance Team xử lý trong 3-5 ngày làm việc và thông báo kết quả cho khách hàng.

Thông tin này được trích từ tài liệu **[policy/refund-v4.pdf | Điều 5]**.

**Expected:** Khách hàng có thể chọn nhận store credit (credit nội bộ) thay thế với giá trị 110% so với số tiền hoàn gốc. Đây là tùy chọn, không phải bắt buộc.

---

### [gq05] Contractor từ bên ngoài công ty có thể được cấp quyền Admin Access không? Nếu có, cần bao nhiêu ngày và có yêu cầu đặc biệt gì?
**Answer:** **Quy định**: Contractor từ bên ngoài công ty không được cấp quyền Admin Access.

**Điều kiện áp dụng**: Quyền Admin Access chỉ được cấp cho nhân viên nội bộ và không áp dụng cho các bên thứ ba hoặc contractor.

**Quy trình / Bước thực hiện**: Không có quy trình nào cho việc cấp quyền Admin Access cho contractor, vì điều này không được phép theo chính sách hiện hành.

Tôi không tìm thấy thông tin này trong tài liệu nội bộ hiện có.

**Expected:** Có. Theo Access Control SOP, quy trình áp dụng cho tất cả nhân viên, contractor và third-party vendor. Admin Access (Level 4) cần phê duyệt từ IT Manager và CISO, thời gian xử lý 5 ngày làm việc, và có yêu cầu thêm: training bắt buộc về security policy.

---

### [gq06] Lúc 2 giờ sáng xảy ra sự cố P1, on-call engineer cần cấp quyền tạm thời cho một engineer xử lý incident. Quy trình cụ thể như thế nào và quyền này tồn tại bao lâu?
**Answer:** **Quy định**: Cấp quyền tạm thời cho engineer xử lý incident P1.

**Điều kiện áp dụng**: 
- Sự cố P1 xảy ra và cần cấp quyền tạm thời để khắc phục.

**Quy trình / Bước thực hiện**:
1. On-call IT Admin có thể cấp quyền tạm thời tối đa 24 giờ sau khi được Tech Lead phê duyệt bằng lời.
2. Sau 24 giờ, quyền này sẽ bị thu hồi tự động hoặc cần có ticket chính thức để duy trì quyền.
3. Mọi quyền tạm thời phải được ghi log vào hệ thống Security Audit.

Quyền tạm thời này tồn tại tối đa **24 giờ** trước khi cần có hành động tiếp theo [it/access-control-sop.md | Section 4].

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
- **Nghỉ ốm**: Nhân viên phải thông báo cho Line Manager trước **9:00 sáng** ngày nghỉ. Nếu nghỉ trên **3 ngày liên tiếp**, cần có giấy tờ y tế từ bệnh viện.

**Quy trình / Bước thực hiện**:
1. **Nghỉ phép năm**:
   - Gửi yêu cầu qua hệ thống HR Portal.
   - Line Manager phê duyệt trong vòng 1 ngày làm việc.
2. **Nghỉ ốm**:
   - Thông báo cho Line Manager trước 9:00 sáng.
   - Nếu nghỉ trên 3 ngày, cung cấp giấy tờ y tế.

Tóm lại, số ngày thông báo trước cho nghỉ phép năm là **3 ngày làm việc**, trong khi nghỉ ốm cần thông báo trước **9:00 sáng** ngày nghỉ. **[hr/leave-policy-2026.pdf | Phần 2]**.

**Expected:** Nghỉ phép năm cần báo trước ít nhất 3 ngày làm việc qua HR Portal. Nghỉ ốm cũng liên quan đến mốc 3 ngày: nếu nghỉ TRÊN 3 ngày liên tiếp thì cần giấy tờ y tế từ bệnh viện. Hai con số '3 ngày' này áp dụng cho hai tình huống khác nhau.

---

### [gq09] Mật khẩu tài khoản công ty cần đổi định kỳ không? Nếu có, hệ thống sẽ nhắc nhở trước bao nhiêu ngày và đổi qua đâu?
**Answer:** **Quy định**: Mật khẩu tài khoản công ty cần thay đổi định kỳ.

**Điều kiện áp dụng**: Mật khẩu phải được thay đổi mỗi 90 ngày.

**Quy trình / Bước thực hiện**:
1. Hệ thống sẽ nhắc nhở người dùng 7 ngày trước khi mật khẩu hết hạn.
2. Để thay đổi mật khẩu, người dùng có thể truy cập vào portal SSO tại https://sso.company.internal/reset.

**[source | Section 1]**

**Expected:** Có. Mật khẩu phải được thay đổi mỗi 90 ngày. Hệ thống sẽ nhắc nhở 7 ngày trước khi hết hạn. Để đổi mật khẩu, truy cập https://sso.company.internal/reset hoặc liên hệ Helpdesk qua ext. 9000.

---

### [gq10] Chính sách hoàn tiền hiện tại áp dụng cho đơn hàng đặt trước ngày 01/02/2026 không?
**Answer:** **Quy định**: Chính sách hoàn tiền hiện tại áp dụng cho đơn hàng đặt từ ngày 01/02/2026.

**Điều kiện áp dụng**: 
- Các đơn hàng đặt trước ngày 01/02/2026 sẽ áp dụng theo chính sách hoàn tiền phiên bản 3.

**Quy trình / Bước thực hiện**: 
1. Đối với đơn hàng đặt sau ngày 01/02/2026, khách hàng có quyền yêu cầu hoàn tiền nếu đáp ứng đủ các điều kiện:
   - Yêu cầu được gửi trong vòng 7 ngày kể từ thời điểm xác nhận đơn hàng.
   - Sản phẩm bị lỗi do nhà sản xuất, không phải do người dùng.
   - Đơn hàng chưa được sử dụng hoặc chưa bị mở seal.

**[source | Điều 1]**

**Expected:** Không. Chính sách hoàn tiền phiên bản 4 có hiệu lực từ ngày 01/02/2026 và chỉ áp dụng cho các đơn hàng được đặt kể từ ngày đó. Các đơn hàng đặt trước ngày này sẽ áp dụng theo chính sách hoàn tiền phiên bản 3.

---

