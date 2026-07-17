# Tại Sao Nên Ưu Tiên Văn Bản Pháp Luật Mới Nhất Trong Hệ Thống RAG Pháp Lý

## Tổng Quan

Khi xây dựng chatbot tư vấn pháp luật hoặc hệ thống Retrieval-Augmented Generation (RAG) cho lĩnh vực pháp lý, việc đưa toàn bộ văn bản pháp luật từ nhiều thời kỳ vào cơ sở dữ liệu mà không kiểm soát hiệu lực văn bản có thể dẫn đến các câu trả lời không chính xác, lỗi thời hoặc mâu thuẫn với quy định hiện hành.

Vì vậy, hệ thống nên ưu tiên sử dụng các văn bản pháp luật đang còn hiệu lực hoặc phiên bản mới nhất của văn bản pháp luật thay vì lưu trữ và truy xuất đồng thời tất cả các phiên bản lịch sử.

---

# Vấn Đề Khi Sử Dụng Đồng Thời Văn Bản Cũ Và Mới

## 1. Nguy Cơ Truy Xuất Nhầm Văn Bản Hết Hiệu Lực

Các hệ thống vector search hoạt động dựa trên độ tương đồng ngữ nghĩa.

Ví dụ người dùng hỏi:

> Điều kiện kết hôn là gì?

Vector database có thể đồng thời trả về:

* Luật Hôn nhân và Gia đình năm 2000
* Luật Hôn nhân và Gia đình năm 2014

Mặc dù văn bản năm 2014 đang có hiệu lực, nhưng nếu văn bản năm 2000 được xếp hạng cao hơn thì mô hình có thể sử dụng thông tin cũ để trả lời.

Kết quả là chatbot cung cấp căn cứ pháp lý không còn giá trị áp dụng.

---

## 2. Một Điều Luật Có Thể Được Sửa Đổi Nhiều Lần

Trong thực tế, nhiều luật được sửa đổi hoặc bổ sung qua nhiều năm.

Ví dụ:

* Luật Đất đai 2003
* Luật Đất đai 2013
* Luật Đất đai 2024

Nếu tất cả các phiên bản đều tồn tại trong cơ sở dữ liệu mà không có cơ chế lọc hiệu lực, chatbot có thể:

* Trích dẫn sai điều luật
* Trả lời theo quy định đã bị thay thế
* Đưa ra căn cứ pháp lý không còn áp dụng

---

## 3. Gây Nhiễu Trong Quá Trình Retrieval

Khi có quá nhiều phiên bản tương tự nhau:

* Điều luật cũ
* Điều luật sửa đổi
* Điều luật thay thế

Vector search sẽ trả về nhiều chunk có nội dung gần giống nhau.

Điều này làm:

* Giảm độ chính xác retrieval
* Tăng chi phí reranking
* Tăng khả năng hallucination

---

## 4. Khó Kiểm Chứng Tính Đúng Đắn Của Câu Trả Lời

Người dùng kỳ vọng chatbot trả lời theo quy định hiện hành.

Nếu chatbot sử dụng văn bản đã hết hiệu lực:

* Câu trả lời có thể đúng về mặt lịch sử
* Nhưng sai về mặt pháp lý tại thời điểm hiện tại

Đây là rủi ro rất lớn đối với các hệ thống tư vấn pháp luật.

---

# Lợi Ích Của Việc Chỉ Giữ Văn Bản Mới Nhất

## Tăng Độ Chính Xác

Hệ thống chỉ truy xuất các quy định đang có hiệu lực.

Kết quả:

* Giảm sai sót
* Giảm mâu thuẫn
* Tăng độ tin cậy

---

## Giảm Kích Thước Vector Database

Ví dụ:

Nếu một bộ luật có:

* 5 phiên bản lịch sử

Nhưng chỉ giữ:

* Phiên bản hiện hành

Thì số lượng chunk có thể giảm đáng kể.

Lợi ích:

* Tiết kiệm bộ nhớ
* Tăng tốc độ tìm kiếm
* Giảm chi phí embedding

---

## Cải Thiện Chất Lượng Retrieval

Retriever không còn phải lựa chọn giữa:

* Văn bản cũ
* Văn bản mới

Do đó các kết quả tìm kiếm trở nên ổn định hơn.

---

## Giảm Hallucination

Khi ngữ cảnh được cung cấp rõ ràng và nhất quán:

* Mô hình ít suy diễn hơn
* Ít trích dẫn sai hơn
* Ít kết hợp nhầm nhiều phiên bản luật khác nhau

---

# Chiến Lược Đề Xuất

## Mức Độ 1: Chỉ Lưu Văn Bản Hiện Hành

Đối với chatbot tư vấn pháp luật thông thường:

* Chỉ ingest văn bản đang còn hiệu lực.
* Loại bỏ văn bản đã hết hiệu lực.
* Loại bỏ văn bản đã bị thay thế hoàn toàn.

Đây là phương án đơn giản và hiệu quả nhất.

---

## Mức Độ 2: Lưu Văn Bản Lịch Sử Nhưng Không Đưa Vào Retrieval

Có thể lưu riêng:

* Văn bản hiện hành
* Văn bản lịch sử

Ví dụ:

```text
/current_laws
/historical_laws
```

Retriever chỉ tìm kiếm trong:

```text
/current_laws
```

Trong khi văn bản lịch sử vẫn được giữ lại để phục vụ nghiên cứu hoặc tra cứu chuyên sâu.

---

## Mức Độ 3: Metadata Hiệu Lực Văn Bản

Mỗi document nên có metadata:

```json
{
  "document_id": "...",
  "title": "...",
  "issued_date": "...",
  "effective_date": "...",
  "expired_date": "...",
  "status": "active"
}
```

Trước khi retrieval:

```python
filter = {
    "status": "active"
}
```

Chỉ các văn bản còn hiệu lực mới được đưa vào kết quả tìm kiếm.

---

# Khuyến Nghị Cho Chatbot Pháp Luật

Đối với chatbot tư vấn pháp luật Việt Nam:

1. Chỉ sử dụng văn bản còn hiệu lực trong hệ thống RAG chính.
2. Loại bỏ các văn bản đã bị thay thế hoặc hết hiệu lực.
3. Chunk theo Điều, Khoản, Điểm thay vì chunk theo số ký tự.
4. Sử dụng metadata để lọc văn bản trước khi retrieval.
5. Bắt buộc trích dẫn nguồn pháp luật khi trả lời.
6. Nếu không tìm thấy căn cứ pháp luật phù hợp thì trả lời không đủ dữ liệu thay vì tự suy diễn.

---

# Kết Luận

- `[x]` Chạy benchmark kiểm tra chất lượng RAG: `python3 scripts/run_benchmark.py`.
- `[x]` Triển khai bộ công cụ mới (Tools) cho RAG Agent và Ingestion Agent.
- `[x]` Restructure `orchestrator.py` thành mô hình Agent 2 tầng với `DeepAgent` wrapper và Orchestrator định tuyến.
- `[/]` Chạy benchmark kiểm định hệ thống sau tái cấu trúc Agent.

Việc đưa toàn bộ lịch sử văn bản pháp luật vào hệ thống RAG không đồng nghĩa với việc chatbot sẽ thông minh hơn. Ngược lại, nó có thể làm tăng nguy cơ truy xuất nhầm văn bản hết hiệu lực và dẫn đến các câu trả lời sai về mặt pháp lý.

Đối với đa số bài toán tư vấn pháp luật, chiến lược hiệu quả nhất là chỉ sử dụng các văn bản đang còn hiệu lực hoặc phiên bản mới nhất của văn bản pháp luật. Cách tiếp cận này giúp nâng cao độ chính xác, giảm hallucination, tối ưu hiệu năng hệ thống và đảm bảo các câu trả lời phù hợp với quy định pháp luật hiện hành.
