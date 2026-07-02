"""
Benchmark script — run test questions against the system and measure quality.
"""
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


TEST_QUESTIONS = [
    {
        "question": "Điều kiện kết hôn theo pháp luật Việt Nam?",
        "expected_keywords": ["đủ 20 tuổi", "đủ 18 tuổi", "tự nguyện", "Điều 8"],
    },
    {
        "question": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động?",
        "expected_keywords": ["45 ngày", "30 ngày", "03 ngày", "Điều 35"],
    },
    {
        "question": "Nguyên tắc bồi thường thiệt hại theo Bộ luật Dân sự?",
        "expected_keywords": ["bồi thường toàn bộ", "kịp thời", "Điều 585"],
    },
    {
        "question": "Tài sản chung của vợ chồng được xác định như thế nào?",
        "expected_keywords": ["thu nhập", "lao động", "tài sản chung", "Điều 33"],
    },
]


def main():
    from src.agents.orchestrator import answer_question

    print("=" * 60)
    print("🧪 LEGAL RAG — Benchmark")
    print("=" * 60)

    total = len(TEST_QUESTIONS)
    passed = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["question"]
        expected = test["expected_keywords"]

        print(f"\n--- Câu hỏi {i}/{total} ---")
        print(f"❓ {q}")

        start = time.time()
        try:
            answer = answer_question(q)
            elapsed = time.time() - start

            # Check keyword coverage
            found = [kw for kw in expected if kw.lower() in answer.lower()]
            coverage = len(found) / len(expected) if expected else 1.0

            print(f"⏱️  Thời gian: {elapsed:.1f}s")
            print(f"📊 Keyword coverage: {coverage:.0%} ({len(found)}/{len(expected)})")
            print(f"✅ Keywords found: {found}")
            print(f"📝 Trả lời (200 ký tự đầu): {answer[:200]}...")

            if coverage >= 0.5:
                passed += 1

        except Exception as e:
            print(f"❌ Lỗi: {e}")

    print(f"\n{'=' * 60}")
    print(f"📊 Kết quả: {passed}/{total} câu đạt yêu cầu ({passed/total:.0%})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
