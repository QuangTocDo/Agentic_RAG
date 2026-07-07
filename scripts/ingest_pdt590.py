"""
CLI Script — Ingest pdt590/vietnamese-legal-documents dataset from Hugging Face into the RAG system.

Usage:
    python scripts/ingest_pdt590.py --limit 2000
"""
import argparse
import sys
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Ingest pdt590/vietnamese-legal-documents dataset.")
    parser.add_argument("--limit", type=int, default=2000, help="Number of documents to ingest (default: 2000).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa database cũ trước khi nạp dữ liệu mới."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  LEGAL RAG — PDT590 Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load HF token
    from configs.setting import settings
    token = settings.hf_token
    if token == "your_huggingface_token_here" or not token:
        token = None
        print("⚠️  Cảnh báo: Không phát hiện HF_TOKEN. Tốc độ tải có thể bị giới hạn.")
    else:
        print("🔑 Đã phát hiện HF_TOKEN cho kết nối an toàn.")

    # Step 2: Stream dataset configs
    print(f"\n📂 Bước 1: Kết nối và stream bộ dữ liệu pdt590/vietnamese-legal-documents (Giới hạn: {args.limit} tài liệu)...")
    try:
        from datasets import load_dataset
        content_ds = load_dataset("pdt590/vietnamese-legal-documents", "content", split="data", streaming=True, token=token)
        meta_ds = load_dataset("pdt590/vietnamese-legal-documents", "metadata", split="data", streaming=True, token=token)
    except ImportError:
        print("❌ Lỗi: Vui lòng cài đặt thư viện 'datasets': pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi khi tải bộ dữ liệu: {e}")
        sys.exit(1)

    # Step 3: Parse and chunk documents
    print("\n🧹 Bước 2: Tải dữ liệu song song và thực hiện phân tách đoạn (chunking)...")
    from src.ingestion.cleaner import clean_text
    from src.ingestion.chunker import chunk_document

    raw_docs_count = 0
    all_chunks = []

    for c_item, m_item in zip(content_ds, meta_ds):
        doc_id = c_item["id"]
        
        # Ensure IDs match
        if c_item["id"] != m_item["id"]:
            print(f"   ⚠️  Bỏ qua dòng bị lệch ID: Content ID {c_item['id']} != Metadata ID {m_item['id']}")
            continue

        raw_text = c_item["content"] or ""
        cleaned_text = clean_text(raw_text)

        # Skip empty documents
        if not cleaned_text.strip():
            continue

        # Construct parent metadata
        law_name = m_item.get("legal_type", "Văn bản")
        doc_no = m_item.get("document_number", "")
        if doc_no:
            law_name = f"{law_name} {doc_no}"

        meta = {
            "source": "pdt590/vietnamese-legal-documents",
            "filename": f"doc_{doc_id}",
            "law_name": law_name,
            "title": m_item.get("title", ""),
            "url": m_item.get("url", ""),
            "issuing_authority": m_item.get("issuing_authority", ""),
            "issuance_date": m_item.get("issuance_date", "")
        }

        doc_item = {
            "page_content": cleaned_text,
            "metadata": meta
        }

        # Chunk the document into smaller sub-chunks
        doc_chunks = chunk_document(doc_item)
        all_chunks.extend(doc_chunks)

        raw_docs_count += 1
        if raw_docs_count % 200 == 0:
            print(f"   Đã xử lý {raw_docs_count} tài liệu thô, tạo ra {len(all_chunks)} phân đoạn...")

        if raw_docs_count >= args.limit:
            break

    print(f"   ✅ Hoàn tất tiền xử lý. Tổng cộng: {raw_docs_count} tài liệu thô -> {len(all_chunks)} phân đoạn RAG.")

    if not all_chunks:
        print("❌ Lỗi: Không có phân đoạn nào được tạo ra.")
        sys.exit(1)

    # Step 4: Store in ChromaDB
    print("\n💾 Bước 3: Lưu vào ChromaDB (vector store)...")
    from src.indexing.chroma_store import add_documents, reset_collection

    if args.reset:
        print("   🧹 Đang xóa database cũ theo yêu cầu --reset...")
        reset_collection()
    # Batch insertion to avoid massive payloads
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        ids = [f"pdt_{idx}" for idx in range(i, i + len(batch))]
        add_documents(batch, ids=ids)
        print(f"   Đã nạp {i + len(batch)} / {len(all_chunks)} phân đoạn vào vector store...")

    # Step 5: Build BM25 index
    print("\n🔤 Bước 4: Xây dựng BM25 index (keyword search)...")
    from src.indexing.bm25_index import BM25Index

    bm25 = BM25Index()
    if args.reset:
        bm25.build(all_chunks)
    else:
        bm25.append_and_build(all_chunks)
    bm25.save()

    # Step 6: Build knowledge graph
    print("\n🕸️  Bước 5: Xây dựng đồ thị pháp lý (knowledge graph)...")
    try:
        from src.retrieval.graph import build_graph, append_graph
        if args.reset:
            build_graph(all_chunks)
        else:
            append_graph(all_chunks)
        print("   ✅ Đã tạo đồ thị liên kết pháp lý.")
    except Exception as e:
        print(f"   ⚠️  Bỏ qua việc tạo đồ thị ({e})")

    print("\n" + "=" * 60)
    print("🎉 Hoàn tất! Hệ thống đã nạp thành công bộ dữ liệu pdt590/vietnamese-legal-documents.")
    print("=" * 60)
    print(f"\n💡 Hãy chạy lại ứng dụng bằng lệnh: python ui/app.py")


if __name__ == "__main__":
    main()
