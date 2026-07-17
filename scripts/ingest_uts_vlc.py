"""
CLI Script — Ingest undertheseanlp/UTS_VLC dataset from Hugging Face into the RAG system.

Usage:
    python scripts/ingest_uts_vlc.py --split 2026 --limit 100 --reset
"""
import argparse
import sys
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Ingest undertheseanlp/UTS_VLC dataset from Hugging Face.")
    parser.add_argument(
        "--split",
        type=str,
        default="2026",
        help="Dataset split to ingest (default: 2026). Options: 2026, 2026_01, 2023, 2021"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of documents to ingest. Set to 0 or negative for no limit (default: 0)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa database cũ trước khi nạp dữ liệu mới."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  LEGAL RAG — UTS_VLC Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load HF token
    from configs.setting import settings
    token = settings.hf_token
    if token == "your_huggingface_token_here" or not token:
        token = None
        print("⚠️  Cảnh báo: Không phát hiện HF_TOKEN. Tốc độ tải có thể bị giới hạn.")
    else:
        print("🔑 Đã phát hiện HF_TOKEN cho kết nối an toàn.")

    # Step 2: Stream dataset split
    limit_str = f"Giới hạn: {args.limit}" if args.limit > 0 else "Không giới hạn"
    print(f"\n📂 Bước 1: Kết nối và stream bộ dữ liệu undertheseanlp/UTS_VLC (Split: {args.split}, {limit_str} tài liệu)...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("undertheseanlp/UTS_VLC", split=args.split, streaming=True, token=token)
    except ImportError:
        print("❌ Lỗi: Vui lòng cài đặt thư viện 'datasets': pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Lỗi khi tải bộ dữ liệu: {e}")
        sys.exit(1)

    # Step 3: Parse, filter, chunk and ingest in batches
    print("\n🧹 Bước 2: Tải dữ liệu, làm sạch, chunk và nạp vào ChromaDB theo lô (batch)...")
    from src.ingestion.cleaner import clean_text
    from src.ingestion.chunker import chunk_document
    from src.indexing.chroma_store import add_documents, reset_collection

    if args.reset:
        print("   🧹 Đang xóa database cũ theo yêu cầu --reset...")
        reset_collection()

    raw_docs_count = 0
    chunk_buffer = []
    all_chunks_for_indices = []
    
    batch_size = 1000  # Size of batch to insert into ChromaDB
    total_chunks_count = 0

    for item in dataset:
        doc_id = item["id"]
        raw_text = item["content"] or ""
        cleaned_text = clean_text(raw_text)

        # Skip empty documents
        if not cleaned_text.strip():
            continue

        # Construct parent metadata
        law_type = item.get("type", "")
        title = item.get("title", "")
        
        # Combine type and title to form law_name
        if law_type and title:
            law_name = f"{law_type} {title}"
        else:
            law_name = title or law_type or "Văn bản"

        meta = {
            "source": f"undertheseanlp/UTS_VLC:{args.split}",
            "filename": item.get("filename") or f"doc_{doc_id}",
            "law_name": law_name,
            "title": title,
            "url": "",
            "issuing_authority": "Quốc hội",
            "issuance_date": ""
        }

        doc_item = {
            "page_content": cleaned_text,
            "metadata": meta
        }

        # Chunk the document into smaller sub-chunks
        doc_chunks = chunk_document(doc_item)
        
        chunk_buffer.extend(doc_chunks)
        all_chunks_for_indices.extend(doc_chunks)

        raw_docs_count += 1
        
        # If chunk buffer exceeds batch_size, upsert to ChromaDB
        if len(chunk_buffer) >= batch_size:
            start_idx = total_chunks_count
            end_idx = start_idx + len(chunk_buffer)
            ids = [f"uts_{idx}" for idx in range(start_idx, end_idx)]
            add_documents(chunk_buffer, ids=ids)
            total_chunks_count = end_idx
            print(f"   💾 [ChromaDB] Đã nạp xong lô {len(chunk_buffer)} phân đoạn (Lũy kế: {total_chunks_count} chunks)...")
            chunk_buffer.clear()

        if raw_docs_count % 100 == 0:
            print(f"   Đã quét và xử lý {raw_docs_count} tài liệu...")

        if args.limit > 0 and raw_docs_count >= args.limit:
            break

    # Insert remaining chunks in the buffer
    if chunk_buffer:
        start_idx = total_chunks_count
        end_idx = start_idx + len(chunk_buffer)
        ids = [f"uts_{idx}" for idx in range(start_idx, end_idx)]
        add_documents(chunk_buffer, ids=ids)
        total_chunks_count = end_idx
        print(f"   💾 [ChromaDB] Đã nạp xong lô cuối {len(chunk_buffer)} phân đoạn (Lũy kế: {total_chunks_count} chunks)...")
        chunk_buffer.clear()

    print(f"   ✅ Hoàn tất nạp ChromaDB. Thống kê: xử lý {raw_docs_count} tài liệu -> {total_chunks_count} phân đoạn.")

    if not all_chunks_for_indices:
        print("❌ Lỗi: Không có phân đoạn nào được tạo ra.")
        sys.exit(1)

    # Step 4: Build BM25 index
    print("\n🔤 Bước 3: Xây dựng BM25 index (keyword search)...")
    from src.indexing.bm25_index import BM25Index

    bm25 = BM25Index()
    if args.reset:
        bm25.build(all_chunks_for_indices)
    else:
        bm25.append_and_build(all_chunks_for_indices)
    bm25.save()

    # Step 5: Build knowledge graph
    print("\n🕸️  Bước 4: Xây dựng đồ thị pháp lý (knowledge graph)...")
    try:
        from src.retrieval.graph import build_graph, append_graph
        if args.reset:
            build_graph(all_chunks_for_indices)
        else:
            append_graph(all_chunks_for_indices)
        print("   ✅ Đã cập nhật đồ thị liên kết pháp lý.")
    except Exception as e:
        print(f"   ⚠️  Bỏ qua việc tạo đồ thị ({e})")

    print("\n" + "=" * 60)
    print("🎉 Hoàn tất! Hệ thống đã nạp thành công bộ dữ liệu undertheseanlp/UTS_VLC.")
    print("=" * 60)
    print(f"\n💡 Hãy chạy lại ứng dụng bằng lệnh: python ui/app.py")


if __name__ == "__main__":
    main()
