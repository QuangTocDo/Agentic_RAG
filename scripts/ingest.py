"""
CLI Script — Ingest legal documents into the RAG system using streaming and batch processing.

Usage:
    python scripts/ingest.py --path data/legal_docs/
"""
from __future__ import annotations
import argparse
import sys
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu văn bản pháp luật vào hệ thống RAG quy mô lớn."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "hf"],
        default="local",
        help="Nguồn văn bản pháp luật: 'local' (thư mục cục bộ) hoặc 'hf' (Hugging Face dataset)."
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=None,
        help="Danh sách ID văn bản cần nạp khi chọn nguồn 'hf'."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Giới hạn số lượng tài liệu cần tải khi source='hf' và không truyền --ids. Dùng 0 hoặc số âm để tải toàn bộ."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "legal_docs"),
        help="Đường dẫn tới thư mục hoặc file chứa văn bản pháp luật (dùng khi source='local').",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Xóa database cũ trước khi nạp dữ liệu mới."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Số lượng tài liệu xử lý trong mỗi lô (batch) để tối ưu bộ nhớ."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "ingest_checkpoint.json"),
        help="Đường dẫn lưu file checkpoint để khôi phục khi bị gián đoạn."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  LEGAL RAG — Ingestion Pipeline (Optimized)")
    print("=" * 60)

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    processed_sources = set()
    
    if args.reset:
        print("🧹 Đang xóa database cũ theo yêu cầu --reset...")
        from src.indexing.chroma_store import reset_collection
        reset_collection()
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print("   🗑️ Đã xóa file checkpoint cũ.")
            except Exception as e:
                print(f"   ⚠️  Không thể xóa file checkpoint ({e})")
    else:
        # Load existing checkpoint
        if os.path.exists(checkpoint_path):
            try:
                import json
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                    processed_sources = set(checkpoint_data.get("processed_sources", []))
                print(f"   📂 Đã tải checkpoint. Đã nạp thành công {len(processed_sources)} tài liệu trước đó.")
            except Exception as e:
                print(f"   ⚠️  Lỗi đọc checkpoint ({e}), tiến hành chạy lại từ đầu.")

    # 1. Đọc stream tài liệu
    print("\n📂 Bước 1: Chuẩn bị nguồn dữ liệu...")
    if args.source == "hf":
        from src.ingestion.loader import stream_hf_dataset
        if args.ids:
            print(f"   Thiết lập nạp dữ liệu từ Hugging Face cho các ID: {args.ids}...")
        elif args.limit <= 0:
            print("   Thiết lập nạp toàn bộ dữ liệu từ Hugging Face...")
        else:
            print(f"   Thiết lập nạp dữ liệu từ Hugging Face (Giới hạn {args.limit} tài liệu)...")
        doc_stream = stream_hf_dataset(args.ids, limit=args.limit)
    else:
        from src.ingestion.loader import stream_directory, load_text_file
        if os.path.isdir(args.path):
            print(f"   Thiết lập nạp dữ liệu từ thư mục cục bộ: {args.path}")
            doc_stream = stream_directory(args.path)
        else:
            print(f"   Thiết lập nạp dữ liệu từ tệp cục bộ: {args.path}")
            def single_doc_stream():
                try:
                    yield load_text_file(args.path)
                except Exception as e:
                    print(f"❌ Lỗi tải file: {e}")
            doc_stream = single_doc_stream()

    # Tải các chỉ mục sẵn vào RAM để cập nhật in-memory
    print("\n📦 Đang khởi tạo/tải các chỉ mục sẵn có...")
    from src.indexing.bm25_index import get_bm25_index
    bm25 = get_bm25_index()
    if args.reset:
        bm25.corpus_chunks = []
        bm25.bm25 = None
    else:
        # Load bm25 from disk if it exists
        try:
            bm25.load()
        except FileNotFoundError:
            pass
            
    # Tải graph
    try:
        from src.retrieval.graph import get_graph
        graph = get_graph()
        if args.reset:
            import networkx as nx
            graph.graph = nx.DiGraph()
        else:
            try:
                graph.load()
            except FileNotFoundError:
                import networkx as nx
                graph.graph = nx.DiGraph()
    except ImportError:
        graph = None
        print("   ⚠️  Không thể sử dụng Knowledge Graph (networkx chưa cài)")

    from src.ingestion.cleaner import clean_text
    from src.ingestion.chunker import chunk_document
    from src.indexing.chroma_store import add_documents

    batch_docs = []
    total_processed = 0
    total_skipped = 0
    total_chunks = 0
    batch_idx = 1
    
    def process_batch(docs_to_process, batch_num, is_last_batch=False):
        nonlocal total_chunks, total_processed
        print(f"\n📦 [Lô #{batch_num}] Đang xử lý {len(docs_to_process)} tài liệu...")
        
        batch_chunks = []
        batch_processed_sources = []
        
        for doc in docs_to_process:
            doc_source = doc.get("metadata", {}).get("source", "")
            doc_name = doc.get("metadata", {}).get("filename", "unknown")
            
            # Kiểm tra tài liệu rỗng hoặc lỗi
            if not doc or not doc.get("page_content") or not doc["page_content"].strip():
                print(f"   ⚠️ Bỏ qua tài liệu rỗng hoặc lỗi: {doc_name}")
                continue
                
            try:
                # Bước 2: Làm sạch văn bản
                cleaned = clean_text(doc["page_content"])
                if not cleaned.strip():
                    print(f"   ⚠️ Bỏ qua tài liệu rỗng sau khi làm sạch: {doc_name}")
                    continue
                doc["page_content"] = cleaned
                
                # Bước 3: Chia nhỏ văn bản (chunking)
                chunks = chunk_document(doc)
                if chunks:
                    batch_chunks.extend(chunks)
                    batch_processed_sources.append(doc_source)
            except Exception as doc_err:
                print(f"   ⚠️ Lỗi xử lý tài liệu {doc_name}: {doc_err}")
                continue
        
        if not batch_chunks:
            print(f"   ℹ️ Lô #{batch_num} không có phân đoạn nào hợp lệ để lưu.")
            return
            
        # Loại bỏ các chunk trùng lặp ID trong cùng một lô trước khi upsert vào ChromaDB
        seen_chunk_ids = set()
        unique_batch_chunks = []
        for chunk in batch_chunks:
            cid = chunk.get("metadata", {}).get("chunk_id")
            if cid:
                if cid in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(cid)
            unique_batch_chunks.append(chunk)
        batch_chunks = unique_batch_chunks

        print(f"   - Chia được {len(batch_chunks)} phân đoạn duy nhất từ {len(batch_processed_sources)} tài liệu.")
        
        # Bước 4: Lưu vào ChromaDB (vector store - upsert)
        try:
            add_documents(batch_chunks)
            print("   ✅ Bước 4: Đã lưu vào ChromaDB (upsert thành công)")
        except Exception as e:
            print(f"   ❌ Lỗi lưu ChromaDB: {e}")
            raise e
            
        # Bước 5: Cập nhật BM25 index (in-memory update & save theo chu kỳ)
        try:
            bm25.append_and_build(batch_chunks, load_from_disk=False)
            if batch_num % 10 == 0 or is_last_batch:
                bm25.save()
                print(f"   ✅ Bước 5: Đã cập nhật và ghi chỉ mục BM25 xuống đĩa (Tổng {len(bm25.corpus_chunks)} phân đoạn)")
            else:
                print("   ✅ Bước 5: Đã cập nhật chỉ mục BM25 (in-memory)")
        except Exception as e:
            print(f"   ❌ Lỗi cập nhật BM25: {e}")
            raise e
            
        # Bước 6: Cập nhật Đồ thị pháp lý (Knowledge Graph - save theo chu kỳ)
        if graph is not None:
            try:
                graph.append_from_chunks(batch_chunks, load_from_disk=False)
                if batch_num % 10 == 0 or is_last_batch:
                    graph.save()
                    print(f"   ✅ Bước 6: Đã cập nhật và ghi Đồ thị pháp lý xuống đĩa (Tổng {graph.graph.number_of_nodes()} nút)")
                else:
                    print("   ✅ Bước 6: Đã cập nhật Đồ thị pháp lý (in-memory)")
            except Exception as e:
                print(f"   ⚠️ Lỗi cập nhật Đồ thị pháp lý: {e}")
                
        # Lưu checkpoint đồng bộ khi ghi đĩa
        for src in batch_processed_sources:
            processed_sources.add(src)
            
        if batch_num % 10 == 0 or is_last_batch:
            try:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                import json
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"processed_sources": list(processed_sources)}, f, ensure_ascii=False, indent=2)
                print("   💾 Đã cập nhật file checkpoint đồng bộ xuống đĩa.")
            except Exception as e:
                print(f"   ⚠️ Không thể ghi file checkpoint ({e})")
            
        total_processed += len(batch_processed_sources)
        total_chunks += len(batch_chunks)
        print(f"   [Lô #{batch_num}] Hoàn tất.")

    # Bắt đầu stream và gom lô
    try:
        for doc in doc_stream:
            doc_source = doc.get("metadata", {}).get("source", "")
            if doc_source in processed_sources:
                total_skipped += 1
                continue
                
            batch_docs.append(doc)
            if len(batch_docs) >= args.batch_size:
                process_batch(batch_docs, batch_idx, is_last_batch=False)
                batch_docs = []
                batch_idx += 1
                
        # Xử lý lô cuối cùng nếu còn sót
        if batch_docs:
            process_batch(batch_docs, batch_idx, is_last_batch=True)
            
    except Exception as stream_err:
        print(f"\n❌ Tiến trình bị gián đoạn do lỗi hệ thống: {stream_err}")
        print("💡 Bạn có thể chạy lại lệnh để tiếp tục từ checkpoint hiện tại.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 Hoàn tất! Hệ thống đã sẵn sàng để trả lời câu hỏi pháp lý.")
    print("=" * 60)
    print(f"📊 Thống kê tiến trình:")
    print(f"   - Số tài liệu đã xử lý mới: {total_processed}")
    print(f"   - Số tài liệu bỏ qua (đã có từ trước): {total_skipped}")
    print(f"   - Tổng số phân đoạn đã được nạp mới: {total_chunks}")
    print(f"\n💡 Chạy ứng dụng: python ui/app.py")


if __name__ == "__main__":
    main()
