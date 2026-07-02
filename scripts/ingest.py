"""
CLI Script — Ingest legal documents into the RAG system.

Usage:
    python scripts/ingest.py --path data/legal_docs/
"""
import argparse
import sys
import os

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Nạp dữ liệu văn bản pháp luật vào hệ thống RAG."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "legal_docs"),
        help="Đường dẫn tới thư mục hoặc file chứa văn bản pháp luật.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  LEGAL RAG — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    print("\n📂 Bước 1: Đọc tài liệu...")
    from src.ingestion.loader import load_directory, load_text_file

    if os.path.isdir(args.path):
        docs = load_directory(args.path)
    else:
        docs = [load_text_file(args.path)]

    print(f"   Đã đọc {len(docs)} tài liệu")
    for doc in docs:
        print(f"   - {doc['metadata']['filename']}: {len(doc['page_content'])} ký tự")

    # Step 2: Clean text
    print("\n🧹 Bước 2: Làm sạch văn bản...")
    from src.ingestion.cleaner import clean_text

    for doc in docs:
        doc["page_content"] = clean_text(doc["page_content"])
    print("   ✅ Hoàn tất")

    # Step 3: Chunk documents
    print("\n✂️  Bước 3: Chia nhỏ văn bản (chunking)...")
    from src.ingestion.chunker import chunk_document

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"   - {doc['metadata']['filename']}: {len(chunks)} phân đoạn")
    print(f"   Tổng cộng: {len(all_chunks)} phân đoạn")

    # Step 4: Store in ChromaDB
    print("\n💾 Bước 4: Lưu vào ChromaDB (vector store)...")
    from src.indexing.chroma_store import add_documents

    add_documents(all_chunks)

    # Step 5: Build BM25 index
    print("\n🔤 Bước 5: Xây dựng BM25 index (keyword search)...")
    from src.indexing.bm25_index import BM25Index

    bm25 = BM25Index()
    bm25.build(all_chunks)
    bm25.save()

    # Step 6: Build knowledge graph
    print("\n🕸️  Bước 6: Xây dựng đồ thị pháp lý (knowledge graph)...")
    try:
        from src.retrieval.graph import build_graph
        build_graph(all_chunks)
    except ImportError as e:
        print(f"   ⚠️  Bỏ qua graph ({e})")

    print("\n" + "=" * 60)
    print("🎉 Hoàn tất! Hệ thống đã sẵn sàng để trả lời câu hỏi pháp lý.")
    print("=" * 60)
    print(f"\n💡 Chạy ứng dụng: python ui/app.py")


if __name__ == "__main__":
    main()
