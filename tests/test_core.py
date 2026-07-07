import unittest

from src.ingestion.chunker import chunk_document
from src.indexing.chroma_store import _stable_chunk_id
from src.retrieval.hybrid import reciprocal_rank_fusion, _deduplicate


class ChunkerTests(unittest.TestCase):
    def test_chunk_document_preserves_article_metadata(self):
        doc = {
            "page_content": "LUẬT TEST\n\nĐiều 1. Nội dung một.\n\nĐiều 2. Nội dung hai.",
            "metadata": {"filename": "test.txt", "source": "/tmp/test.txt"},
        }

        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0)

        self.assertEqual([c["metadata"]["article"] for c in chunks], ["header", "1", "2"])

    def test_chunk_document_splits_by_clause(self):
        doc = {
            "page_content": "Điều 1. Nội dung điều 1\n1. Đây là khoản một có độ dài tương đối để test việc chia tách.\n2. Đây là khoản hai của điều luật này.",
            "metadata": {"filename": "test.txt", "source": "/tmp/test.txt"},
        }
        # Set a small chunk_size to force splitting on Clause level
        chunks = chunk_document(doc, chunk_size=80, chunk_overlap=0)
        
        # Verify that we split into multiple chunks and preserved clause boundaries
        self.assertTrue(len(chunks) >= 2)
        # Check that the clauses are split cleanly
        self.assertIn("1. Đây là khoản một", chunks[1]["page_content"])
        self.assertIn("2. Đây là khoản hai", chunks[-1]["page_content"])


class RetrievalTests(unittest.TestCase):
    def test_rrf_prefers_documents_appearing_in_multiple_lists(self):
        doc_a = {"page_content": "Điều 1. A", "metadata": {"article": "1"}}
        doc_b = {"page_content": "Điều 2. B", "metadata": {"article": "2"}}
        doc_c = {"page_content": "Điều 3. C", "metadata": {"article": "3"}}

        fused = reciprocal_rank_fusion([doc_a, doc_b], [doc_c, doc_a], k=60)

        self.assertEqual(fused[0]["page_content"], "Điều 1. A")
        self.assertIn("rrf_score", fused[0])

    def test_deduplicate_keeps_first_occurrence(self):
        first = {"page_content": "same content", "metadata": {"source": "s", "article": "1"}}
        second = {"page_content": "same content", "metadata": {"source": "s", "article": "1"}}

        self.assertEqual(_deduplicate([first, second]), [first])


class IndexingTests(unittest.TestCase):
    def test_stable_chunk_id_is_repeatable(self):
        chunk = {
            "page_content": "Điều 1. Nội dung",
            "metadata": {"source": "law.txt", "article": "1"},
        }

        self.assertEqual(_stable_chunk_id(chunk, 0), _stable_chunk_id(chunk, 0))


if __name__ == "__main__":
    unittest.main()
