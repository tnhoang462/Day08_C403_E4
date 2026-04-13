import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import index


class FakeCollection:
    def __init__(self, get_result=None):
        self.upserts = []
        self._get_result = get_result or {"documents": [], "metadatas": []}

    def upsert(self, ids, embeddings, documents, metadatas):
        self.upserts.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )

    def get(self, limit=None, include=None):
        if limit is None:
            return self._get_result
        return {
            "documents": self._get_result["documents"][:limit],
            "metadatas": self._get_result["metadatas"][:limit],
        }


class FakeClient:
    def __init__(self, collection):
        self.collection = collection
        self.created_with = None

    def get_or_create_collection(self, name, metadata):
        self.created_with = {"name": name, "metadata": metadata}
        return self.collection

    def get_collection(self, name):
        return self.collection


class TestPreprocessDocument(unittest.TestCase):
    def test_preprocess_extracts_metadata_and_removes_header_lines(self):
        raw_text = """REFUND POLICY
Source: policy/refund-v4.pdf
Department: Finance
Effective Date: 2026-01-01
Access: internal

=== Eligibility ===
Customers can request a refund.
"""

        result = index.preprocess_document(raw_text, "fallback.txt")

        self.assertEqual(result["metadata"]["source"], "policy/refund-v4.pdf")
        self.assertEqual(result["metadata"]["department"], "Finance")
        self.assertEqual(result["metadata"]["effective_date"], "2026-01-01")
        self.assertEqual(result["metadata"]["access"], "internal")
        self.assertEqual(result["metadata"]["section"], "")
        self.assertIn("=== Eligibility ===", result["text"])
        self.assertNotIn("Department: Finance", result["text"])

    def test_preprocess_uses_defaults_and_normalizes_blank_lines(self):
        raw_text = """DOCUMENT TITLE

=== General ===
Line 1



Line 2
"""

        result = index.preprocess_document(raw_text, "fallback.txt")

        self.assertEqual(result["metadata"]["source"], "fallback.txt")
        self.assertEqual(result["metadata"]["department"], "unknown")
        self.assertEqual(result["metadata"]["effective_date"], "unknown")
        self.assertEqual(result["metadata"]["access"], "internal")
        self.assertNotIn("\n\n\n", result["text"])


class TestChunking(unittest.TestCase):
    def test_split_by_size_returns_single_chunk_for_short_text(self):
        chunks = index._split_by_size(
            "short text",
            base_metadata={"source": "a.txt"},
            section="General",
            chunk_chars=100,
            overlap_chars=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "short text")
        self.assertEqual(chunks[0]["metadata"]["section"], "General")
        self.assertEqual(chunks[0]["metadata"]["source"], "a.txt")

    def test_split_by_size_creates_overlapping_chunks(self):
        text = "abcdefghij"

        chunks = index._split_by_size(
            text,
            base_metadata={"source": "a.txt"},
            section="Long",
            chunk_chars=4,
            overlap_chars=1,
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["abcd", "defg", "ghij"])
        self.assertTrue(all(chunk["metadata"]["section"] == "Long" for chunk in chunks))

    def test_chunk_document_splits_sections_and_preserves_metadata(self):
        doc = {
            "text": "=== First ===\nAlpha\n=== Second ===\nBeta",
            "metadata": {
                "source": "doc.txt",
                "department": "HR",
                "effective_date": "2026-01-01",
                "access": "internal",
                "section": "",
            },
        }

        with patch.object(index, "_split_by_size") as split_mock:
            split_mock.side_effect = lambda text, base_metadata, section: [
                {"text": text, "metadata": {**base_metadata, "section": section}}
            ]

            chunks = index.chunk_document(doc)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["metadata"]["section"], "First")
        self.assertEqual(chunks[1]["metadata"]["section"], "Second")
        self.assertTrue(all(chunk["metadata"]["department"] == "HR" for chunk in chunks))


class TestGetEmbedding(unittest.TestCase):
    def test_get_embedding_returns_vector_when_openai_succeeds(self):
        fake_response = types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
        )

        class FakeOpenAI:
            def __init__(self, api_key=None):
                self.embeddings = types.SimpleNamespace(create=lambda input, model: fake_response)

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)

        with patch.dict(sys.modules, {"openai": fake_module}):
            embedding = index.get_embedding("hello")

        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_get_embedding_returns_none_when_openai_fails(self):
        class FakeOpenAI:
            def __init__(self, api_key=None):
                def raise_error(input, model):
                    raise RuntimeError("boom")

                self.embeddings = types.SimpleNamespace(create=raise_error)

        fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
        stdout = io.StringIO()

        with patch.dict(sys.modules, {"openai": fake_module}):
            with redirect_stdout(stdout):
                embedding = index.get_embedding("hello")

        self.assertIsNone(embedding)
        self.assertIn("API OPENAI-Embedding Error", stdout.getvalue())


class TestBuildIndex(unittest.TestCase):
    def test_build_index_reads_docs_and_upserts_chunks(self):
        collection = FakeCollection()
        client = FakeClient(collection)
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            db_dir = Path(tmpdir) / "db"
            docs_dir.mkdir()
            (docs_dir / "policy.txt").write_text("raw content", encoding="utf-8")

            fake_doc = {"text": "cleaned", "metadata": {"source": "policy.txt"}}
            fake_chunks = [
                {"text": "chunk 1", "metadata": {"source": "policy.txt", "section": "A"}},
                {"text": "chunk 2", "metadata": {"source": "policy.txt", "section": "B"}},
            ]

            stdout = io.StringIO()
            with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
                with patch.object(index, "preprocess_document", return_value=fake_doc) as preprocess_mock:
                    with patch.object(index, "chunk_document", return_value=fake_chunks) as chunk_mock:
                        with patch.object(index, "get_embedding", side_effect=[[0.1], [0.2]]) as embed_mock:
                            with redirect_stdout(stdout):
                                index.build_index(docs_dir=docs_dir, db_dir=db_dir)

            self.assertTrue(db_dir.exists())

        self.assertEqual(preprocess_mock.call_count, 1)
        self.assertEqual(chunk_mock.call_count, 1)
        self.assertEqual(embed_mock.call_count, 2)
        self.assertEqual(len(collection.upserts), 2)
        self.assertEqual(collection.upserts[0]["ids"], ["policy_0"])
        self.assertEqual(collection.upserts[1]["ids"], ["policy_1"])
        self.assertIn("Hoàn thành! Tổng số chunks: 2", stdout.getvalue())

    def test_build_index_prints_message_when_no_txt_files(self):
        collection = FakeCollection()
        client = FakeClient(collection)
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            db_dir = Path(tmpdir) / "db"
            docs_dir.mkdir()
            stdout = io.StringIO()

            with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
                with redirect_stdout(stdout):
                    index.build_index(docs_dir=docs_dir, db_dir=db_dir)

        self.assertIn("Không tìm thấy file .txt", stdout.getvalue())
        self.assertEqual(collection.upserts, [])


class TestInspectFunctions(unittest.TestCase):
    def test_list_chunks_prints_chunk_preview(self):
        collection = FakeCollection(
            {
                "documents": ["Document body"],
                "metadatas": [
                    {
                        "source": "doc.txt",
                        "section": "General",
                        "effective_date": "2026-01-01",
                    }
                ],
            }
        )
        client = FakeClient(collection)
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        stdout = io.StringIO()

        with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
            with redirect_stdout(stdout):
                index.list_chunks(db_dir=Path("unused"), n=1)

        output = stdout.getvalue()
        self.assertIn("Top 1 chunks", output)
        self.assertIn("Source: doc.txt", output)
        self.assertIn("Section: General", output)

    def test_inspect_metadata_coverage_prints_department_counts(self):
        collection = FakeCollection(
            {
                "documents": [],
                "metadatas": [
                    {"department": "HR", "effective_date": "2026-01-01"},
                    {"department": "HR", "effective_date": "unknown"},
                    {"department": "IT", "effective_date": ""},
                ],
            }
        )
        client = FakeClient(collection)
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        stdout = io.StringIO()

        with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
            with redirect_stdout(stdout):
                index.inspect_metadata_coverage(db_dir=Path("unused"))

        output = stdout.getvalue()
        self.assertIn("Tổng chunks: 3", output)
        self.assertIn("HR: 2 chunks", output)
        self.assertIn("IT: 1 chunks", output)
        self.assertIn("Chunks thiếu effective_date: 2", output)


if __name__ == "__main__":
    unittest.main()
