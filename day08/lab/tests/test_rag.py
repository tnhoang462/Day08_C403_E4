import io
import shutil
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import index
import rag_answer


class FakeCollection:
    def __init__(self, query_result=None):
        self.query_result = query_result or {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.query_calls = []

    def query(self, query_embeddings, n_results, include):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include,
            }
        )
        return self.query_result


class FakeClient:
    def __init__(self, collection):
        self.collection = collection
        self.requested_collection = None

    def get_collection(self, name):
        self.requested_collection = name
        return self.collection


class ChromaDirTestCase(unittest.TestCase):
    def setUp(self):
        self.created_fake_db = False
        if index.CHROMA_DB_DIR.exists():
            self.chroma_db_dir = index.CHROMA_DB_DIR
        else:
            self.chroma_db_dir = Path(__file__).resolve().parents[1] / "fake-db"
            self.chroma_db_dir.mkdir(exist_ok=True)
            self.created_fake_db = True

    def tearDown(self):
        if self.created_fake_db and self.chroma_db_dir.exists():
            shutil.rmtree(self.chroma_db_dir)


class TestRetrieveDense(ChromaDirTestCase):
    def test_retrieve_dense_maps_chroma_results_to_chunks(self):
        collection = FakeCollection(
            query_result={
                "documents": [["chunk 1", "chunk 2"]],
                "metadatas": [[
                    {"source": "a.txt", "section": "Intro"},
                    {"source": "b.txt", "section": "Policy"},
                ]],
                "distances": [[0.1, 0.25]],
            }
        )
        client = FakeClient(collection)
        fake_chromadb = types.SimpleNamespace(PersistentClient=lambda path: client)
        fake_index = types.SimpleNamespace(
            CHROMA_DB_DIR=self.chroma_db_dir,
            get_embedding=lambda query: [0.5, 0.6],
        )

        with patch.dict(sys.modules, {"chromadb": fake_chromadb, "index": fake_index}):
            results = rag_answer.retrieve_dense("refund", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "chunk 1")
        self.assertEqual(results[0]["metadata"]["source"], "a.txt")
        self.assertAlmostEqual(results[0]["score"], 0.9)
        self.assertEqual(collection.query_calls[0]["n_results"], 2)
        self.assertEqual(
            collection.query_calls[0]["include"],
            ["documents", "metadatas", "distances"],
        )

    def test_retrieve_dense_returns_empty_list_on_error(self):
        fake_index = types.SimpleNamespace(
            CHROMA_DB_DIR=self.chroma_db_dir,
            get_embedding=lambda query: [0.5, 0.6],
        )
        fake_chromadb = types.SimpleNamespace(
            PersistentClient=lambda path: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        stdout = io.StringIO()

        with patch.dict(sys.modules, {"chromadb": fake_chromadb, "index": fake_index}, clear=False):
            with redirect_stdout(stdout):
                results = rag_answer.retrieve_dense("refund", top_k=2)

        self.assertEqual(results, [])
        self.assertIn("Error in retrieve_dense", stdout.getvalue())


class TestHelpers(unittest.TestCase):
    def test_retrieve_sparse_returns_empty_list_and_logs_placeholder(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            results = rag_answer.retrieve_sparse("P1", top_k=3)

        self.assertEqual(results, [])
        self.assertIn("Chưa implement", stdout.getvalue())

    def test_retrieve_hybrid_falls_back_to_dense(self):
        dense_results = [{"text": "alpha", "metadata": {"source": "a.txt"}, "score": 0.8}]
        stdout = io.StringIO()

        with patch.object(rag_answer, "retrieve_dense", return_value=dense_results) as dense_mock:
            with redirect_stdout(stdout):
                results = rag_answer.retrieve_hybrid("approval", top_k=4)

        self.assertEqual(results, dense_results)
        dense_mock.assert_called_once_with("approval", 4)

    def test_rerank_returns_first_top_k_candidates(self):
        candidates = [
            {"text": "one", "metadata": {"source": "1.txt"}},
            {"text": "two", "metadata": {"source": "2.txt"}},
            {"text": "three", "metadata": {"source": "3.txt"}},
        ]

        results = rag_answer.rerank("query", candidates, top_k=2)

        self.assertEqual(results, candidates[:2])

    def test_transform_query_returns_original_query(self):
        self.assertEqual(rag_answer.transform_query("hello"), ["hello"])

    def test_build_context_block_formats_source_section_and_score(self):
        chunks = [
            {
                "text": "SLA P1 is 1 hour.",
                "metadata": {"source": "policy.txt", "section": "SLA"},
                "score": 0.875,
            },
            {
                "text": "Fallback chunk.",
                "metadata": {"source": "faq.txt"},
                "score": 0,
            },
        ]

        context = rag_answer.build_context_block(chunks)

        self.assertIn("[1] policy.txt | SLA | score=0.88", context)
        self.assertIn("SLA P1 is 1 hour.", context)
        self.assertIn("[2] faq.txt", context)
        self.assertNotIn("score=0.00", context)

    def test_build_grounded_prompt_includes_query_and_context(self):
        prompt = rag_answer.build_grounded_prompt(
            "SLA ticket P1?",
            "[1] policy.txt | SLA\nP1 is handled in 1 hour.",
        )

        self.assertIn("Question: SLA ticket P1?", prompt)
        self.assertIn("Context:\n[1] policy.txt | SLA", prompt)
        self.assertIn("Respond in the same language as the question.", prompt)


class TestRagPipeline(unittest.TestCase):
    def test_rag_answer_uses_dense_pipeline_without_rerank(self):
        candidates = [
            {"text": "A", "metadata": {"source": "a.txt", "section": "S1"}, "score": 0.9},
            {"text": "B", "metadata": {"source": "b.txt", "section": "S2"}, "score": 0.8},
            {"text": "C", "metadata": {"source": "a.txt", "section": "S3"}, "score": 0.7},
        ]

        with patch.object(rag_answer, "retrieve_dense", return_value=candidates) as retrieve_mock:
            with patch.object(rag_answer, "build_context_block", return_value="CTX") as context_mock:
                with patch.object(rag_answer, "build_grounded_prompt", return_value="PROMPT") as prompt_mock:
                    with patch.object(rag_answer, "call_llm", return_value="ANSWER") as llm_mock:
                        result = rag_answer.rag_answer(
                            "What is SLA?",
                            retrieval_mode="dense",
                            top_k_search=5,
                            top_k_select=2,
                            use_rerank=False,
                        )

        retrieve_mock.assert_called_once_with("What is SLA?", top_k=5)
        context_mock.assert_called_once_with(candidates[:2])
        prompt_mock.assert_called_once_with("What is SLA?", "CTX")
        llm_mock.assert_called_once_with("PROMPT")
        self.assertEqual(result["answer"], "ANSWER")
        self.assertEqual(result["chunks_used"], candidates[:2])
        self.assertEqual(set(result["sources"]), {"a.txt", "b.txt"})
        self.assertEqual(result["config"]["retrieval_mode"], "dense")

    def test_rag_answer_uses_rerank_when_enabled(self):
        retrieved = [
            {"text": "A", "metadata": {"source": "a.txt"}, "score": 0.9},
            {"text": "B", "metadata": {"source": "b.txt"}, "score": 0.8},
        ]
        reranked = [retrieved[1]]

        with patch.object(rag_answer, "retrieve_dense", return_value=retrieved):
            with patch.object(rag_answer, "rerank", return_value=reranked) as rerank_mock:
                with patch.object(rag_answer, "build_context_block", return_value="CTX"):
                    with patch.object(rag_answer, "build_grounded_prompt", return_value="PROMPT"):
                        with patch.object(rag_answer, "call_llm", return_value="ANSWER"):
                            result = rag_answer.rag_answer(
                                "What is SLA?",
                                use_rerank=True,
                                top_k_select=1,
                            )

        rerank_mock.assert_called_once_with("What is SLA?", retrieved, top_k=1)
        self.assertEqual(result["chunks_used"], reranked)
        self.assertEqual(result["sources"], ["b.txt"])

    def test_rag_answer_raises_for_invalid_mode(self):
        with self.assertRaises(ValueError):
            rag_answer.rag_answer("test", retrieval_mode="invalid")


if __name__ == "__main__":
    unittest.main()
