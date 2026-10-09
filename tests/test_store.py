import sys

import store


class CountingEmbedder:
    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def test_torch_and_sentence_transformers_are_not_imported():
    import api  # noqa: F401  (the whole app is loaded by conftest)

    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_embedder_is_created_lazily_and_only_once(monkeypatch):
    CountingEmbedder.instances = 0
    monkeypatch.setattr(store, "ONNXMiniLM_L6_V2", CountingEmbedder)
    monkeypatch.setattr(store, "_embedder", None)

    assert CountingEmbedder.instances == 0  # nothing loaded before first use
    store.embed(["first"])
    store.embed(["second", "third"])

    assert CountingEmbedder.instances == 1
    assert store.get_embedder() is store.get_embedder()


def test_embed_runs_in_small_batches_and_keeps_order(monkeypatch):
    embedder = CountingEmbedder()
    monkeypatch.setattr(store, "_embedder", embedder)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = store.embed(texts)

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]
    assert all(len(batch) <= store.EMBED_BATCH_SIZE for batch in embedder.calls)
    assert [t for batch in embedder.calls for t in batch] == texts


def test_embedder_is_chromas_onnx_minilm():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    assert store.ONNXMiniLM_L6_V2 is ONNXMiniLM_L6_V2
    assert ONNXMiniLM_L6_V2.MODEL_NAME == store.EMBED_MODEL_NAME == "all-MiniLM-L6-v2"
