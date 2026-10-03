"""
Stages 3 and 4 of the pipeline: embedding chunks and retrieving them.

Three things in here are worth knowing about, because they'd quietly break the
rest of the project if they were wrong:

1. The Chroma collection is created with cosine distance, explicitly. Chroma
   defaults to squared L2, and the 0.6 threshold the course uses is calibrated
   against cosine. Getting this wrong makes every distance number meaningless.

2. `search` returns the distance alongside each chunk. Milestone 4 has you
   compare distances, so they have to be visible.

3. The embedding model is the one Chroma bundles, not one loaded through
   `sentence-transformers`. It is the same model — `all-MiniLM-L6-v2`, 384
   dimensions — but it arrives as an ONNX build from Chroma's own CDN, so the
   install needs neither PyTorch nor a reachable Hugging Face. See `_embedder`.
"""

import os
import shutil
from dataclasses import dataclass

# Must be set BEFORE chromadb is imported. Without it, some Chroma versions
# print "Failed to send telemetry event ..." on every single call — which looks
# exactly like a real error, isn't one, and cost a previous cohort a lot of
# confused help-channel messages.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402

import config
from chunker import Chunk


@dataclass
class Result:
    """One retrieved chunk and how far it was from the question."""

    text: str
    source: str
    label: str
    distance: float   # LOWER IS BETTER. 0.3 is close, 0.9 is unrelated.
    produced_by: str


_model = None

# The model Chroma bundles. Anything else in config.EMBEDDING_MODEL means
# "fetch that one from Hugging Face instead" — see `_embedder`.
BUNDLED_MODEL = "all-MiniLM-L6-v2"


class _OnnxEmbedder:
    """
    Chroma's built-in embedder, wrapped to look like the other two.

    Chroma's embedding functions are called directly and hand back numpy
    arrays. The rest of this file wants `.encode(texts)`, so the adapter lives
    here rather than making every caller care which embedder it got.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, show_progress_bar: bool = False):
        return [vector.tolist() for vector in self._ef(list(texts))]


def _sentence_transformer(name: str):
    """
    The escape hatch: any model that isn't the bundled one.

    Unit 2's "try a second embedding model" stretch option comes through here,
    and so does anything you set `EMBEDDING_MODEL` to. This path *does* need
    `sentence-transformers` and a reachable Hugging Face, neither of which the
    default install has — which is the whole point of the default install.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"config.EMBEDDING_MODEL is set to {name!r}, which isn't the model "
            f"Chroma bundles ({BUNDLED_MODEL!r}), so it has to be downloaded "
            f"from Hugging Face.\n"
            f"Install the optional dependency first:\n"
            f"    pip install 'sentence-transformers>=3.4,<3.5'\n"
            f"Or set EMBEDDING_MODEL back to {BUNDLED_MODEL!r}."
        ) from exc

    return SentenceTransformer(name)


def _embedder():
    """
    Load the embedding model once and keep it.

    First call is slow — it downloads about 80 MB. That's why setup happens
    before class.
    """
    global _model

    if _model is not None:
        return _model

    # Used only by this repo's own smoke test, which runs where no model can be
    # downloaded at all. Never set this yourself.
    if os.getenv("AI201_FAKE_EMBEDDINGS") == "1":
        from _smoke_embedder import FakeEmbedder

        _model = FakeEmbedder()
    elif config.EMBEDDING_MODEL == BUNDLED_MODEL:
        _model = _OnnxEmbedder()
    else:
        _model = _sentence_transformer(config.EMBEDDING_MODEL)

    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Turn text into vectors. Runs on your machine, costs no API quota."""
    vectors = _embedder().encode(texts, show_progress_bar=False)
    # sentence-transformers and the smoke stand-in return something with a
    # .tolist(); _OnnxEmbedder has already done that conversion itself.
    return vectors.tolist() if hasattr(vectors, "tolist") else vectors


def _client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def build_index(
    chunks: list[Chunk],
    corpus: str | None = None,
    variant: str = "default",
) -> int:
    """
    Embed every chunk and store it.

    `variant` lets you keep more than one index of the same corpus at the same
    time. In unit 2, when you compare two chunking strategies, index the second
    one as variant="v2" and you can query both instead of deleting the first
    and starting over.
    """
    name = config.collection_name(corpus, variant)
    client = _client()

    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        # ⚠️ Do not remove. Chroma defaults to squared L2, and every distance
        # number in this course assumes cosine.
        metadata={"hnsw:space": "cosine"},
    )

    batch = 256
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.add(
            ids=[f"{c.source}#{c.index}" for c in window],
            documents=[c.text for c in window],
            embeddings=embed([c.text for c in window]),
            metadatas=[
                {"source": c.source, "index": c.index, "produced_by": c.produced_by}
                for c in window
            ],
        )

    return len(chunks)


import re
from rank_bm25 import BM25Okapi

_bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[dict], list[str]]] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_bm25_index(collection, name: str):
    if name not in _bm25_cache or len(_bm25_cache[name][1]) != collection.count():
        data = collection.get()
        docs = data["documents"]
        metas = data["metadatas"]
        ids = data["ids"]
        tokens = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokens)
        _bm25_cache[name] = (bm25, docs, metas, ids)
    return _bm25_cache[name]


def search(
    question: str,
    top_k: int | None = None,
    corpus: str | None = None,
    variant: str = "default",
    hybrid: bool = True,
) -> list[Result]:
    """
    Retrieve the chunks closest to a question.

    Uses Hybrid Search (BM25 keyword search + dense semantic vectors combined
    via Reciprocal Rank Fusion) by default, providing both keyword precision for
    proper nouns/entities and semantic understanding for paraphrases.
    """
    top_k = top_k or config.TOP_K
    name = config.collection_name(corpus, variant)

    try:
        collection = _client().get_collection(name)
    except Exception as exc:
        raise RuntimeError(
            f"No index called '{name}'. Run `python app.py index` first."
        ) from exc

    total_count = collection.count()
    if total_count == 0:
        return []

    # 1. Dense semantic search
    dense_fetch_k = min(20, total_count)
    raw = collection.query(
        query_embeddings=embed([question]),
        n_results=dense_fetch_k,
    )

    dense_docs = raw["documents"][0]
    dense_metas = raw["metadatas"][0]
    dense_distances = raw["distances"][0]
    dense_ids = raw["ids"][0]

    if not hybrid:
        results: list[Result] = []
        for text, meta, distance in zip(
            dense_docs[:top_k], dense_metas[:top_k], dense_distances[:top_k]
        ):
            results.append(
                Result(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    label=f"{meta.get('source', 'unknown')}#{meta.get('index', 0)}",
                    distance=float(distance),
                    produced_by=str(meta.get("produced_by", "unknown")),
                )
            )
        return results

    # 2. BM25 keyword search
    bm25, all_docs, all_metas, all_ids = _get_bm25_index(collection, name)
    q_tokens = _tokenize(question)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )

    dense_rank = {dense_ids[i]: i + 1 for i in range(len(dense_ids))}
    id_dist = {dense_ids[i]: float(dense_distances[i]) for i in range(len(dense_ids))}
    id_to_idx = {all_ids[i]: i for i in range(len(all_ids))}
    bm25_rank = {
        all_ids[bm25_ranked_indices[r]]: r + 1
        for r in range(len(bm25_ranked_indices))
    }

    # 3. Reciprocal Rank Fusion (RRF)
    k = 60
    candidates = set(dense_ids) | set(
        all_ids[bm25_ranked_indices[r]] for r in range(min(15, len(all_ids)))
    )
    fused_scores = {}
    for cid in candidates:
        r_dense = dense_rank.get(cid, 100)
        r_bm25 = bm25_rank.get(cid, 100)
        fused_scores[cid] = (1.0 / (k + r_dense)) + (1.0 / (k + r_bm25))

    sorted_cids = sorted(
        candidates, key=lambda cid: fused_scores[cid], reverse=True
    )[:top_k]

    results: list[Result] = []
    for cid in sorted_cids:
        idx = id_to_idx[cid]
        dist = id_dist.get(cid)
        if dist is None:
            dist = 1.0
        results.append(
            Result(
                text=all_docs[idx],
                source=str(all_metas[idx].get("source", "unknown")),
                label=f"{all_metas[idx].get('source', 'unknown')}#{all_metas[idx].get('index', 0)}",
                distance=dist,
                produced_by=str(all_metas[idx].get("produced_by", "unknown")),
            )
        )
    return results


def index_exists(corpus: str | None = None, variant: str = "default") -> bool:
    """Is there an index here to search, without searching it?

    `serve.py`'s health check asks this. It deliberately does not embed
    anything: loading the embedding model takes 80 MB and a few seconds, and a
    health check that heavy is a health check nobody can afford to call.
    """
    try:
        collection = _client().get_collection(config.collection_name(corpus, variant))
        return collection.count() > 0
    except Exception:
        return False


def reset():
    """Delete every index. Occasionally the fastest way out of a mess."""
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
