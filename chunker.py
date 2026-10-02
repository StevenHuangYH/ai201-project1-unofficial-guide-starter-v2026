"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

from dataclasses import dataclass

import config
from ingest import Document


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


import re


def _split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Split documents into chunks. Replaces the generic fallback chunker.

    Strategy for campus_life:
      - Posts are short (average ~317 chars). Posts under chunk_size are preserved
        whole to maintain full topic coherence and heading context.
      - Posts exceeding chunk_size are split along paragraph boundaries. If a
        single paragraph exceeds chunk_size, it is split on complete sentence
        boundaries.
      - Each split chunk retains the document title/heading so context is never
        lost (e.g. which dorm or course is being reviewed).
      - Overlap preserves boundary paragraphs or sentences without cutting
        words or thoughts in half.
      - Sets produced_by to 'chunker.py::split_documents'.
    """
    max_chars = chunk_size or config.CHUNK_SIZE
    overlap_chars = overlap or config.CHUNK_OVERLAP

    chunks: list[Chunk] = []
    for doc in documents:
        text = doc.text.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        title = lines[0] if lines else ""

        # If the entire post fits within chunk_size, keep it intact
        if len(text) <= max_chars:
            chunks.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    index=0,
                    produced_by="chunker.py::split_documents",
                )
            )
            continue

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        has_title = len(paragraphs) > 1 and (paragraphs[0] == title or len(paragraphs[0]) < 60)
        heading_prefix = f"{title}\n\n" if has_title else ""
        body_paras = paragraphs[1:] if has_title else paragraphs

        curr_paras: list[str] = []
        curr_len = len(heading_prefix)
        index = 0

        for p in body_paras:
            # If paragraph itself is too large, split into sentences
            if len(p) + len(heading_prefix) > max_chars:
                sentences = _split_into_sentences(p)
                curr_sents: list[str] = []
                s_len = len(heading_prefix)
                for s in sentences:
                    if curr_sents and (s_len + len(s) + 1 > max_chars):
                        c_text = heading_prefix + " ".join(curr_sents)
                        chunks.append(
                            Chunk(
                                text=c_text.strip(),
                                source=doc.source,
                                index=index,
                                produced_by="chunker.py::split_documents",
                            )
                        )
                        index += 1
                        # Overlap: keep the last sentence
                        curr_sents = [curr_sents[-1]] if curr_sents else []
                        s_len = len(heading_prefix) + (len(curr_sents[0]) + 1 if curr_sents else 0)
                    curr_sents.append(s)
                    s_len += len(s) + 1
                if curr_sents:
                    c_text = heading_prefix + " ".join(curr_sents)
                    chunks.append(
                        Chunk(
                            text=c_text.strip(),
                            source=doc.source,
                            index=index,
                            produced_by="chunker.py::split_documents",
                        )
                    )
                    index += 1
                continue

            # If adding this paragraph exceeds max_chars, flush current chunk
            if curr_paras and (curr_len + len(p) + 2 > max_chars):
                c_text = heading_prefix + "\n\n".join(curr_paras)
                chunks.append(
                    Chunk(
                        text=c_text.strip(),
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::split_documents",
                    )
                )
                index += 1
                # Overlap: keep previous paragraph if it fits in overlap_chars
                if len(curr_paras[-1]) <= overlap_chars:
                    curr_paras = [curr_paras[-1]]
                    curr_len = len(heading_prefix) + len(curr_paras[0]) + 2
                else:
                    curr_paras = []
                    curr_len = len(heading_prefix)

            curr_paras.append(p)
            curr_len += len(p) + 2

        if curr_paras:
            c_text = heading_prefix + "\n\n".join(curr_paras)
            chunks.append(
                Chunk(
                    text=c_text.strip(),
                    source=doc.source,
                    index=index,
                    produced_by="chunker.py::split_documents",
                )
            )

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
