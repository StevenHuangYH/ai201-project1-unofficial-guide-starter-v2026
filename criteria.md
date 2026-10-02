# Acceptance criteria — The Unofficial Guide

Five criteria that say what "working" means for this system, written in unit 1
**before** any results existed.

An acceptance criterion names a target: a number, a count, a rate, or something
a person could plainly observe. *"Retrieval works"* is an opinion. *"For at
least 4 of my 5 test questions, the top results include a chunk containing the
answer"* is a criterion.

Under each one, write a sentence or two on **why that target** and not a
stricter or looser one. A reason that says something about your corpus or your
pipeline earns credit; *"80% seemed reasonable"* does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## 1. Retrieved chunks contain the answer

For at least 4 of my 5 test questions, the retrieved chunks include one that
contains the answer.

**Why this target:**
I picked 4 of 5 because while every test question has a direct corresponding document in `campus_life`, one question (Fenwick Court laundry) shares vocabulary and concepts with multiple other residence hall laundry documents, meaning semantic retrieval could potentially rank distractor dorm documents ahead of the exact match.

---

## 2. Every answer names a source

Every answer the system produces names at least one source document.

**Why this target:**
I set this to all 5 because `generate.py` injects a strict `GROUNDING_INSTRUCTION` that commands the model to name the source document, and `build_prompt` prefixes every retrieved chunk with an explicit `[from filename]` label. If any answer omits a source, it indicates a failure of prompt compliance.

---

## 3. The relevance gate stops out-of-corpus questions

When I ask a question my documents clearly don't cover, the relevance gate
stops it and the system returns "I don't have enough information about that" —
in at least 4 of 5 tries.

**Why this target:**
I set this to 4 of 5 because the relevance gate stops queries with cosine distance greater than our threshold before generation ever occurs. For out-of-scope queries (like diesel engines or world capitals), distances consistently sit well above 0.70, though 4 of 5 leaves tolerance for queries that might share accidental surface vocabulary with campus topics.

---

## 4. Complete semantic thoughts with no mid-sentence cuts

Across all generated chunks in the corpus, 100% of chunks begin on a clean heading or sentence boundary and terminate on valid sentence-ending punctuation (. ! ?), with zero sentences split across chunks.

**Why this target:**
Documents in `campus_life` are short (averaging ~317 characters across 1–3 paragraphs). Fixed character chunking frequently cuts through sentences or creates tiny 2-character trailing fragments. Enforcing structural paragraph and sentence boundary splitting guarantees that every chunk stands alone as an intelligible unit.

---

## 5. Ground-truth source attribution

For at least 4 of the 5 test questions, the source document cited in the generated answer matches the specific ground-truth document that contains the answer.

**Why this target:**
Criterion 2 only verifies that *some* source filename is named; Criterion 5 tests that the model actually attributes the answer to the correct ground-truth document rather than citing a distractor retrieved in top-k. A target of 4 of 5 allows for cases where closely related documents (e.g. course overview vs course exams) are both present in the prompt context.



---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 2 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath it, like this:

         ## 1. Retrieved chunks contain the answer

         For at least 4 of my 5 test questions, the retrieved chunks include
         one that contains the answer.

         **Why this target:** ...

         > **Revised in unit 2:** For at least 4 of 5 questions, the top three
         > results contain the answer.
         >
         > **Why revised:** I couldn't judge "the chunks include one that
         > contains the answer" the same way twice — I scored two questions
         > differently on Monday than on Wednesday. The new version is
         > something I can actually check.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "I said 4 of 5 but got 2 of 5, so 2 of 5 is more realistic."

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.

     The whole reason the originals stay visible is so someone can see what you
     said before you knew the answer.
     ───────────────────────────────────────────────────────────────────────── -->
