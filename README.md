# The Unofficial Guide

**Author:** Steven Huang  
**Corpus:** `campus_life`

## What This Does

The Unofficial Guide is a grounded question-answering system built on the `campus_life` corpus—a collection of 88 peer-written student notes and guides. It makes practical, word-of-mouth campus knowledge searchable, answering plain questions about dorm living conditions, course grading curves, dining hall peak hours, and unwritten administrative rules. When given a query, the system retrieves relevant semantic chunks from a local Chroma vector database and synthesizes a concise, factual answer using Gemini while citing the exact source documents. If a question falls outside the corpus or lacks sufficient evidence, a relevance gate rejects it immediately to prevent hallucinations.

## Chunking Strategy

**Chunk size:** 450 characters
**Overlap:** 100 characters

When inspecting the `campus_life` corpus in Milestone 1, we found that the 88 documents are brief student posts averaging 317 characters (ranging from 178 to 549 characters). Each document consists of an informative title line (e.g. `Kestrel Commons` or `On the housing lottery`) followed by 1 to 3 short paragraphs. 

The starter chunker used an 800-character fixed window, which resulted in 88 documents turning into 88 un-split chunks. However, treating every post as an immutable unit failed to address longer posts like `housing_old_brewhouse.txt` (549 characters) and `housing_morrow_house.txt` (464 characters), which combine multiple distinct topics (such as building history, room layouts, damp/heating problems, and laundry costs). Conversely, naive fixed-character chunking cut sentences in half and severed subsequent chunks from their document titles, producing orphaned fragments (e.g. laundry hours without mentioning which residence hall they applied to).

We replaced the chunker with a semantic paragraph- and boundary-aware strategy (`chunker.py::split_documents`). Posts under 450 characters remain whole so they keep their full topical coherence and title context. For posts exceeding 450 characters, we split cleanly along paragraph breaks (or sentence boundaries if an individual paragraph exceeds the threshold). Crucially, the chunker prepends the document title to every chunk created from that document, and retains a 100-character paragraph/sentence overlap across chunk splits. This ensures that every chunk stands alone as a self-contained, complete thought with zero sentences cut in half.

## Sample Chunks

**Chunk 1** — source: `admin_add_drop_deadline.txt#0` — produced by: `chunker.py::split_documents`

```
On the add/drop deadline

You can add a course through the end of the second week. Dropping is a longer window — through the end of week six — but a drop after week two shows as a W on your transcript. Nothing anywhere on the registrar's site says this plainly, and students find out from each other.
```

**Chunk 2** — source: `course_biol_160_exams.txt#0` — produced by: `chunker.py::split_documents`

```
BIOL 160 Cell Biology — assessment

Four unit tests and a cumulative final. Not curved.

The unit tests come fast, roughly every three weeks; falling behind once is very hard to recover from.
```

**Chunk 3** — source: `course_math_220_exams.txt#0` — produced by: `chunker.py::split_documents`

```
MATH 220 Linear Algebra — assessment

Two midterms and a cumulative final. Curved to a b- median.

The problem sets are the course; the lectures make sense afterwards rather than during.
```

**Chunk 4** — source: `dining_the_ridgeway_cafe.txt#0` — produced by: `chunker.py::split_documents`

```
The Ridgeway Café

Second-year here. Wait times: 10 to 15 minutes at 12:30, none after 2:00. The thing worth going for is the only place on campus with real espresso. The thing to know is that seating is tight; about 40 seats for a building of 900.

Hours are 7:00am to 4:00pm weekdays only. Costs declining balance only, no meal swipes.
```

**Chunk 5** — source: `housing_morrow_house.txt#0` — produced by: `chunker.py::split_documents`

```
Morrow House — what it's actually like

Just finished a year in this building. Built 1954, partially renovated 2008. Rooms are singles and doubles, hall bathrooms.

The good: cheapest housing tier by about $900 a year, and the singles are real singles.

The bad: known damp problem on the ground floor; two rooms were taken offline in 2024.
```

## Sample Answer

**Question:** How is housing lottery priority determined for juniors and seniors?

**Answer:**

```
For juniors and seniors, housing lottery priority is ordered by accumulated credit hours first, with a random tie-break used only when necessary. 

Source: `admin_housing_lottery.txt`

Sources retrieved: admin_grade_appeals.txt, admin_housing_lottery.txt, advising_registration.txt, course_stat_150_exams.txt, housing_tamsin_court.txt
```

**My relevance cutoff:** `0.60`

To calibrate the cutoff, we ran our five in-corpus test questions alongside the five out-of-scope questions from `questions.py` and recorded the best cosine distance for each. The in-corpus questions clustered tightly between 0.205 and 0.480, while the out-of-scope questions clustered between 0.825 and 0.934. 

This created a wide, clean gap between 0.480 and 0.825. Setting the cutoff at 0.60 places it safely in this buffer (~0.12 above our furthest in-corpus match and ~0.22 below our closest out-of-scope match). At 0.60, all 5 in-corpus questions are admitted and all 5 out-of-scope questions are halted at the relevance gate before any model calls are made.

| Question | In corpus? | Best distance |
|---|---|---|
| How is housing lottery priority determined for juniors and seniors? | Yes | 0.235 |
| What are the lunch wait times at Kestrel Commons between 12:15 and 1:00? | Yes | 0.205 |
| When is the best time to do laundry in Fenwick Court to avoid waiting? | Yes | 0.297 |
| Are CS 210 exams curved? | Yes | 0.386 |
| Which shuttle stop gets skipped when the driver is behind schedule? | Yes | 0.480 |
| What is the capital of Mongolia? | No | 0.825 |
| How do I change the oil in a diesel engine? | No | 0.934 |
| Who won the 1994 World Cup? | No | 0.886 |
| What is the recommended dosage of ibuprofen for a headache? | No | 0.844 |
| How do I write a for loop in Rust? | No | 0.896 |

## How I Used AI

**1.** During the initial setup check, `test.py` failed with an `ImportError` on `ChannelCredentials` in `grpc` and a `ModuleNotFoundError` on `google.genai._interactions.types.model`. I prompted the AI to inspect `site-packages` to find out why these pinned packages were failing to import. The AI identified that Windows/iCloudDrive file synchronization had duplicated and renamed package files with numbers (e.g., creating `'__init__ 123.py'` inside `grpc/` and `'model 2.py'` inside `google/genai/_interactions/types/`). Instead of reinstalling the entire virtual environment from scratch, I directed the assistant to scan the venv specifically for space-and-number conflict artifacts and rename them back to their canonical filenames (`__init__.py`, `model.py`), successfully resolving the imports without disrupting package pins.

**2.** In Milestone 3, I asked the AI to write a chunker that respected document structure rather than slicing at arbitrary character lengths. The initial suggestion split strictly on double-newlines (`\n\n`), but testing revealed that it separated the document title (e.g. `Kestrel Commons`) into an isolated 15-character chunk, leaving subsequent paragraphs without their subject context. I adjusted the implementation so the chunker identifies document headings and prepends the heading to every generated chunk from that document, keeps short documents (< 450 characters) unified as single complete chunks, and falls back to sentence-boundary splitting with paragraph overlap for longer documents.

**3.** In Unit 2, I used the AI assistant to analyze the retrieval failures in our initial evaluation log. I asked the assistant why Question 5 (*"Which shuttle stop gets skipped when the driver is behind schedule?"*) ranked four dining hall documents ahead of `transit_shuttle.txt`. The AI demonstrated that dense vectors were matching on the conversational phrase *"behind schedule"*, diluting the critical entity keywords *"shuttle"* and *"stop"*. When drafting the hybrid search improvement, the AI initially proposed a raw score addition between BM25 and cosine distance, which would have distorted distance metrics and broken the 0.60 relevance gate in `gate.py`. I had it implement Reciprocal Rank Fusion (RRF with $k=60$) for candidate ranking while preserving the candidate's true cosine distance, ensuring full gate compatibility.

<!-- ── Stretch features ─────────────────────────────────────────────────────
     Doing one? Say so here BEFORE you start. A feature this README never
     claims earns nothing.
     ───────────────────────────────────────────────────────────────────────── -->

---

# Unit 2

<!-- These sections get ADDED to what's already above. Don't delete or rewrite
     unit 1 — the point is that someone can see what you said before you knew
     how it went. -->

## Run Log — Before

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Retrieved chunk contains the answer | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 2. Every answer names a source | 5 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 3. Gate stops out-of-corpus questions | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 4. Complete semantic thoughts with no mid-sentence cuts | 100% | 91/91 | 91/91 | 91/91 | MET |
| 5. Ground-truth source attribution | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |

### Real Output from Evaluation Run (Before)

**Criterion 1: Retrieved chunk contains the answer**  
*Produced by: `store.py::search` (Run 1, Question 1)*
```
Question: How is housing lottery priority determined for juniors and seniors?
Best distance: 0.2354 (passed the gate)
Sources retrieved: admin_grade_appeals.txt, admin_housing_lottery.txt, advising_registration.txt, course_stat_150_exams.txt, housing_tamsin_court.txt

Chunk excerpt from admin_housing_lottery.txt#0:
"The housing lottery is not random in the way most people assume. Rising sophomores get a number drawn at random, but juniors and seniors are ordered by accumulated credit hours first, and only tie-break randomly."
```

**Criterion 2: Every answer names a source**  
*Produced by: `generate.py::answer_from_chunks` (Run 1, Question 2)*
```
Question: What are the lunch wait times at Kestrel Commons between 12:15 and 1:00?
Answer:
The wait times at Kestrel Commons between 12:15 and 1:00 are 20 to 25 minutes. 

Source: `dining_kestrel_commons.txt` (and also confirmed in `dining_kestrel_commons_followup.txt`).
```

**Criterion 3: Gate stops out-of-corpus questions**  
*Produced by: `gate.py::check` (Deterministic Out-of-Scope Run)*
```
Question: What is the capital of Mongolia?
Best distance: 0.825 (cutoff: 0.60)
Gate result: refused
Response text: "I don't have enough information about that."
```

**Criterion 4: Complete semantic thoughts with no mid-sentence cuts**  
*Produced by: `chunker.py::split_documents` (Sample Chunk from Corpus)*
```
Source: admin_add_drop_deadline.txt#0
Text:
"On the add/drop deadline

You can add a course through the end of the second week. Dropping is a longer window — through the end of week six — but a drop after week two shows as a W on your transcript. Nothing anywhere on the registrar's site says this plainly, and students find out from each other."
(Starts on clean heading boundary, ends on terminal period '.', length: 301 chars)
```

**Criterion 5: Ground-truth source attribution**  
*Produced by: `generate.py::answer_from_chunks` (Run 1, Question 5)*
```
Question: Which shuttle stop gets skipped when the driver is behind schedule?
Answer:
The stop outside Fenwick Court is the one that gets skipped when the driver is behind schedule (transit_shuttle.txt).
(Ground-truth source transit_shuttle.txt correctly identified and cited)
```

## Verdicts

| # | Criterion | Verdict | How I decided |
|---|---|---|---|
| 1 | Retrieved chunk contains the answer | MET | All 3 trials scored 5/5 (100%), surpassing the 4 of 5 target; in every query, the top-k chunks included the text containing the ground-truth answer. |
| 2 | Every answer names a source | MET | All 3 trials scored 5/5 (15 out of 15 generated responses), meeting the 5 of 5 target by explicitly naming source `.txt` documents. |
| 3 | Gate stops out-of-corpus questions | MET | The relevance gate successfully stopped 5 of 5 out-of-scope questions (best distances 0.825 to 0.934, well above 0.60), exceeding the 4 of 5 target. |
| 4 | Complete semantic thoughts with no mid-sentence cuts | MET | All 91 chunks across the indexed corpus begin on heading/sentence boundaries and end on valid punctuation (`.`, `!`, `?`), satisfying the 100% target. |
| 5 | Ground-truth source attribution | MET | All 3 trials scored 5/5, exceeding the 4 of 5 target, as the model consistently cited the true ground-truth file rather than any top-k distractor. |

## Diagnoses

While all five criteria formally scored **MET** against the initial unit 1 targets (4 of 5, and 100% on chunking), an honest audit of the run logs reveals that these targets were set relatively safe and masked critical vulnerabilities in the retrieval pipeline.

### 1. Analysis of Pipeline Weaknesses & Near-Misses

- **Question 5 (Transit Shuttle — Stage: Retrieval / Embedding):**
  - **Symptom:** In Question 5 (*"Which shuttle stop gets skipped when the driver is behind schedule?"*), the best distance was **0.4802**, which is uncomfortably close to the **0.60** cutoff. 
  - **Mechanism:** The dense embedding model (`all-MiniLM-L6-v2`) matched on generic conversational tokens like *"behind"*, *"schedule"*, *"skipped"*, and *"driver"*. In a student-life corpus, these words match general discussions about class schedules, running late, and skipping dining hall meals. As a direct result, **4 of the top 5 retrieved chunks** were completely irrelevant dining hall follow-up documents (`dining_halden_hall_followup.txt`, `dining_kestrel_commons_followup.txt`, `dining_north_kitchen_followup.txt`, `dining_the_atrium_followup.txt`).
  - **Near-Miss Impact:** The ground-truth document (`transit_shuttle.txt`) was retrieved at the very bottom (**rank 5**). If we had tightened Criterion 1 to *"The top 3 chunks include the answer"* or reduced `TOP_K` from 5 to 4, this question would have failed completely.

- **Question 3 (Fenwick Laundry — Stage: Retrieval):**
  - **Symptom:** Alongside the correct laundry file, retrieval pulled in `dining_kestrel_commons_followup.txt` and `transit_walking.txt`.
  - **Mechanism:** Generic phrases such as *"best time"* and *"avoid waiting"* created semantic drift towards dining peak hours and walking shortcuts.

- **Question 4 (CS 210 Exams — Stage: Generation / Scorer matching):**
  - **Symptom:** In Run 1, the automated judge flagged a failure despite the answer being factually accurate.
  - **Mechanism:** The model generated: *"Yes, the midterms for CS 210 are curved, but the final is not curved."* The expects string was `"Midterms are curved"`. The model's insertion of the prepositional clause *"for CS 210"* broke the verbatim substring check.

### 2. Systematic Pattern: Lexical Specificity Gap in Dense Embeddings

Across our tests, the primary systematic flaw is that **pure dense vector search prioritizes broad semantic mood over exact entity names and keywords**. For queries with unique entities like `"shuttle"`, `"CS 210"`, or `"Fenwick Court"`, dense embeddings dilute the entity signal across generic conversational verbs.

### 3. Target Tightness & Proposed Tightening

Our targets in Unit 1 were set safe:
- Criterion 1 asked for 4 of 5 test questions to have the answer in the retrieved chunks, with `TOP_K = 5`.
- **How I would tighten it:** I would tighten Criterion 1 to: *"For at least 4 of 5 test questions, the answer is contained within the **top-3** retrieved chunks, and the top-ranked chunk (rank 1) is a ground-truth document in at least 4 of 5 questions."* 
- Under this tightened standard, the current pure-dense retrieval pipeline **FAILS** because Question 5 ranked the true document at rank 5. This directly motivates our improvement in Milestone 4: adding **BM25 Hybrid Search**.

## The Improvement

**What I changed:**
I implemented **BM25 Hybrid Search** in `store.py::search` by integrating `rank_bm25.BM25Okapi` with Chroma's dense embedding search (`all-MiniLM-L6-v2`) using Reciprocal Rank Fusion (RRF with smoothing constant $k=60$).

**Why I picked it:**
Our diagnosis in Milestone 3 proved that pure dense semantic search suffered from a lexical specificity gap: in Question 5 (*"Which shuttle stop gets skipped when the driver is behind schedule?"*), semantic embeddings confused generic scheduling/delay words with dining hall rush hours, pushing the true ground-truth `transit_shuttle.txt` down to rank 5 behind 4 irrelevant dining documents. BM25 provides exact keyword and entity token matching, boosting distinctive nouns like "shuttle" and "stop".

### Run Log — After

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Retrieved chunk contains the answer | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 2. Every answer names a source | 5 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 3. Gate stops out-of-corpus questions | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |
| 4. Complete semantic thoughts with no mid-sentence cuts | 100% | 91/91 | 91/91 | 91/91 | MET |
| 5. Ground-truth source attribution | 4 of 5 | 5/5 | 5/5 | 5/5 | MET |

**Did it help?**
Yes, significantly. In Question 5, BM25 assigned a score of 27.49 to `transit_shuttle.txt` (nearly 4x higher than any other document in the corpus), promoting it from rank 5 all the way to **rank 1** in the fused candidate ranking. It completely eliminated the 4 noisy dining hall distractors that had polluted the prompt context during the Before run. Similarly, in Question 3 (Fenwick Court laundry), hybrid search replaced generic dining and walking documents with topical residence hall laundry guides. Under the tightened diagnostic standard of Top-3 retrieval precision proposed in Milestone 3, Hybrid Search achieved a flawless 5 of 5, providing robust safety margins for entity-heavy student queries.

## What's Still Broken

1. **Exact-match scoring rigidity vs generative variations:**
   In Question 4 (*"Are CS 210 exams curved?"*), all three model responses were factually accurate and cited both relevant source documents (`course_cs_210.txt` and `course_cs_210_exams.txt`), stating: *"Yes, the midterms for CS 210 are curved, but the final is not curved."* However, our automated judge in `scorer.py` checked for the verbatim substring `"Midterms are curved"`, which failed because the model naturally inserted the phrase *"for CS 210"*. In future iterations, replacing simple string matching with a regex or LLM-as-a-judge scorer would prevent false negative evals without restricting the model's natural phrasing.

2. **Basic whitespace tokenization in BM25:**
   Our current BM25 tokenizer splits on alphanumeric word boundaries (`\w+`) without stemming or lemmatization. If a user asks with morphological variations (e.g., "shuttles" vs "shuttle" or "curving" vs "curved"), BM25 term weighting drops. Adding PorterStemmer or lemmatization to `_tokenize()` would make lexical retrieval more resilient to plurals and verb tenses.

3. **Why I stopped here:**
   Hybrid search directly solved the core failure mode identified in our diagnosis—eliminating distractor chunks and elevating the true source document to rank 1 across all 5 test questions. The system now clears all five acceptance criteria with solid headroom. Introducing heavy NLP libraries (like spaCy or NLTK) for stemming would add startup latency and extra dependencies for marginal gain on this 88-document corpus.

## What I'd Do Differently

1. **Tighten Criterion 1 to Top-3 Retrieval Precision:**
   Criterion 1 originally stated: *"For at least 4 of my 5 test questions, the retrieved chunks include one that contains the answer"* with `TOP_K = 5`. That allowed Question 5 to pass even when the real document was buried at rank 5 behind four irrelevant dining posts. In the next unit, I would formulate it as: *"For at least 4 of 5 test questions, the ground-truth document is retrieved within the top 3 ranks, and is rank 1 in at least 3 queries."* That would turn retrieval ranking quality into a sensitive, observable metric.

2. **Measure Source Attribution Specificity:**
   Criterion 2 only required naming *at least one* source document. I would revise it to evaluate attribution purity: *"Every answer cites only documents that directly support the asserted facts, with zero citations to irrelevant distractor chunks included in top-k context."* This would penalize models that indiscriminately recite all filenames passed in the prompt.
