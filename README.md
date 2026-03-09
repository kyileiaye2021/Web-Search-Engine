# UCI Search Engine

A lightweight end-to-end search engine for the UCI web corpus, with a Flask UI, disk-based inverted index, Boolean + ranked retrieval, and relevance boosting.

## 1) Problem We Are Solving

Given a large set of crawled UCI webpages (stored as JSON with URL + HTML content), we need to:

- Build a searchable index efficiently without exhausting memory.
- Return relevant results quickly for user queries.
- Improve ranking quality beyond plain keyword matching.
- Reduce noise from duplicate or near-duplicate pages.

In short: **turn raw web pages into a fast, relevance-aware local search experience**.

## 2) How the Problem Is Approached

The repository is split into two main workflows:

1. **Offline indexing pipeline** (`indexer.py`)
2. **Online query + serving pipeline** (`search.py` + `app.py`)

### A. Offline indexing pipeline

- Reads crawled JSON documents from `DEV/`.
- Parses HTML content with BeautifulSoup.
- Tokenizes and stems text (Porter stemmer).
- Tracks “important” terms from high-signal tags (`title`, `h1/h2/h3`, `b`, `strong`).
- Builds an inverted index in chunks (to control memory usage).
- Adds extra features:
  - Near-duplicate filtering with SimHash + Hamming distance.
  - N-gram indexing (bigrams and trigrams).
  - Anchor-text contribution to target pages.
- Merges partial chunks into a single binary postings file.
- Stores a byte-offset lexicon for O(1)-style term lookup into the binary index.

### B. Online query/serving pipeline

- User submits query from web UI or CLI.
- Query is preprocessed with the same tokenizer/stemmer as indexing.
- Retrieval strategy:
  - Try **Boolean AND** intersection first (precision-oriented).
  - If empty, fallback to **OR retrieval** (recall-oriented).
- Ranking uses TF-IDF with additional boosts:
  - Important-term boost.
  - N-gram phrase boost.
  - Query-term coverage/match-ratio boost (in OR mode).
- Top-k results are returned with URL + score through `/api/search`.

## 3) Tech Stack and Techniques Used

### Languages & Frameworks

- **Python** for indexing, ranking, and serving.
- **Flask** for web server and JSON API.
- **HTML/CSS + vanilla JS** for the frontend query page.

### Parsing & NLP

- **BeautifulSoup** (`lxml` or `html.parser`) for HTML parsing.
- **Regex tokenization** (`[A-Za-z0-9]+`).
- **NLTK PorterStemmer** for normalization.

### Information Retrieval / Ranking

- **Inverted index** with postings `(doc_id, tf, is_important)`.
- **TF-IDF scoring**: `(1 + log(tf)) * idf`.
- **Boolean AND intersection** with postings-list merge in linear time.
- **Fallback OR retrieval** for robustness.
- **N-gram (2/3-gram) indexing and query boosting** for phrase-like relevance.

### Storage & Efficiency

- **Chunked indexing** to avoid huge in-memory structures.
- **Binary postings encoding** via `struct.pack("IIB")`.
- **Byte-offset dictionary** (`byte_position.pkl`) to seek directly to term postings.
- **Pickle** for metadata persistence (URL mapping and offsets).

### Quality Improvements

- **Exact duplicate suppression** via URL normalization (`#` fragment removal).
- **Near-duplicate suppression** via SimHash + Hamming threshold.
- **Anchor text indexing** to transfer link text signal to target documents.

## 4) How These Choices Solve the Problem

- **Scalability for indexing**: chunking + merge prevents memory blowups.
- **Fast query-time access**: byte-offset seeks avoid scanning full index.
- **Better ranking quality**: TF-IDF + important tags + n-gram boosts improve relevance.
- **Robust retrieval behavior**: AND-first gives precise results; OR fallback avoids dead ends.
- **Cleaner corpus**: duplicate filtering reduces redundant/noisy results.
- **Practical usability**: simple web API/UI gives immediate interactive search.

## 5) Future Directions

Potential improvements, prioritized for impact:

1. **Compression upgrades**
   - Delta encoding + variable-byte/gamma coding for smaller postings and faster I/O.

2. **Better ranking models**
   - Move from TF-IDF toward BM25.
   - Add field-aware weighting (title > headings > body > anchor).

3. **Index architecture enhancements**
   - Maintain a true lexicon file (term -> df, offset, length).
   - Store document lengths for normalized scoring.
   - Add incremental indexing instead of full rebuilds.

4. **Query understanding**
   - Spelling correction and query suggestions.
   - Synonym expansion and optional semantic retrieval.

5. **Serving and UX improvements**
   - Snippet generation with highlighted matches.
   - Pagination and caching for common queries.
   - Better frontend filtering/sorting controls.

6. **Evaluation and observability**
   - Add IR evaluation metrics (Precision@k, Recall@k, nDCG).
   - Build benchmark scripts and profiling dashboards for latency/throughput.

## 6) Repository Map

- `indexer.py` — builds chunked inverted index, deduplication, n-grams, anchor indexing, and merge.
- `search.py` — query preprocessing, Boolean/OR retrieval, TF-IDF ranking.
- `app.py` — Flask app and `/api/search` endpoint.
- `encode.py` / `decode.py` — binary serialization/deserialization of postings.
- `posting.py` — posting object shape.
- `templates/index.html`, `static/style.css` — frontend UI.

## 7) Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build index artifacts (expects crawled data under `DEV/`):

```bash
python indexer.py
```

3. Start search web app:

```bash
python app.py
```

4. Open:

- `http://localhost:8000`

---

If you want, the next iteration of this README can also include a short architecture diagram and an example query trace (query -> tokens -> postings -> scores -> top-k).
