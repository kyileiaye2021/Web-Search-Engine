import json, os, pickle
from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

"""
Offline PageRank and HITS computation.
PageRank formula: PR(A) = (1-d) + d * sum(PR(Ti)/C(Ti)), d=0.85
HITS: h(x) <- sum a(y), a(x) <- sum h(y), 5 iterations + scaling
"""

ROOT_DIR = "DEV"
MAPPING_FILE = "doc_mapping.pkl"
PAGERANK_FILE = "pagerank.pkl"
HITS_FILE = "hits.pkl"
DAMPING = 0.85
ITERATIONS = 50

def build_link_graph():
    """
    Parse all JSON files, extract outlinks from <a href>, 
    build doc_id-based adjacency: outlinks[doc_id] = set of target doc_ids
    """
    with open(MAPPING_FILE, 'rb') as f:
        doc_id_to_url = pickle.load(f)
    url_to_doc_id = {v: k for k, v in doc_id_to_url.items()}

    outlinks = defaultdict(set)   # doc_id -> set of doc_ids it links to
    inlinks  = defaultdict(set)   # doc_id -> set of doc_ids that link to it

    total = len(doc_id_to_url)
    processed = 0

    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                url = data['url']
                if '#' in url:
                    url = url.split('#')[0]
                if url not in url_to_doc_id:
                    continue
                src_id = url_to_doc_id[url]

                soup = BeautifulSoup(data['content'], 'lxml')
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if '#' in href:
                        href = href.split('#')[0]
                    if href in url_to_doc_id:
                        dst_id = url_to_doc_id[href]
                        if dst_id != src_id:
                            outlinks[src_id].add(dst_id)
                            inlinks[dst_id].add(src_id)
            except:
                continue

            processed += 1
            if processed % 5000 == 0:
                print(f"  Link graph: {processed}/{total} docs processed...")

    return doc_id_to_url, outlinks, inlinks

def compute_pagerank(doc_id_to_url, outlinks, inlinks):
    """
    PR(i) = (1-d) + d * sum_{j->i} PR(j) / C(j)
    Iterates until convergence (max ITERATIONS).
    """
    N = len(doc_id_to_url)
    all_ids = list(doc_id_to_url.keys())

    pr = {doc_id: 1.0 for doc_id in all_ids}

    for iteration in range(ITERATIONS):
        new_pr = {}
        for i in all_ids:
            in_sum = 0.0
            for j in inlinks[i]:
                c_j = len(outlinks[j])
                if c_j > 0:
                    in_sum += pr[j] / c_j
                # dead-end: contributes 0 (simple version)
            new_pr[i] = (1 - DAMPING) + DAMPING * in_sum

        # Check convergence
        diff = sum(abs(new_pr[i] - pr[i]) for i in all_ids)
        pr = new_pr
        if iteration % 10 == 0:
            print(f"  PageRank iter {iteration}, diff={diff:.4f}")
        if diff < 1e-6 * N:
            print(f"  PageRank converged at iteration {iteration}")
            break

    return pr

def compute_hits(doc_id_to_url, outlinks, inlinks):
    """
    HITS: 5 iterations of h/a updates with scaling.
    h(x) <- sum_{x->y} a(y)
    a(x) <- sum_{y->x} h(y)
    Scale after each iteration.
    """
    all_ids = list(doc_id_to_url.keys())

    # Initialize h=1, a=1 for all pages (as per lecture)
    h = {doc_id: 1.0 for doc_id in all_ids}
    a = {doc_id: 1.0 for doc_id in all_ids}

    for iteration in range(5):
        # Update hub: h(x) = sum of a(y) for all y that x points to
        new_h = {}
        for x in all_ids:
            new_h[x] = sum(a[y] for y in outlinks[x]) if outlinks[x] else 0.0

        # Update authority: a(x) = sum of h(y) for all y that point to x
        new_a = {}
        for x in all_ids:
            new_a[x] = sum(new_h[y] for y in inlinks[x]) if inlinks[x] else 0.0

        # Scale (lecture: scaling factor doesn't matter, only relative values)
        h_max = max(new_h.values()) or 1.0
        a_max = max(new_a.values()) or 1.0
        h = {x: v / h_max for x, v in new_h.items()}
        a = {x: v / a_max for x, v in new_a.items()}

        print(f"  HITS iter {iteration+1} done")

    return h, a

def main():
    print("Building link graph...")
    doc_id_to_url, outlinks, inlinks = build_link_graph()
    print(f"Link graph built: {len(doc_id_to_url)} nodes")

    print("\nComputing PageRank...")
    pr = compute_pagerank(doc_id_to_url, outlinks, inlinks)
    with open(PAGERANK_FILE, 'wb') as f:
        pickle.dump(pr, f)
    print(f"PageRank saved to {PAGERANK_FILE}")

    print("\nComputing HITS...")
    h, a = compute_hits(doc_id_to_url, outlinks, inlinks)
    with open(HITS_FILE, 'wb') as f:
        pickle.dump({"hub": h, "authority": a}, f)
    print(f"HITS saved to {HITS_FILE}")

if __name__ == "__main__":
    main()