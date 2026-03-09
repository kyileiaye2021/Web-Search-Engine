import pickle
import math
import time
from collections import defaultdict
from indexer import preprocess_text, generate_ngrams
from decode import decode
import heapq

BYTE_POSITION_OFFSET_FILE = "byte_position.pkl" 
MAPPING_FILE = "doc_mapping.pkl"
MERGED_INDEX = "merged_index.bin"

def load_byte_pos_offset_file():
    """
    Loading byte position offset data file to use for locating the byte position of postings
    
    Returns:
        a dictionary of term and its corresponding byte position and len of postings
    """
    with open(BYTE_POSITION_OFFSET_FILE, 'rb') as f:
        posting_byte_position = pickle.load(f)
        return posting_byte_position

def load_doc_mapping_file():
    """
    Loading load doc mapping data file to use for retrieving urls based on retrieved doc ID
    
    Returns:
        a dictionary of doc id and its corresponding url
    """
    with open(MAPPING_FILE, 'rb') as f:
        return pickle.load(f)
    
def preprocess_query(search_query):
    """
    Apply the same rule of preprocessing search query as the documents

    Args:
        search_query (str): user search query

    Returns:
        a list of tokenized and stemmed tokens
    """
    return preprocess_text(search_query)
    
# def read_postings(token, posting_byte_pos):
#     """
#     Look up the term in posting byte pos and retrieve the posting docs in O(1) time

#     Args:
#         token(str): a token or term from a list of user query tokens
#         posting_byte_pos (dict(tuple)): a dictionary of term and its corresponding byte position and len of postings

#     Returns:
#         list(Postings): a list of postings
#     """
#     if token not in posting_byte_pos:
#         return []
    
#     offset, length = posting_byte_pos[token]
#     with open(MERGED_INDEX, "rb") as f:
#         f.seek(offset)
#         byte_raw_data = f.read(length)
#         posting_data = decode(byte_raw_data)
        
#     return posting_data

def intersect(p1, p2):
    """
    Applying boolean AND to retrieve the docs that have both terms

    Args:
        p1 (list(Postings)): a posting of first term
        p2 (list(Postings)): a posting of second term

    Returns:
        _type_: _description_
    """
    i = j = 0
    result = []
    while i < len(p1) and j < len(p2):
        if p1[i][0] == p2[j][0]:
            result.append(p1[i]) #not for p1[0]
            i += 1
            j += 1
            
        elif p1[i][0] < p2[j][0]:
            i += 1
        
        else:
            j += 1
    return result

def search_with_or(query_tokens, posting_byte_pos, doc_mapping, top_k=5):
    """
    Perform an OR-based ranked search over the inverted index using TF-IDF scoring,
    with additional support for 2-gram (bigram) matching and importance boosting.

    This function retrieves documents that contain at least one of the query terms.
    It expands the query by generating bigrams from the original query tokens to
    capture phrase-level relevance. Both unigrams and 2-grams/3-grams are used to search
    the inverted index.

    The ranking score for each document is computed using a modified TF-IDF formula:
        score = (1 + log(tf)) * idf

    Additional scoring adjustments:
    - Bigram boost: Bigram matches are weighted more heavily than unigram matches to reward phrase-level relevance.
    - Important text boost: Terms appearing in important sections of a document (e.g., title, headings, or emphasized content) receive additional weight.
    - Match ratio boost: Documents matching a higher proportion of the query terms receive an extra multiplier to reward broader query coverage.

    The function reads postings lists directly from the merged inverted index file using byte offsets stored in `posting_byte_pos`, which enables efficient disk
    access without loading the entire index into memory.

    Args:
        query_tokens (List[str]):
            Preprocessed query tokens (lowercased, tokenized, and stemmed).

        posting_byte_pos (Dict[str, Tuple[int, int]]):
            Dictionary mapping each term to a tuple of
            (byte_offset, byte_length) in the merged index file.
            This allows direct retrieval of the postings list for each term.

        doc_mapping (Dict[int, str]):
            Mapping from internal document IDs to their corresponding URLs.

        top_k (int, optional):
            Number of top-ranked results to return. Default is 5.

    Returns:
        List[Dict[str, Any]]:
            A list of ranked search results, each containing:
            - "url": The document URL.
            - "score": The computed relevance score.

            Example:
            [
                {"url": "https://example.com/page1", "score": 8.42},
                {"url": "https://example.com/page2", "score": 7.91}
            ]
    """
    total_docs = len(doc_mapping)
    scores = defaultdict(float)
    doc_term_count = defaultdict(int)
    
    # EC: search 2-gram
    bigrams = generate_ngrams(query_tokens, 2)
    all_search_terms = query_tokens + bigrams
    
    with open(MERGED_INDEX, 'rb') as f:
        for token in all_search_terms:
            if token not in posting_byte_pos:
                continue
            
            offset, length = posting_byte_pos[token]
            f.seek(offset)
            byte_raw_data = f.read(length)
            postings = decode(byte_raw_data)
            
            df = len(postings)
            if df == 0:
                continue
            
            idf = math.log(total_docs / df)
            
            # EC: n-gram
            is_ngram = "_" in token
            ngram_boost = 2.0 if is_ngram else 1.0
            
            for doc_id, tf, is_important in postings:
                if tf > 0:
                    tfidf = (1 + math.log(tf)) * idf * ngram_boost
                    if is_important:
                        tfidf *= 2.0
                    scores[doc_id] += tfidf
                    doc_term_count[doc_id] += 1
    
    for doc_id in scores:
        match_ratio = doc_term_count[doc_id] / len(query_tokens)
        scores[doc_id] *= (1 + match_ratio)
    
    if not scores:
        return []
    
    ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    results = []
    for doc_id, score in ranked:
        url = doc_mapping.get(doc_id, "Unknown URL")
        results.append({"url": url, "score": score})
    
    return results

def search_query(query_tokens, posting_byte_pos, doc_mapping, top_k=5):
    """
    Execute a ranked search for a user query using a Boolean AND retrieval model
    combined with TF-IDF scoring.

    The function retrieves postings lists for each query token using byte offsets
    stored in `posting_byte_pos`, which allows efficient access to the inverted
    index without loading the entire index into memory.

    Search workflow:
    1. Retrieve postings lists for all query tokens from the merged inverted index.
    2. Perform Boolean AND intersection across postings lists to find documents
       that contain all query terms.
    3. If the AND result is empty, fall back to an OR-based search using
       `search_with_or()` to still return relevant results.
    4. Rank documents using TF-IDF scoring:
           tfidf = (1 + log(tf)) * idf
       where:
           tf = term frequency in the document
           idf = log(total_docs / document_frequency)
    5. Apply additional ranking boosts:
       - Terms appearing in important sections of the document receive extra weight.
       - Bigram matches (2-gram phrases from the query) receive additional boosting
         to reward phrase-level relevance.
    6. Return the top-k ranked documents.

    Args:
        query_tokens (List[str]):
            Preprocessed query tokens (lowercased, tokenized, and stemmed).

        posting_byte_pos (Dict[str, Tuple[int, int]]):
            Dictionary mapping each term to a tuple of
            (byte_offset, byte_length) in the merged inverted index file.
            This allows direct retrieval of postings lists from disk.

        doc_mapping (Dict[int, str]):
            Mapping from internal document IDs to their corresponding URLs.

        top_k (int, optional):
            Number of top-ranked search results to return. Default is 5.

    Returns:
        List[Dict[str, Any]]:
            A list of ranked search results containing:
            - "url": the document URL
            - "score": the computed TF-IDF relevance score

            Example:
            [
                {"url": "https://example.com/page1", "score": 9.21},
                {"url": "https://example.com/page2", "score": 8.73}
            ]
    """
    
    total_docs = len(doc_mapping)

    #Get all token postings
    all_postings = []
    
    # Open stored index file
    with open(MERGED_INDEX, 'rb') as f:
        # read postings once per token
        for token in query_tokens:
            # postings = read_postings(token, posting_byte_pos)
            if token not in posting_byte_pos:
                return []
    
            offset, length = posting_byte_pos[token]
            f.seek(offset)
            byte_raw_data = f.read(length)
            postings = decode(byte_raw_data)
                
            if not postings:
                return [] #means not find any word
            all_postings.append(postings)

        #Find documents that contain all query tokens.
        if len(all_postings) == 0:
            return []
    
    #start with shortest posting list to do Boolean Retrieval model in O(m + n) time
    all_postings.sort(key=len)
    result = all_postings[0]
    for i in range(1, len(all_postings)):
        result = intersect(result, all_postings[i])
        #return with using OR search
        if not result:
            return search_with_or(query_tokens, posting_byte_pos, doc_mapping, top_k)
        
    # Calculate TF-IDF scores, only for documents in the AND result
    valid_doc_ids = set(p[0] for p in result)
    scores = defaultdict(float)
    
    for postings in all_postings:
        df = len(postings)
        
        if df == 0:
            continue
        
        idf = math.log(total_docs / df)
            
        for doc_id, tf, is_important in postings:
            if doc_id not in valid_doc_ids:
                continue
                
            if tf > 0 and idf > 0:
                tfidf = (1 + math.log(tf)) * idf
                
                #import will earn 2x weight
                if is_important:
                    tfidf *= 2.0
                
                scores[doc_id] += tfidf

    # EC: 2-gram
    bigrams = generate_ngrams(query_tokens, 2)
    with open(MERGED_INDEX, 'rb') as f:
        for bigram in bigrams:
            if bigram not in posting_byte_pos:
                continue
            
            offset, length = posting_byte_pos[bigram]
            f.seek(offset)
            byte_raw_data = f.read(length)
            postings = decode(byte_raw_data)
            
            df = len(postings)
            if df == 0:
                continue
            
            idf = math.log(total_docs / df)
            
            for doc_id, tf, is_important in postings:
                if doc_id in valid_doc_ids and tf > 0:
                    tfidf = (1 + math.log(tf)) * idf * 3.0
                    scores[doc_id] += tfidf

    # ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])
    results = []
    for doc_id, score in ranked:
        url = doc_mapping.get(doc_id, "Unknown URL")
        # results.append((url, score))
        results.append({
            "url": url,
            "score": score
        })

    return results

def main():
    posting_byte_pos = load_byte_pos_offset_file()
    doc_mapping = load_doc_mapping_file()

    while True:
        query = input("\nPlease enter your query:").strip()

        if query.lower() == 'quit':
            break

        if not query:
            continue

        start_time = time.time()
        query_tokens = preprocess_query(query)
        results = search_query(query_tokens, posting_byte_pos, doc_mapping, top_k=5)
        end_time = time.time()
        
        #check response time < 30ms
        response_time = (end_time - start_time) * 1000
        print(f"Response time: {response_time:.2f} ms")

        if not results:
            print("not finding any result")
        else:
            print(f"\nSearch Results (Top 5):")
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['url']}")
                print(f"   Score: {res['score']:.4f}")

if __name__ == "__main__":
    main()