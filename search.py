import pickle
import math
import time
from collections import defaultdict
from indexer import preprocess_text
from decode import decode

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
        doc_url_mapping = pickle.load(f)
        return doc_url_mapping
    
def preprocess_query(search_query):
    """
    Apply the same rule of preprocessing search query as the documents

    Args:
        search_query (str): user search query

    Returns:
        a list of tokenized and stemmed tokens
    """
    tokens = preprocess_text(search_query)
    return tokens
    
def read_postings(token, posting_byte_pos):
    """
    Look up the term in posting byte pos and retrieve the posting docs in O(1) time

    Args:
        token(str): a token or term from a list of user query tokens
        posting_byte_pos (dict(tuple)): a dictionary of term and its corresponding byte position and len of postings

    Returns:
        list(Postings): a list of postings
    """
    if token not in posting_byte_pos:
        return []
    
    offset, length = posting_byte_pos[token]
    with open(MERGED_INDEX, "rb") as f:
        f.seek(offset)
        byte_raw_data = f.read(length)
        posting_data = decode(byte_raw_data)
        
    return posting_data

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

def search_query(query_tokens, posting_byte_pos, doc_mapping, top_k=5):
    """
    Search and rank documents by TF-IDF
    """
    
    total_docs = len(doc_mapping)

    #Get all token postings
    all_postings = []
    for token in query_tokens:
        postings = read_postings(token, posting_byte_pos)
        if not postings:
            return [] #means not find any word
        all_postings.append(postings)

    #Find documents that contain all query tokens.
    if len(all_postings) == 0:
        return []
    
    #start with shortest posting list
    all_postings.sort(key=len)
    result = all_postings[0]
    for i in range(1, len(all_postings)):
        result = intersect(result, all_postings[i])
        if not result:
            return []
        
    # Calculate TF-IDF scores, only for documents in the AND result
    valid_doc_ids = set(p[0] for p in result)
    scores = defaultdict(float)
    
    for token in query_tokens:
        postings = read_postings(token, posting_byte_pos)
        df = len(postings)
        
        for doc_id, tf, is_important in postings:
            if doc_id not in valid_doc_ids:
                continue
                
            if tf > 0 and df > 0:
                idf = math.log(total_docs / df)
                tfidf = (1 + math.log(tf)) * idf
                
                #import will earn 1.5x weight
                if is_important:
                    tfidf *= 1.5
                
                scores[doc_id] += tfidf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in ranked[:top_k]:
        url = doc_mapping.get(doc_id, "Unknown URL")
        results.append((url, score))

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
            print("not fing any result")
        else:
            print(f"\nSearch Results (Top 5):")
            for i, (url, score) in enumerate(results, 1):
                print(f"{i}. {url}")
                print(f"   Score: {score:.4f}")

if __name__ == "__main__":
    main()