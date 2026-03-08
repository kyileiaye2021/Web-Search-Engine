import json
from posting import Posting
import os
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import pickle
import re
import warnings
from encode import encode
import hashlib

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

CHUNK_SIZE = 14000
CHUNK_DIR = "index_chunks" # directory to store chunks of index 
MERGED_INDEX_FILE = "merged_index.bin" # to store posting bytes
MAPPING_FILE = "doc_mapping.pkl"
BYTE_POSITION_OFFSET_FILE = "byte_position.pkl"

#SimHash for near (extra credit)
def compute_simhash(tokens, hash_bits=64):
    if not tokens:
        return 0
    
    v = [0] * hash_bits
    
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(hash_bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint

def hamming_distance(hash1, hash2):
    xor = hash1 ^ hash2
    distance = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance

def is_near_duplicate(new_hash, existing_hashes, threshold=3):
    for existing_hash in existing_hashes:
        if hamming_distance(new_hash, existing_hash) <= threshold:
            return True
    return False

# N-gram for extra credit
def generate_ngrams(tokens, n=2):
    if len(tokens) < n:
        return []
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = "_".join(tokens[i:i+n])
        ngrams.append(ngram)
    
    return ngrams
    

def preprocess_text(content):
    """
    Tokenize the page content and apply stemming
    
    Args:
        content (str): contents of the url page from json file
    Return:
        a list of tokenized and stemmed tokens
    """
    tokens = re.compile(r"[A-Za-z0-9]+").findall(content.lower())
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def parse_url_content(content):
    """
    Parse HTML content
    - Extract all text
    - Distinguish important text
    - Anchor text for extra credit

    Args:
        content (str): contents of the url page from json file
    Return:
    
    """
    IMPORTANT_TAGS = ["title", "h1", "h2", "h3", "b", "strong"]
    try:
        soup = BeautifulSoup(content, "lxml")
    except:
        try:
            soup = BeautifulSoup(content, "html.parser")
        except:
            return [], set(), {}
        
    important_tokens = set()
    all_tokens = []
    anchor_texts = {}  # EC: Anchor text
    
    # Remove script and style
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()

    # for important text
    for tag in IMPORTANT_TAGS:
        for ele in soup.find_all(tag):
            tokens = preprocess_text(ele.get_text())
            important_tokens.update(tokens)
            all_tokens.extend(tokens)

    # EC: Extract anchor text
    for a_tag in soup.find_all('a', href=True):
        href = a_tag.get('href', '')
        anchor_text = a_tag.get_text().strip()
        
        if anchor_text and href and href.startswith('http'):
            tokens = preprocess_text(anchor_text)
            if tokens:
                if href not in anchor_texts:
                    anchor_texts[href] = []
                anchor_texts[href].extend(tokens)
    
    # for body text
    body_text = soup.get_text(separator=" ")
    tokens = preprocess_text(body_text)
    all_tokens.extend(tokens)

    return all_tokens, important_tokens, anchor_texts

def build_index(doc_id, all_tokens, important_tokens, CHUNK_INDEX):
    """
    Creating a map between all tokens to postings {token (str): posting(obj)}
    contain 2-gram/3-gram for extra credit

    Args:
        doc_id (int): Document id
        tf (int): frequency count of the token appeared in the doc
        important_tokens (set): tokens considered as important because of the occurence in headings/titles
    """
    token_frequency = Counter(all_tokens)
    for token, tf in token_frequency.items():
        is_important = token in important_tokens
        posting = Posting(doc_id, tf, is_important)
        CHUNK_INDEX[token].append(posting)

    # EC: Add 2-gram index
    bigrams = generate_ngrams(all_tokens, 2)
    bigram_freq = Counter(bigrams)
    for ngram, tf in bigram_freq.items():
        posting = Posting(doc_id, tf, False)
        CHUNK_INDEX[ngram].append(posting)

    # EC: Add 3-gram index
    trigrams = generate_ngrams(all_tokens, 3)
    trigram_freq = Counter(trigrams)
    for ngram, tf in trigram_freq.items():
        posting = Posting(doc_id, tf, False)
        CHUNK_INDEX[ngram].append(posting)
 
def build_anchor_index(anchor_texts, url_to_doc_id, CHUNK_INDEX):
    """
    Build anchor text index for extra credit
    """
    for target_url, words in anchor_texts.items():
        if '#' in target_url:
            target_url = target_url.split('#')[0]
        
        if target_url in url_to_doc_id:
            target_doc_id = url_to_doc_id[target_url]
            
            word_freq = Counter(words)
            for word, tf in word_freq.items():
                posting = Posting(target_doc_id, tf, True)
                CHUNK_INDEX[word].append(posting)


def save_chunk(CHUNK_INDEX, CHUNK_ID):
    """
    Saving the files in chunks.
    CHUNK SIZE - 14000
    Every 14000 docs, there is one chunk saved.
    
    Args:
        CHUNK_INDEX - temporaray in-memory inverted index
        CHUNK_ID - chunk file number
    
    Return:
        None
    """
    filename = os.path.join(CHUNK_DIR,f"partial_{CHUNK_ID}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(CHUNK_INDEX, f)
        
def merge_chunks():
    """
    Combine and merge all chunk index into final merged index
    Store the byte positions of each term and len of posting list for the term in BYTE_POSITION_OFFSET.pkl file 
    
    For each term and posting lists
    - encode them to bytes to shrink the index size and access the query term in O(1) time
    - add the bytes of posting lists to MERGED_INDEX_FILE
    - keep track of the offset (start of the query term) and len of posting bytes (to keep track of where the postings end for that term)
    """
    final_index = defaultdict(list) # to store all index stored across the chunks
    byte_position = defaultdict(tuple) # to store the byte postions of each term to know where to start looking for the term in pkl file
    offset = 0
    
    #building final index
    for file in sorted(os.listdir(CHUNK_DIR)):
        if not file.startswith("partial_") or not file.endswith('.pkl'):
            continue
        
        file_path = os.path.join(CHUNK_DIR, file)
        with open(file_path, 'rb') as f:
            partial_index = pickle.load(f)
    
        for token, postings in partial_index.items():
            final_index[token].extend(postings)
        
        #delete temp doc
        os.remove(file_path)

    # writing the final data to index file and meta data (offset) file
    with open(MERGED_INDEX_FILE, "wb") as f:
        for term in sorted(final_index):
            postings = final_index[term]
            postings.sort(key=lambda p: p.doc_id)  # sort the postings by doc id
            posting_bytes = encode(postings) # encode postings for each term
            byte_position[term] = (offset, len(posting_bytes)) 
            f.write(posting_bytes) # write posting bytes
            offset += len(posting_bytes)
    
    with open(BYTE_POSITION_OFFSET_FILE, "wb") as f:
        pickle.dump(byte_position, f)
    
def read_json():
    """
    Read the json files from DEV folder/sub folders
    Check if the page is duplicated
    Extract content from the files
    Preprocess the text and important tokens
    Build inverted index in chunks
    Save the chunks and merge
    
    Args: None
    Return: None
    """
    ROOT_DIR = "DEV"
    DOC_ID = 0
    CHUNK_ID = 0
    CHUNK_INDEX = defaultdict(list)
    DOC_ID_TO_URL = {}
    URL_TO_DOC_ID = {}  # EC: for anchor text
    URL_SEEN = set()
    SIMHASH_SET = set()  # EC: for near duplication
    ALL_ANCHOR_TEXTS = {}  # EC: for anchor texts

    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    exact_dup_count = 0
    near_dup_count = 0
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        url = data['url']
                        content = data['content']
                        
                        # URL duplication for extra credit
                        if '#' in url:
                            url = url.split('#')[0]
                        if url in URL_SEEN:
                            exact_dup_count += 1
                            continue
                        URL_SEEN.add(url)

                        # parse contents
                        all_tokens, important_tokens, anchor_texts = parse_url_content(content)
                        
                        # EC: SimHash
                        if all_tokens:
                            page_hash = compute_simhash(all_tokens)
                            if is_near_duplicate(page_hash, SIMHASH_SET):
                                near_dup_count += 1
                                continue
                            SIMHASH_SET.add(page_hash)

                        DOC_ID_TO_URL[DOC_ID] = url
                        URL_TO_DOC_ID[url] = DOC_ID
                        
                        # EC: collect anchor texts
                        for target_url, words in anchor_texts.items():
                            if target_url not in ALL_ANCHOR_TEXTS:
                                ALL_ANCHOR_TEXTS[target_url] = []
                            ALL_ANCHOR_TEXTS[target_url].extend(words)

                        build_index(DOC_ID, all_tokens, important_tokens, CHUNK_INDEX)

                        DOC_ID += 1
                        
                        if DOC_ID % 1000 == 0:
                            print(f"Processed {DOC_ID} documents...")

                        if DOC_ID % CHUNK_SIZE == 0:
                            save_chunk(CHUNK_INDEX, CHUNK_ID)
                            CHUNK_INDEX.clear()
                            CHUNK_ID += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    pass
    
    # EC: build anchor text index
    print("Building anchor text index...")
    build_anchor_index(ALL_ANCHOR_TEXTS, URL_TO_DOC_ID, CHUNK_INDEX)
    
    if CHUNK_INDEX:
        save_chunk(CHUNK_INDEX, CHUNK_ID)
        
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(DOC_ID_TO_URL, f)

    print(f"\nExact duplicates removed: {exact_dup_count}")
    print(f"Near duplicates removed: {near_dup_count}")

    return DOC_ID
   
def compute_analytics(total_doc):
    """
    Generating for report:
    - Num of indexed docs
    - Num of unique tokens
    - Total size of the index
    """
    with open(BYTE_POSITION_OFFSET_FILE, 'rb') as f:
        byte_position = pickle.load(f)
        
    num_of_unique_tokens = len(byte_position)
    size_bytes = os.path.getsize(MERGED_INDEX_FILE)
    size_kb = size_bytes / 1024
    
    print(f"Num of indexed documents: {total_doc}")
    print(f"Num of unique tokens: {num_of_unique_tokens}")
    print(f"Total size (bytes): {size_bytes}")
    print(f"Total size (KB) of the index: {size_kb:.4f}")
    
def main():
    total_doc = read_json()
    merge_chunks()
    compute_analytics(total_doc)
if __name__ == "__main__":
    main()