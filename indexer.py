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

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

CHUNK_SIZE = 14000
CHUNK_DIR = "index_chunks" # directory to store chunks of index 
MERGED_INDEX_FILE = "merged_index.bin" # to store posting bytes
MAPPING_FILE = "doc_mapping.pkl"
BYTE_POSITION_OFFSET_FILE = "byte_position.pkl"

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
            return [], set()
    important_tokens = set()
    all_tokens = []
    
    # Remove script and style
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()

    # for important text
    for tag in IMPORTANT_TAGS:
        for ele in soup.find_all(tag):
            tokens = preprocess_text(ele.get_text())
            important_tokens.update(tokens)
            all_tokens.extend(tokens)
    
    # for body text
    body_text = soup.get_text(separator=" ")
    tokens = preprocess_text(body_text)
    all_tokens.extend(tokens)
    
    return all_tokens, important_tokens

def build_index(doc_id, all_tokens, important_tokens, CHUNK_INDEX):
    """
    Creating a map between all tokens to postings {token (str): posting(obj)}

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
    URL_SEEN = set()

    os.makedirs(CHUNK_DIR, exist_ok=True)
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            
            if file.endswith(".json"):
                
                #DOC_ID += 1 reomove this 
                #because When skipping duplicate URLs, 
                #DOC_ID is also incremented, resulting in an inaccurate DOC_ID.
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        url = data['url']
                        content = data['content']
                        
                        #url duplication
                        if '#' in url:
                            url = url.split('#')[0]
                        if url in URL_SEEN:
                            continue
                        URL_SEEN.add(url)

                        DOC_ID_TO_URL[DOC_ID] = url

                        all_tokens, important_tokens = parse_url_content(content)

                        build_index(DOC_ID, all_tokens, important_tokens, CHUNK_INDEX)

                        DOC_ID += 1

                        # if the doc count hits 14000, it is stored in one chunk
                        if DOC_ID % CHUNK_SIZE == 0:
                            save_chunk(CHUNK_INDEX, CHUNK_ID)
                            CHUNK_INDEX.clear()
                            CHUNK_ID += 1
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {file_path}: {e}")
    # save final partial chunk
    if CHUNK_INDEX:
        save_chunk(CHUNK_INDEX, CHUNK_ID)
        
    #save url
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(DOC_ID_TO_URL, f)

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