from flask import Flask, render_template, request, jsonify
import time
from search import search_query, load_byte_pos_offset_file, load_doc_mapping_file, preprocess_query, load_pagerank, load_hits

posting_byte_pos = load_byte_pos_offset_file()
doc_mapping = load_doc_mapping_file()
pr_scores = load_pagerank()
hits_scores = load_hits()
      
app = Flask(__name__) # create flask app instance

@app.route('/') # homepage url route
def home():
    return render_template("index.html")

@app.route("/api/search")
def api_search():
    query = request.args.get("q", "") # query request (default = empty)
    k = int(request.args.get("k", 5)) 
    
    start = time.time()
    query_tokens = preprocess_query(query)
    results = search_query(query_tokens, posting_byte_pos, doc_mapping, pr_scores, hits_scores, top_k=k)
    elapsed_ms = (time.time() - start) * 1000
    
    formatted_res = []
    for res in results:
        print(f"URL: {res['url']}")
        print(f"Score: {res['score']}")
        formatted_res.append({
            "url": res['url'],
            "score": round(res['score'], 4)
        })
    return jsonify({
        "query": query,
        "k": k,
        "time_ms": round(elapsed_ms, 2),
        "results": formatted_res
    })
    
if __name__ == "__main__":
    # run app on local development server with debug mode enabled
    app.run(host="0.0.0.0", port=8000, debug=True)