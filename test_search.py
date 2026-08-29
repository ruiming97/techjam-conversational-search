import json
from rank_bm25 import BM25Okapi

def build_index_and_search(file_path, user_query):
    catalog = []
    search_corpus = []
    
    # 1. Load data and extract text for indexing
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            catalog.append(item)
            
            # Combine title and features into one massive string per product
            features_text = " ".join(item.get("features", []))
            combined_text = f"{item.get('title', '')} {features_text} {item.get('store', '')}"
            
            # Tokenize the string (split into lowercase words) for the index
            search_corpus.append(combined_text.lower().split())
            
    # 2. Build the BM25 Index in-memory
    print("Building index (this takes a few seconds)...")
    bm25 = BM25Okapi(search_corpus)
    
    # 3. Search the Index
    tokenized_query = user_query.lower().split()
    top_5_items = bm25.get_top_n(tokenized_query, catalog, n=5)
    
    return top_5_items

if __name__ == "__main__":
    # Test a fuzzy conversational query 
    results = build_index_and_search("data/catalog.jsonl", "kandinsky fabric earrings")
    
    # Print just the titles of the top 5 matches
    for rank, product in enumerate(results, 1):
        print(f"{rank}. {product.get('title')}")