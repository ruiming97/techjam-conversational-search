import json
from rank_bm25 import BM25Okapi

class CatalogSearcher:
    def __init__(self, file_path):
        print("Initializing CatalogSearcher (One-time setup)...")
        self.catalog = []
        search_corpus = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.catalog.append(item)
                
                features_text = " ".join(item.get("features", []))
                combined_text = f"{item.get('title', '')} {features_text} {item.get('store', '')}"
                search_corpus.append(combined_text.lower().split())
                
        self.bm25 = BM25Okapi(search_corpus)
        print("Index ready.")

    def retrieve(self, user_query, top_k=5):
        # This method runs instantly on every conversational turn
        tokenized_query = user_query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.catalog, n=top_k)

if __name__ == "__main__":
    # 1. Start the engine once
    search_engine = CatalogSearcher("data/catalog.jsonl")
    
    # 2. Execute multiple searches instantly
    print("\n--- Search 1 ---")
    results_1 = search_engine.retrieve("kandinsky fabric earrings")
    for r in results_1: print(r.get('title'))
        
    print("\n--- Search 2 ---")
    results_2 = search_engine.retrieve("black running shoes")
    for r in results_2: print(r.get('title'))