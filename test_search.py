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

    def retrieve(self, user_query, constraints=None, top_k=5):
        # Default to an empty dictionary if no constraints are provided
        if constraints is None:
            constraints = {}

        # 1. Get BM25 scores for the entire catalog instantly
        tokenized_query = user_query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # 2. Pair every item with its computed score
        scored_items = list(zip(self.catalog, scores))

        filtered_items = []
        
        # 3. THE FILTER PHASE: Enforce hard constraints
        for item, score in scored_items:
            passes_constraints = True
            
            # Check every rule in the constraints dictionary
            for key, required_value in constraints.items():
                # For V1, we check if the required value exists anywhere in the item's title or features
                item_text = f"{item.get('title', '')} {item.get('features', [])}".lower()
                
                if required_value.lower() not in item_text:
                    passes_constraints = False
                    break # Fails the rule, drop the item
                    
            if passes_constraints:
                filtered_items.append((item, score))

        # 4. THE RANKING PHASE: Sort surviving items by score (highest first)
        filtered_items.sort(key=lambda x: x[1], reverse=True)

        # 5. Return the top K items (stripping the score away)
        return [item for item, score in filtered_items[:top_k]]

if __name__ == "__main__":
    search_engine = CatalogSearcher("data/catalog.jsonl")
    
    print("\n--- Search 1: Free Text Only ---")
    results_1 = search_engine.retrieve("running shoes")
    for r in results_1: print(r.get('title'))
        
    print("\n--- Search 2: Text + Hard Constraint ---")
    # Simulating a user who specifically asked for the brand "Newton"
    mocked_constraints = {"brand": "Newton"}
    results_2 = search_engine.retrieve("running shoes", constraints=mocked_constraints)
    for r in results_2: print(r.get('title'))