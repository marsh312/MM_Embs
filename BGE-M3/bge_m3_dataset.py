import json
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class BGEM3QueryDataset(Dataset):
    def __init__(self, query_path):
        self.data = []
        with open(query_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __getitem__(self, item):
        sample = self.data[item]
        return {
            "q_text": sample.get("question", ""),
            "q_id": sample.get("q_id"),
            "target_ids": sample.get("target_ids", [])
        }
    
    def __len__(self):
        return len(self.data)

class BGEM3CorpusDataset(Dataset):
    def __init__(self, corpus_path, pages_path):
        """
        Args:
            corpus_path: Path to corpus.jsonl (defines the list of documents/pages to retrieve from)
            pages_path: Path to pages.jsonl (contains the actual text content mapped by doc_id and page_index)
        """
        self.corpus_items = []
        
        # Load corpus items (defining the order and IDs)
        print(f"Loading corpus list from {corpus_path}...")
        with open(corpus_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.corpus_items.append(json.loads(line))
        
        # Load pages content into a lookup dictionary
        print(f"Loading pages content from {pages_path}...")
        self.pages_lookup = {}
        with open(pages_path, 'r') as f:
            for line in tqdm(f, desc="Loading pages"):
                if line.strip():
                    page_data = json.loads(line)
                    # Create a unique key using doc_id and page_index
                    # Ensure types match (e.g. string vs int). 
                    # In corpus.jsonl: "doc_id": "1107.3765", "page_idx": 0
                    # In pages.jsonl: "doc_id": "0001010", "page_index": 0
                    # Note the key difference: page_idx vs page_index.
                    
                    doc_id = str(page_data.get("doc_id"))
                    page_idx = int(page_data.get("page_index"))
                    self.pages_lookup[(doc_id, page_idx)] = page_data.get("text", "")

    def __getitem__(self, item):
        sample = self.corpus_items[item]
        doc_id = str(sample.get("doc_id"))
        # In corpus.jsonl it uses "page_idx"
        page_idx = int(sample.get("page_idx"))
        
        # Lookup text
        text = self.pages_lookup.get((doc_id, page_idx), "")
        
        if not text:
            print(f"Warning: Text not found for doc_id={doc_id}, page_idx={page_idx}")
            
        return {
            "text": text,
            "p_id": sample.get("p_id")
        }
    
    def __len__(self):
        return len(self.corpus_items)
