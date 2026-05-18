import os
from torch.utils.data import Dataset

class TomoroColQwen3QueryDataset(Dataset):
    def __init__(self, query_data):
        self.texts = []
        self.ids = []
        self.targets = []
        
        for sample in query_data:
            self.texts.append(sample.get("question", ""))
            self.ids.append(sample.get("q_id"))
            self.targets.append(sample.get("target_ids", []))
    
    def __getitem__(self, item):
        return {
            "text": self.texts[item],
            "id": self.ids[item],
            "target": self.targets[item]
        }
    
    def __len__(self):
        return len(self.texts)

class TomoroColQwen3CorpusDataset(Dataset):
    def __init__(self, corpus_data):
        self.contexts = []
        self.image_paths = []
        self.ids = []
        
        for sample in corpus_data:
            self.contexts.append(sample.get("context", ""))
            self.image_paths.append(sample.get("page_path", ""))
            self.ids.append(sample.get("p_id"))
    
    def __getitem__(self, item):
        return {
            "text": self.contexts[item],
            "image": self.image_paths[item],
            "id": self.ids[item]
        }
    
    def __len__(self):
        return len(self.contexts)
