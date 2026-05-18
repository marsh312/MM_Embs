import math
import numpy as np
from tqdm import tqdm
import re

class BM25Okapi:
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_len = []
        self.idf = {}
        self.inverted_index = {}  # term -> list of (doc_index, freq)
        
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer
        
        print("Indexing corpus...")
        self._initialize(corpus)

    def _default_tokenizer(self, text):
        # Simple whitespace tokenizer with lowercasing and basic cleanup
        text = str(text).lower()
        # Keep alphanumeric
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def _initialize(self, corpus):
        total_len = 0
        
        for i, text in enumerate(tqdm(corpus, desc="Building BM25 Index")):
            tokens = self.tokenizer(text)
            length = len(tokens)
            self.doc_len.append(length)
            total_len += length
            
            # Count frequencies in this doc
            freqs = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            
            # Update inverted index
            for token, freq in freqs.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((i, freq))
        
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        # Compute IDFs
        print("Computing IDFs...")
        idf_sum = 0
        negative_idfs = []
        
        for token, doc_list in self.inverted_index.items():
            df = len(doc_list)
            # Standard BM25 IDF
            idf = math.log(self.corpus_size - df + 0.5) - math.log(df + 0.5)
            self.idf[token] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(token)
        
        self.average_idf = idf_sum / len(self.idf) if len(self.idf) > 0 else 0
        
        # Handle negative IDFs (common in Okapi BM25 for very frequent terms)
        eps = self.epsilon * self.average_idf
        for token in negative_idfs:
            self.idf[token] = eps

    def get_scores(self, query):
        """
        Get scores for all documents for a given query.
        Returns a dictionary {doc_idx: score} for non-zero scores.
        """
        query_tokens = self.tokenizer(query)
        scores = {}  # doc_idx -> score
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
                
            idf = self.idf[token]
            for doc_idx, freq in self.inverted_index[token]:
                doc_len = self.doc_len[doc_idx]
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = numerator / denominator
                scores[doc_idx] = scores.get(doc_idx, 0) + score
                
        return scores

    def retrieve(self, query, top_k=10):
        """
        Retrieve top_k documents for a query.
        Returns list of (doc_idx, score) sorted by score descending.
        """
        scores = self.get_scores(query)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
