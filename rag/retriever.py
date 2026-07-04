"""
RAG Retriever

Implements Retrieval-Augmented Generation (Week 2 skill) over a small
knowledge base of grid management guidelines. Uses TF-IDF + cosine
similarity for retrieval -- a lightweight, fully explainable, classic
RAG retrieval method (no heavy embedding model / GPU dependency needed,
appropriate for a solo bootcamp project).

Usage:
    retriever = KnowledgeBaseRetriever()
    top_docs = retriever.retrieve("holiday demand forecast deviation", k=3)
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


class KnowledgeBaseRetriever:
    def __init__(self, kb_dir: str = DEFAULT_KB_DIR):
        self.kb_dir = kb_dir
        self.doc_names = []
        self.doc_texts = []
        self._load_documents()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _load_documents(self):
        paths = sorted(glob.glob(os.path.join(self.kb_dir, "*.txt")))
        if not paths:
            raise FileNotFoundError(f"No knowledge base documents found in {self.kb_dir}")
        for path in paths:
            with open(path, "r") as f:
                text = f.read().strip()
            self.doc_names.append(os.path.basename(path))
            self.doc_texts.append(text)

    def retrieve(self, query: str, k: int = 3) -> list:
        """
        Returns the top-k most relevant knowledge base documents for the
        given query, as a list of {"source": filename, "text": content, "score": float}.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue  # skip completely irrelevant matches
            results.append({
                "source": self.doc_names[idx],
                "text": self.doc_texts[idx],
                "score": round(float(scores[idx]), 4),
            })
        return results


if __name__ == "__main__":
    # quick manual test
    r = KnowledgeBaseRetriever()
    results = r.retrieve("holiday demand forecast deviation above average", k=3)
    for res in results:
        print(f"[{res['score']}] {res['source']}")
        print(res['text'][:100], "...\n")
