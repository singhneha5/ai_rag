from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class HybridSearcher:
    def __init__(self, chunks):
        self.chunks = chunks
        # Initialize TF-IDF for keyword search
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
        self.tfidf_matrix = self.tfidf.fit_transform(chunks)

    def keyword_search(self, query, k=5):
        """BM25-style keyword search using TF-IDF"""
        query_vec = self.tfidf.transform([query])
        scores = query_vec.dot(self.tfidf_matrix.T).toarray().flatten()
        top_indices = np.argsort(scores)[-k:][::-1]
        return top_indices, scores[top_indices]

    def hybrid_search(self, query, semantic_scores, semantic_indices, k=3, alpha=0.5):
        """
        Combine keyword and semantic search results
        alpha=0.5 means 50% keyword, 50% semantic
        """
        keyword_indices, keyword_scores = self.keyword_search(query, k=k * 2)

        # Normalize scores to [0, 1]
        keyword_scores_norm = (keyword_scores - keyword_scores.min()) / (
            keyword_scores.max() - keyword_scores.min() + 1e-10
        )
        semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (
            semantic_scores.max() - semantic_scores.min() + 1e-10
        )

        # Create a combined score dictionary
        combined_scores = {}
        for idx, score in zip(keyword_indices, keyword_scores_norm):
            combined_scores[idx] = alpha * score

        for idx, score in zip(semantic_indices, semantic_scores_norm):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * score
            else:
                combined_scores[idx] = (1 - alpha) * score

        # Get top k by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sorted_results[:k]]
        return top_indices
