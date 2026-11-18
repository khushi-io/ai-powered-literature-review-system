"""
vector_store.py
A lightweight local vector store using numpy + pickle.
No external vector database (e.g. ChromaDB) is used, since those pull
in grpc, which triggers a Windows Application Control DLL block on
this machine. This module handles: embedding text, saving/loading
vectors to disk, similarity search, and KMeans clustering (used by
synthesizer.py to group papers into themes).
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import config

# Load the embedding model once, at import time, so every file that
# imports vector_store reuses the same loaded model instead of
# reloading it repeatedly (which would be slow).
print("Loading embedding model (first time may take a few seconds)...")
_embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")


def embed_text(text):
    """
    Convert a string into a 384-dimensional embedding vector.
    Returns a numpy array of shape (384,).
    """
    return _embedding_model.encode(text, convert_to_numpy=True)


def embed_texts(texts):
    """
    Convert a list of strings into embeddings in one batch call
    (faster than calling embed_text in a loop for many items).
    Returns a numpy array of shape (len(texts), 384).
    """
    return _embedding_model.encode(texts, convert_to_numpy=True)


class VectorStore:
    """
    Holds a list of items, each with:
      - an id (e.g. OpenAlex work id)
      - metadata (dict, e.g. title, abstract, authors, year)
      - an embedding vector (numpy array)

    Saved to / loaded from a single .pkl file on disk, keyed by a
    'store_name' (e.g. the search topic, slugified), so each topic's
    papers are cached separately under config.VECTOR_STORE_DIR.
    """

    def __init__(self, store_name):
        self.store_name = store_name
        self.ids = []
        self.metadata = []
        self.embeddings = None  # numpy array, shape (N, 384), set once items are added

    def _file_path(self):
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.store_name)
        return os.path.join(config.VECTOR_STORE_DIR, f"{safe_name}.pkl")

    def exists_on_disk(self):
        return os.path.exists(self._file_path())

    def add_items(self, ids, metadata_list, texts):
        """
        Add a batch of items. `texts` is what gets embedded
        (typically title + abstract concatenated).
        """
        new_embeddings = embed_texts(texts)
        self.ids.extend(ids)
        self.metadata.extend(metadata_list)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def save(self):
        with open(self._file_path(), "wb") as f:
            pickle.dump(
                {"ids": self.ids, "metadata": self.metadata, "embeddings": self.embeddings},
                f,
            )
        print(f"[vector_store] Saved {len(self.ids)} items to {self._file_path()}")

    def load(self):
        with open(self._file_path(), "rb") as f:
            data = pickle.load(f)
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        self.embeddings = data["embeddings"]
        print(f"[vector_store] Loaded {len(self.ids)} items from {self._file_path()}")
        return self

    def search(self, query_text, top_k=5):
        """
        Returns the top_k most similar items to query_text, using
        cosine similarity. Each result is (metadata, score).
        """
        if self.embeddings is None or len(self.ids) == 0:
            return []

        query_vec = embed_text(query_text)

        # Cosine similarity = dot product of normalized vectors
        norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = query_vec / np.linalg.norm(query_vec)
        scores = norm_embeddings @ norm_query

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.metadata[i], float(scores[i])) for i in top_indices]

    def cluster(self, num_clusters):
        """
        Groups all stored items into num_clusters themes using KMeans.
        Returns a list of cluster labels (one integer per item, same
        order as self.ids / self.metadata).
        Used by synthesizer.py to group papers into themes before
        summarizing each group.
        """
        if self.embeddings is None or len(self.ids) == 0:
            return []

        n_items = len(self.ids)
        # KMeans needs at least as many items as clusters; shrink if needed
        actual_k = min(num_clusters, n_items)

        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        return labels.tolist()