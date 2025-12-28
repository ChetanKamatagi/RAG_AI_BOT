import os
import shutil  
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline
from load_data import load_documents

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 800, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        print(f"[INFO] Loading embedding model: {embedding_model}...")
        self.model = SentenceTransformer(embedding_model)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self, documents: List[Any]):
        if not documents:
            print("[WARN] No documents provided to build vector store.")
            return

        print(f"[INFO] Building vector store in '{self.persist_dir}' from {len(documents)} docs...")
        
        # 1. Chunking
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        
        if not chunks:
            print("[WARN] No chunks created. Check document content.")
            return

        # 2. Embedding
        embeddings = emb_pipe.embed_chunks(chunks)
        
        metadatas = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta["text"] = chunk.page_content
            metadatas.append(meta)

        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        
        print(f"[INFO] Added {embeddings.shape[0]} vectors to index.")

    def save(self):
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if self.index is not None:
            faiss.write_index(self.index, faiss_path)
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print(f"[INFO] Saved index to {self.persist_dir}")
        else:
            print("[WARN] No index to save!")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"[WARN] No existing index found at {self.persist_dir}")
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"[INFO] Loaded index from {self.persist_dir} ({self.index.ntotal} vectors)")
        return True

    def clear(self):
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.index = None
        self.metadata = []
        print(f"[INFO] Cleared database at {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return []

        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata) and idx >= 0:
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(dist) # Convert numpy float to standard float
                })
        return results

    def query(self, query_text: str, top_k: int = 3):
        # print(f"[INFO] Querying: '{query_text}'")
        if self.index is None:
            print("[WARN] Index is empty. Cannot query.")
            return []
            
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

if __name__ == "__main__":

    # 1. Profile DB
    profile_db = FaissVectorStore(persist_dir="faiss_profile")
    profile_db.load() 

    # 2. User DB (Simulating a new upload)
    user_db = FaissVectorStore(persist_dir="faiss_user")
    user_db.clear() # Wipe old user data
    
    # Simulate building
    # docs = load_documents("some_path.pdf")
    # user_db.build_from_documents(docs)
    
    # print("Vector Store class is ready for Dual-DB architecture.")