import os
import json
import numpy as np
import logging
import time
import re
import random
from concurrent.futures import ThreadPoolExecutor
from yandex_gpt import YandexGPTHelper
from rank_bm25 import BM25Okapi
import pymorphy3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self, data_dir="data/processed", index_file="backend/index.json", use_reranker=True):
        self.helper = YandexGPTHelper()
        self.data_dir = data_dir
        self.index_file = index_file
        self.chunks = []
        self.embeddings = []
        self.reranker = None
        self.bm25 = None
        self.morph = pymorphy3.MorphAnalyzer()
        
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading Reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
                logger.info("Reranker model loaded.")
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}. Falling back to standard search.")

    def preprocess_text(self, text):
        """Tokenize and lemmatize Russian text for BM25."""
        tokens = re.findall(r'\w+', text.lower())
        return [self.morph.parse(t)[0].normal_form for t in tokens]

    def load_and_chunk(self):
        chunks = []
        max_chunk_size = 1500
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                path = os.path.join(self.data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    import re
                    page_sections = re.split(r"--- \[STR_(\d+)\] ---", text)
                    current_page = 0
                    for i in range(len(page_sections)):
                        section = page_sections[i].strip()
                        if not section: continue
                        if section.isdigit():
                            current_page = int(section)
                            continue
                        paragraphs = section.split("\n\n")
                        for p in paragraphs:
                            p = p.strip()
                            if not p: continue
                            if len(p) > max_chunk_size:
                                for j in range(0, len(p), max_chunk_size):
                                    sub_p = p[j:j + max_chunk_size].strip()
                                    if len(sub_p) > 50:
                                        chunks.append({"text": sub_p, "source": filename, "page": current_page})
                            elif len(p) > 50:
                                chunks.append({"text": p, "source": filename, "page": current_page})
        return chunks

    def build_index(self):
        if not os.path.exists(self.index_file):
            logger.warning("Index file not found. Starting with empty memory.")
            self.chunks = []
            self.embeddings = np.zeros((0, 768)) # Common emb size
            return

        logger.info("Loading existing index...")
        with open(self.index_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                self.chunks = data["chunks"]
                # Embeddings might be empty at very start
                if len(data["embeddings"]) > 0:
                    self.embeddings = np.array(data["embeddings"])
                else:
                    self.embeddings = np.zeros((0, 768))
            except Exception as e:
                logger.error(f"Failed to load index: {e}. Starting empty.")
                self.chunks = []
                self.embeddings = np.zeros((0, 768))
                return
                
            # Initialize BM25 only if we have words
            if len(self.chunks) > 0:
                logger.info(f"Initializing BM25 index for {len(self.chunks)} chunks...")
                start_time = time.time()
                corpus_lemmas = []
                for i, c in enumerate(self.chunks):
                    lemmas = c.get("lemmas")
                    if lemmas is None:
                        lemmas = self.preprocess_text(c["text"])
                    corpus_lemmas.append(lemmas)
                    if i % 1000 == 0 and i > 0:
                        logger.info(f"Processed {i}/{len(self.chunks)} chunks for BM25...")
                
                self.bm25 = BM25Okapi(corpus_lemmas)
                logger.info(f"BM25 index initialized in {time.time() - start_time:.2f}s")
            else:
                self.bm25 = None
        return
        
        logger.info(f"Indexing all {len(self.chunks)} chunks with strict rate limits (max_workers=2)...")
        embeddings = [None] * len(self.chunks)
        
        def process_chunk(idx):
            chunk = self.chunks[idx]
            time.sleep(random.uniform(0.5, 0.8)) # Extra safe delay
            for attempt in range(3):
                emb = self.helper.get_embedding(chunk["text"])
                if emb:
                    embeddings[idx] = emb
                    return
                time.sleep(5 * (attempt + 1)) # Longer retry wait
            logger.error(f"Failed to get embedding for chunk {idx}")

        with ThreadPoolExecutor(max_workers=1) as executor: # Absolute safety: 1 worker
            for i in range(len(self.chunks)):
                executor.submit(process_chunk, i)
                if i % 100 == 0:
                    logger.info(f"Scheduled chunk {i}/{len(self.chunks)}...")
        
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        self.chunks = [self.chunks[i] for i in valid_indices]
        self.embeddings = np.array([embeddings[i] for i in valid_indices])
        
        logger.info(f"Index built with {len(self.chunks)} valid entries.")
        
        # Add lemmas for BM25
        logger.info("Computing lemmas for hybrid search...")
        for c in self.chunks:
            c["lemmas"] = self.preprocess_text(c["text"])
            
        # Initialize BM25
        self.bm25 = BM25Okapi([c["lemmas"] for c in self.chunks])
        
        self.save_index()
        logger.info("Index saved.")

    def save_index(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings.tolist()
            }, f, ensure_ascii=False)

    def search(self, query: str, k=5, rerank_top=50):
        # Source classification mapping
        source_types = {
            "diaries.txt": "BIOGRAPHY",
            "letters.txt": "BIOGRAPHY",
            "on_land.txt": "PHILOSOPHY"
        }

        # 1. Vector similarity search (Semantic)
        query_emb = self.helper.get_query_embedding(query)
        if query_emb is None: return []
        
        if len(self.embeddings) == 0:
            logger.warning("No embeddings loaded. Skipping vector search.")
            similarities = np.zeros(len(self.chunks))
        else:
            similarities = np.dot(self.embeddings, query_emb) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
            )
        vector_rank_idx = np.argsort(similarities)[::-1][:rerank_top]
        
        # 2. BM25 Keyword search (Lexical)
        if self.bm25:
            query_lemmas = self.preprocess_text(query)
            bm25_scores = self.bm25.get_scores(query_lemmas)
            bm25_rank_idx = np.argsort(bm25_scores)[::-1][:rerank_top]
        else:
            bm25_rank_idx = []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_k = 60
        
        for rank, idx in enumerate(vector_rank_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + rrf_k)
        for rank, idx in enumerate(bm25_rank_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + rrf_k)
            
        sorted_rrf_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:rerank_top]
        candidates = [self.chunks[i] for i in sorted_rrf_idx]
        
        # Phase 2: Reranking
        final_results = []
        if self.reranker and len(candidates) > 0:
            pairs = [[query, c['text']] for c in candidates]
            scores = self.reranker.predict(pairs)
            reranked_idx = np.argsort(scores)[::-1]
            final_results = [candidates[i] for i in reranked_idx[:k]]
        else:
            final_results = candidates[:k]

        # Add Source-Type labels
        for res in final_results:
            res["source_type"] = source_types.get(res["source"], "GENERAL")
            
        return final_results

if __name__ == "__main__":
    rag = SimpleRAG()
    # To force initial index load and reranker load
    res = rag.search("О смысле жизни")
    for r in res:
        print(f"[{r['source']}]: {r['text'][:100]}...")
