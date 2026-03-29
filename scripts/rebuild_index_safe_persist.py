import sys
import os
import logging
import time
import json

# Ensure the root project dir is in sys.path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from rag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafeRebuilder")

def rebuild():
    # SETTINGS
    data_path = "Data/processed"
    index_file = "backend/index.json"
    save_every = 10 # FAST SAVING FOR VISIBILITY
    delay = 0.3 # 3 req/sec - VERY SAFE
    
    logger.info(f"Starting ULTRA SAFE Rebuild from {data_path} to {index_file}...")
    rag = SimpleRAG(data_dir=data_path, index_file=index_file, use_reranker=False)
    
    # ALWAYS load chunks from disk first
    rag.chunks = rag.load_and_chunk()
    logger.info(f"Loaded {len(rag.chunks)} chunks from source texts.")

    # Main serial processing loop
    for i, chunk in enumerate(rag.chunks):
        if "embedding" not in chunk or not chunk["embedding"]:
            success = False
            retries = 0
            while not success and retries < 3:
                try:
                    time.sleep(delay)
                    emb = rag.helper.get_embedding(chunk["text"])
                    if emb:
                        chunk["embedding"] = emb
                        success = True
                    else:
                        retries += 1
                        time.sleep(1)
                except Exception as e:
                    if "rate quota limit" in str(e).lower():
                        logger.warning("Rate limit. Waiting 10s...")
                        time.sleep(10)
                    else:
                        logger.error(f"Error at chunk {i}: {e}")
                        time.sleep(2)
                    retries += 1
            
            if (i + 1) % save_every == 0:
                logger.info(f"SAVE Checkpoint: {i+1}/{len(rag.chunks)} processed.")
                rag.save_index()

    rag.save_index()
    logger.info("ULTRA SAFE REBUILD FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    rebuild()
