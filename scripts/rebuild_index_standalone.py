import sys
import os
import logging

# Ensure the root project dir is in sys.path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from rag import SimpleRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndexRebuilder")

def rebuild():
    logger.info("Initializing RAG for a clean rebuild (NO RERANKER to save memory)...")
    # Speed rebuild: set use_reranker=False to avoid OOM
    rag = SimpleRAG(data_dir="Data/processed", index_file="backend/index.json", use_reranker=False)
    
    # Force rebuild
    if os.path.exists("backend/index.json"):
        os.remove("backend/index.json")
        
    logger.info("Starting build_index (1 worker for maximum stability)...")
    # We already patched rag.py to Use 2 workers, but this script will run it linearly
    rag.build_index()
    logger.info("Indexing finished successfully!")

if __name__ == "__main__":
    rebuild()
