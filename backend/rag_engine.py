import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.embeddings import YandexGPTEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TolstoyRAG:
    def __init__(self):
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        self.api_key = os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("YANDEX_FOLDER_ID and YANDEX_API_KEY must be set in .env")

        self.embeddings = YandexGPTEmbeddings(
            api_key=self.api_key,
            folder_id=self.folder_id
        )
        
        self.llm = ChatYandexGPT(
            api_key=self.api_key,
            folder_id=self.folder_id,
            temperature=0.3,
            max_tokens=2000
        )
        
        self.vector_store = None
        self.index_path = "backend/faiss_index"

    def load_documents(self, data_dir="data/processed"):
        documents = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                path = os.path.join(data_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    # Add metadata prefix to help the model identify source types
                    prefix = f"Источник: {filename}\n\n"
                    documents.append(Document(page_content=prefix + text, metadata={"source": filename}))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} text chunks from documents.")
        return splits

    def build_index(self):
        if os.path.exists(self.index_path):
            logger.info("Loading existing FAISS index...")
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Building new FAISS index...")
            splits = self.load_documents()
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            self.vector_store.save_local(self.index_path)
            logger.info("Index saved.")

    def query(self, user_input: str):
        if not self.vector_store:
            self.build_index()
            
        # Retrieval
        docs = self.vector_store.similarity_search(user_input, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # System prompt for Tolstoy persona
        system_prompt = (
            "Ты — Лев Николаевич Толстой, великий русский писатель и философ. "
            "Твоя речь полна достоинства, мудрости и морального поиска. Ты используешь стиль конца XIX века, "
            "но изъясняешься понятно для современного собеседника. "
            "Твои ответы должны основываться на твоих письмах, дневниках и трудах, предоставленных в контексте. "
            "Если в контексте нет прямого ответа, отвечай в своем философском духе, опираясь на свои убеждения: "
            "непротивление злу насилием, опрощение, поиск истины и Бога, важность крестьянского труда и земельного вопроса. "
            "Никогда не выходи из образа. Обращайся к собеседнику вежливо, но назидательно."
        )
        
        messages = [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": f"Контекст из моих трудов:\n{context}\n\nВопрос собеседника: {user_input}"}
        ]
        
        # Call Yandex GPT
        response = self.llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    # Test run
    rag = TolstoyRAG()
    rag.build_index()
    print(rag.query("Что вы думаете о смысле жизни?"))
