import json
import logging
from typing import List, TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from rag import SimpleRAG
from llm_helper import LLMHelper

logger = logging.getLogger(__name__)

# 1. State definition
class GraphState(TypedDict):
    query: str
    original_query: str
    history: List[dict]
    documents: List[dict]
    intent: str
    response: str
    retry_count: int

# 2. Nodes
def condense_query_node(state: GraphState):
    """Rewrites the query based on conversation history."""
    logger.info("--- NODE: Condense Query ---")
    helper = LLMHelper()
    condensed = helper.condense_query(state["query"], state["history"])
    new_retry_count = state.get("retry_count", 0) + 1
    return {**state, "query": condensed, "retry_count": new_retry_count}

def retrieve_node(state: GraphState):
    """Retrieves documents from the RAG index."""
    logger.info("--- NODE: Retrieve Documents ---")
    rag = SimpleRAG(use_reranker=False)
    rag.build_index()
    results = rag.search(state["query"], k=5)
    return {**state, "documents": results}

def grade_documents_node(state: GraphState):
    """Filters irrelevant documents."""
    logger.info("--- NODE: Grade Documents ---")
    helper = LLMHelper()
    doc_texts = [d["text"] for d in state["documents"]]
    relevance_mask = helper.grade_documents(state["query"], doc_texts)
    
    filtered_docs = [
        doc for doc, is_relevant in zip(state["documents"], relevance_mask) if is_relevant
    ]
    
    logger.info(f"Filtered {len(state['documents'])} -> {len(filtered_docs)} documents.")
    return {**state, "documents": filtered_docs}

def generate_node(state: GraphState):
    """Generates the final response in Tolstoy's persona."""
    logger.info("--- NODE: Generate Response ---")
    helper = LLMHelper()
    
    # 1. Classify intent (biography vs general)
    query_lower = state["original_query"].lower()
    personal_keywords = ["ты", "вы", "тебя", "вас", "себя", "жизнь", "биография", "писал", "чувствовал"]
    intent = "PERSONAL" if any(k in query_lower for k in personal_keywords) else "GENERAL"
    
    # 2. Check if we have source type BIOGRAPHY if intent is personal
    has_biography = any(d.get("source_type") == "BIOGRAPHY" for d in state["documents"])
    
    refusal_instruction = ""
    if intent == "PERSONAL" and not has_biography:
        refusal_instruction = (
            " ВНИМАНИЕ: Собеседник спрашивает о твоей жизни, но в твоих текущих записях нет точного свидетельства об этом факте. "
            "Не говори сухо 'я не знаю'. Ответь как мудрый старец: признай, что память твоя не сохранила этой суетной детали, "
            "но тут же переведи разговор на вечные темы — о душе, добре или истине, которые этот вопрос затрагивает."
        )

    # 3. Format context
    context_parts = []
    for d in state["documents"]:
        type_label = d.get("source_type", "GENERAL")
        context_parts.append(f"[{type_label}: {d['source']}, стр. {d.get('page', '?')}]:\n{d['text']}")
    context = "\n---\n".join(context_parts)

    system_prompt = (
        "Ты — граф Лев Николаевич Толстой, в роли мудрого учителя и духовного наставника. "
        "Твоя речь полна достоинства, мудрости и отеческой заботы о душе собеседника. "
        "Говори как человек XIX века. Ты не просто ищешь ответы в памяти (контексте), ты — философ.\n\n"
        "ПРАВИЛА ТВОЕГО ПОВЕДЕНИЯ:\n"
        "1. ОБЪЯСНЯЙ И НАСТАВЛЯЙ: Сначала дай краткий ответ на вопрос, а затем подробно разверни свою мысль. "
        "Объясняй причины своих убеждений. Почему ты считаешь именно так? В чем корень зла или блага в этом вопросе?\n"
        "2. БУДЬ СОБЕСЕДНИКОМ: Задавай встречные вопросы, призывай человека заглянуть в свою совесть. "
        "Диалог должен быть живым, а не поисковым.\n"
        "3. ЦИТИРОВАНИЕ: Твоя память — это предоставленные ниже труды. Используй их как основу, но облекай в живую речь. "
        "Никогда не используй квадратные скобки [ ]. Вместо этого ставь надстрочные знаки (¹, ², ³, ⁴, ⁵) в тексте. "
        "В самом конце обязательно добавь раздел 'Сноски:', где перечисли источники.\n"
        "4. ДИАЛОГ: Тщательно учитывай историю беседы. Если тебя спрашивают 'Почему?', отвечай на основе твоего предыдущего суждения.\n\n"
        f"{refusal_instruction}"
    )
    
    augmented_user_message = f"Твои труды и мысли для размышления:\n{context}\n\nТвой собеседник (друг твой) вопрошает: {state['original_query']}"
    response = helper.chat_completion(system_prompt, augmented_user_message, history=state["history"])
    
    return {**state, "response": response}

# 3. Graph Logic
def decide_to_generate(state: GraphState):
    """Determines whether to proceed to generation or try to re-query."""
    if not state["documents"] and state["retry_count"] < 2:
        logger.info(f"--- DECISION: No docs found (retry {state['retry_count']}), retrying search ---")
        return "retry"
    else:
        logger.info("--- DECISION: Proceed to generation ---")
        return "generate"

# 4. Build Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("condense_query", condense_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)

# Set Entry Point
workflow.set_entry_point("condense_query")

# Define Edges
workflow.add_edge("condense_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "retry": "condense_query", # Simple retry logic
        "generate": "generate"
    }
)

workflow.add_edge("generate", END)

# Compile
app_graph = workflow.compile()
