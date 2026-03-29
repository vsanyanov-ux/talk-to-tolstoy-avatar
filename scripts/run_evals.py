import sys
import os
import json

# Add backend to sys.path to allow imports
sys.path.append(os.path.abspath("backend"))

try:
    from yandex_gpt import YandexGPTHelper
except ImportError:
    print("Error: Could not import YandexGPTHelper. Make sure you are running from the root directory.")
    sys.exit(1)

# Ensure stdout handles Cyrillic in Windows/CMD
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    helper = YandexGPTHelper()
    log_path = "backend/logs/eval_traces.jsonl"
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}. Try interacting with the chatbot first.")
        return

    print("="*60)
    print(" TOLSTOY RAG EVALUATOR - DIAGNOSTIC REPORT ")
    print("="*60)

    traces = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    if not traces:
        print("No traces found in the log file.")
        return

    # Process the last 5 traces to avoid heavy API usage
    num_to_eval = min(5, len(traces))
    recent_traces = traces[-num_to_eval:]
    
    print(f"Analyzing the last {num_to_eval} interactions...\n")

    for i, t in enumerate(recent_traces):
        print(f"Interaction #{len(traces) - num_to_eval + i + 1}")
        print(f"USER: {t['original_query']}")
        print(f"SEARCH QUERY: {t.get('search_query', 'N/A')}")
        
        # Consolidate context for the judge
        context_preview = "\n---\n".join([f"({r['source']}): {r['text'][:200]}..." for r in t['retrieved_results']])
        
        print(f"JUDGING (Calling Yandex GPT)...")
        eval_raw = helper.evaluate_response(t['original_query'], context_preview, t['final_response'])
        
        try:
            # Clean up potential markdown formatting if LLM added it
            clean_eval = eval_raw.replace("```json", "").replace("```", "").strip()
            res = json.loads(clean_eval)
            print(f"RESULT:")
            print(f"  - Relevance:   {res.get('relevance', '?')}/5")
            print(f"  - Faithfulness: {res.get('faithfulness', '?')}/5")
            print(f"  - Root Cause:   {res.get('root_cause', 'unknown')}")
            print(f"  - Reason:       {res.get('reason', 'N/A')}")
        except Exception:
            print(f"  - Raw Evaluator Output: {eval_raw}")
        
        print("-" * 40)

    print("\nEvaluation complete. Check 'backend/logs/eval_traces.jsonl' for raw data.")

if __name__ == "__main__":
    main()
