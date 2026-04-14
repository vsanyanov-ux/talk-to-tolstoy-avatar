# Changelog

All notable changes to the **Talk to Tolstoy Avatar** project will be documented in this file.

## [2.0.0] - 2026-04-14

### Added
- **Agentic Workflow (LangGraph)**: Migrated from a linear RAG pipeline to a dynamic, graph-based agent.
- **Document Grading Node**: Implemented a dedicated node to filter irrelevant retrieved documents before generation.
- **Query Condensation**: Advanced node to rewrite queries based on conversation history for better context awareness.
- **Self-Correction Logic**: The agent can now decide to retry searching or reformulating the query if initial results are insufficient.
- **LiteLLM Integration**: Unified API access for Mistral Large and other models with caching and monitoring support.
- **Mistral Large 3 Transition**: Upgraded from Yandex GPT to Mistral Large for superior reasoning and philosophical depth.
- **Langfuse v4 Observability**: Full tracing, cost tracking, and evaluation pipeline integration.

### Changed
- Refactored `backend/server.py` to use `backend/graph.py` as the main execution engine.
- Enhanced System Prompt for deeper, more "teacher-like" Tolstoy persona.
- Optimized UI with glassmorphism effects and smoother message transitions.

### Fixed
- Improved academic citation accuracy with better source classification.
- Resolved "Nuclear Filter" issues where the model would refuse to answer personal questions without explicit biography sources.

---

## [1.0.0] - 2026-04-10

### Added
- **Stable RAG Pipeline**: Core vector search combined with BM25 hybrid indexing.
- **Cross-Encoder Reranker**: MiniLM-based reranking for high-precision document selection.
- **Yandex GPT v3 Integration**: Initial LLM integration for conversational responses.
- **Source Classification**: Automatic identification of DIARIES, LETTERS, and BIOGRAPHY source types.
- **Academic Citations**: Footnote system with superscript numbering and source list at the bottom.
- **Automated Evaluations**: Initial `run_evals.py` script for Relevance and Faithfulness metrics.
- **Safe Persistence**: Reliable disk-based indexing for FAISS.

[2.0.0]: https://github.com/vsanyanov-ux/talk-to-tolstoy-avatar/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/vsanyanov-ux/talk-to-tolstoy-avatar/releases/tag/v1.0.0
