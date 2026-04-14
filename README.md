# 🎭 Talk to Tolstoy Avatar (v2.0.0)

**"Душа человеческая есть величайшее чудо."** — *Лев Толстой*

Проект представляет собой интерактивного ИИ-аватара Льва Николаевича Толстого, построенного на базе современных технологий **Agentic RAG**. Это не просто чат-бот, а цифровой философ, способный на глубокое размышление, наставничество и диалог, опираясь на свои дневники, письма и философские труды.

---

## 🧠 Агентская Архитектура (Agentic Workflow)

В версии 2.0 мы перешли от линейного поиска к циклическому агентскому процессу с использованием **LangGraph**. Это позволяет системе критически оценивать информацию перед ответом.

`mermaid
graph TD
    A[Пользователь] --> B(Condense Query)
    B --> C{История есть?}
    C -->|Да| D[Rewrite Query]
    C -->|Нет| E[Retrieve Docs]
    D --> E
    E --> F[Grade Documents]
    F --> G{Релевантно?}
    G -->|Нет / Retry < 2| B
    G -->|Да| H[Generate Response]
    H --> I[Tolstoy Persona]
    I --> J[Сноски и Цитаты]
    J --> A
`

---

## 🛠 Технологический стек (Tech Stack)

| Категория | Технологии |
| :--- | :--- |
| **Core Engine** | **LangGraph**, Python 3.10+ |
| **LLM Orchestration** | **LiteLLM**, Mistral Large 24.11 |
| **Observability** | **Langfuse v4**, Monitoring / Cost Tracking |
| **Retrieval** | FAISS, Hybrid Search (Vector + BM25) |
| **Reranker** | Cross-Encoder (MiniLM) |
| **Frontend** | React 19, Vite, Glassmorphism CSS |

---

## 🚀 Ключевые возможности

1. **Agentic Reasoning**: Система умеет переформулировать сложные вопросы, учитывать историю диалога и отсеивать нерелевантный "шум".
2. **Self-Correction**: Если база знаний не дала ответа с первого раза, агент пробует изменить стратегию поиска.
3. **Academic Precision**: Использование надстрочных знаков (¹, ², ³) для точного цитирования источников из корпуса текстов.
4. **Persona-Aware Refusal**: Вместо сухого "я не знаю", Толстой признает несовершенство памяти и переводит разговор на вечные темы (душа, истина, любовь).
5. **Multi-Source Knowledge**: Интегрированная база из дневников, писем и биографических очерков.

---

## 📋 Changelog (Кратко)

- **v2.0.0 (Current)**: Переход на **LangGraph**, интеграция **Mistral Large**, система оценки релевантности (Grading), расширенный мониторинг через Langfuse.
- **v1.0.0**: Стабильный RAG с Cross-Encoder реранкером и академическими ссылками.

---

## ⚙️ Установка и запуск

1. **Клонирование**:
   `ash
   git clone https://github.com/vsanyanov-ux/talk-to-tolstoy-avatar.git
   `
2. **Окружение**:
   Создайте .env файл на основе примера и добавьте API ключи (Mistral/LiteLLM, Langfuse).
3. **Индексация**:
   `ash
   python scripts/rebuild_index_safe_persist.py
   `
4. **Запуск**:
   `ash
   # Backend
   python backend/server.py
   # Frontend
   cd frontend && npm install && npm run dev
   `

---

*Разработано с заботой о смысле и слове.*
