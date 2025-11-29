# Henry Bot M3 – Multi-Agent Intelligent Routing System

Repository: [https://github.com/estebmaister/henry_bot_M3](https://github.com/estebmaister/henry_bot_M3)

⸻

## 1. Context and Objectives

**Henry Bot M3** was implemented to simulate a real enterprise scenario: a mid-sized SaaS company with overloaded support teams and a high volume of misrouted tickets across HR, IT Support, and Finance.

The core business problems targeted were:

- HR questions landing in IT or Finance queues
- Slower resolution times due to manual triage and re-routing
- Inconsistent, non–policy-aligned answers from human agents

### Project Objectives

The assignment required:

1. A **multi-agent orchestration system** with:
   - An **Orchestrator Agent** that classifies user intent (e.g., HR vs Tech vs Finance)
   - **Conditional routing** to specialized RAG agents per department
2. An implementation based on **LangChain** components (chains, retrievers, agents) rather than ad-hoc code.   
3. **Full workflow tracing** using **Langfuse** for observability and debugging.   
4. At least **three specialized RAG agents** with domain-specific document collections.
5. A clear explanation of **technical decisions**.
6. **Bonus**: An Evaluator Agent that scores responses (1–10) on relevance, completeness, and accuracy, integrated via Langfuse.

Henry Bot M3 satisfies all these requirements and is fully reproducible from the repository.

⸻

## 2. System Overview

At a high level, the system works as follows:

1. A user query enters the **Multi-Agent Orchestrator**.
2. An **Intent Classifier** uses semantic similarity (Sentence Transformers) to classify the query into:
   - `hr`
   - `tech`
   - `finance`
3. Based on classification and confidence, the orchestrator **routes** the query to the corresponding **RAG agent**:
   - HR Assistant
   - IT Support Assistant
   - Finance Assistant
4. The chosen agent runs a **RAG pipeline**:
   - Retrieve relevant documents from a FAISS index
   - Generate an answer grounded on the retrieved docs
5. Optionally, a **Quality Evaluator Agent** scores the response.
6. **Langfuse** records the complete trace: intent classification, retrieval, answer generation, and evaluation.

This architecture closely follows modern **multi-agent** patterns for LLM applications, where multiple specialized agents collaborate under an orchestrator instead of a single all-purpose model.   

⸻

## 3. Architecture

### 3.1 Diagram

```text
┌───────────────────────────────────────────────────────┐
│                 Multi-Agent Orchestrator              │
├───────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌─────────────────────────┐ │
│  │ Intent Classifier │───▶│           Router        │ │
│  │  (Semantic        │    │  (Conditional Logic &   │ │
│  │   Similarity)     │    │   Agent Selection)      │ │
│  └───────────────────┘    └─────────────────────────┘ │
└─────────────────────────┬─────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌───────────────┐
│ HR Agent    │  │ Tech Agent  │  │ Finance Agent │
│ (RAG + LLM) │  │ (RAG + LLM) │  │ (RAG + LLM)   │
└─────────────┘  └─────────────┘  └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
              ┌─────────────────┐
              │ Quality         │
              │ Evaluator       │
              │ (Scoring &      │
              │  Assessment)    │
              └─────────────────┘
```

### 3.2 Core Components
	•	MultiAgentSystem (entrypoint with CLI interface)
	•	Intent Classifier (semantic similarity using Sentence Transformers)
	•	Department RAG Agents:
	•	HR Assistant
	•	IT Support Assistant
	•	Finance Assistant
	•	FAISS-based retrievers with persistent storage and caching
	•	CachedFAISSRetriever (production-ready caching layer)
	•	Quality Evaluator Agent with dimension scoring
	•	Langfuse client for complete workflow tracing
	•	Configuration management with environment variable support
	•	CLI commands for system management and operations

### 3.3 Tech Stack
	•	Language: Python 3.9+ (asyncio for concurrency)
	•	LLM Orchestration: LangChain (chains, agents, retrievers, vectorstores)
	•	Vector Store: FAISS for similarity search over embeddings
	•	Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
	•	LLM Provider: OpenRouter (configurable via environment)
	•	Observability: Langfuse (traces, evals, metrics)

⸻

## 4. Implementation Details

### 4.1 Project Structure

The repo is organized for clarity and maintainability:
```text
src/
├── main.py                     # System entry point and CLI interface
├── config.py                   # Configuration management with env support
├── agents/
│   ├── base.py                 # Base agent interface
│   └── rag_agent.py            # RAG agent implementation
├── retrievers/
│   ├── base.py                 # Retriever interface
│   ├── faiss_retriever.py      # FAISS-based retriever
│   └── cached_faiss_retriever.py # Cached FAISS implementation
├── orchestrator/
│   ├── intent_classifier.py    # Semantic intent classification
│   └── orchestrator.py         # Routing & coordination
├── evaluator/
│   └── quality_evaluator.py    # Automated quality scoring
└── utils/
    ├── langfuse_client.py      # Langfuse observability integration
    └── mock_llm.py           # Mock LLM for testing

data/                           # Domain-specific documentation
├── hr_docs/                   # HR policies and benefits
│   ├── employee_benefits.md
│   └── workplace_policies.md
├── tech_docs/                 # IT support and development
│   ├── it_support.md
│   └── software_development.md
└── finance_docs/              # Financial procedures
    ├── expenses_and_budgets.md
    └── financial_policies.md

cache/                         # Runtime cache directory
store/                         # Persistent FAISS indices storage
├── faiss_indices/            # FAISS index files per department
├── embeddings/               # Pre-computed embeddings
└── metadata/                 # Document metadata

tests/                          # Test suite
└── test_integration.py        # Integration tests

test_queries.json              # Test queries in proper JSON format
.env.example                  # Environment variables template
```

This separation allows to evolve each layer (classification, retrieval, evaluation, observability) without touching core orchestration logic.

⸻

## 5. Intent Classification & Routing

### 5.1 Semantic Intent Classifier

The intent classifier uses Sentence Transformers to encode:
	•	Department “prototype” descriptions (HR, Tech, Finance)
	•	Incoming user queries

The bot computes cosine similarity between the query embedding and each department prototype, then pick the department with the highest score.

Key aspects:
	•	Configurable similarity top-k and confidence thresholds:
	•	SIMILARITY_TOP_K (default: 3)
	•	CONFIDENCE_THRESHOLD (0.7 for dev, 0.8+ for prod)
	•	Output includes:
	•	department
	•	classification_confidence

### 5.2 Conditional Routing

Once the intent is classified, the orchestrator performs conditional routing:
	•	hr      → HR Assistant (RAG agent)
	•	tech    → IT Support Assistant
	•	finance → Finance Assistant

If the confidence is below the configured threshold, the bot can:
	•	Route to a fallback agent (future extension), or
	•	Ask for clarification / flag for human review

In the example run:
```bash
python3 -m src.main query --query "What benefits am I entitled to as a new employee?"
```

The system routes the query to the HR Assistant with logged confidence and processing time, and produces a structured HR benefits answer grounded in HR docs.

⸻

## 6. RAG Agents & Retrieval

### 6.1 Per-Department RAG Agents

Each department agent follows the same pattern:
	1.	Retriever
	•	FAISS-based vector store per department:
	•	store/faiss_indices/<department>/faiss.index
	•	store/embeddings/<department>/embeddings.npy
	•	Built using all-MiniLM-L6-v2 embeddings.
	2.	Prompt Template
	•	Department-specific system prompt:
	•	Use only provided context.
	•	If the answer is not in the docs, say so explicitly.
	•	Use department terminology and policy tone.
	3.	LLM Chain
	•	A LangChain RAG-style chain that:
	•	Receives the user query
	•	Calls the retriever (top_k configurable)
	•	Injects retrieved chunks into the prompt
	•	Calls the LLM to generate a grounded answer

### 6.2 Example Answer (HR Assistant)

For the benefits query, the HR agent:
	•	Retrieves the employee benefits and PTO policy docs.
	•	Produces a structured markdown answer including:
	•	Health Insurance
	•	PTO
	•	Holidays
	•	Retirement Benefits
	•	Compensation
	•	Training & Development
	•	Remote Work
	•	Required Training

The console output includes:
	•	Department: hr
	•	Agent: HR Assistant
	•	Confidence: 0.579
	•	Processing Time: ~11s
	•	Quality Score: 9.3/10 (from the evaluator)

### 6.3 Persistent Storage & Caching System

A key production feature is the dual-layer storage system:

#### Persistent Storage (./store/)
- **FAISS Indices**: `store/faiss_indices/<dept>/faiss.index` - Pre-built vector indices per department
- **Embeddings**: `store/embeddings/<dept>/embeddings.npy` - Pre-computed document embeddings
- **Metadata**: `store/metadata/<dept>/` - Document metadata and chunk information
- **Benefit**: Instant system startup without reprocessing documents

#### Runtime Cache (./cache/)
- **In-Memory Caching**: `CachedFAISSRetriever` provides fast access to frequently accessed data
- **Cache Management**: CLI commands for cache inspection and clearing
- **Department-Specific**: Can clear cache for individual departments or entire system

#### Storage Management Commands
```bash
# View persistent store status
python3 -m src.main store-info

# Clear persistent data
python3 -m src.main store-clear --department hr

# Cache management
python3 -m src.main cache-info
python3 -m src.main cache-clear
```

This dual approach ensures both **fast startup** (persistent storage) and **runtime performance** (caching).

⸻

## 7. Observability with Langfuse

### 7.1 Why Langfuse

Given the complexity of multi-step LLM workflows, I wanted deep observability: traces of each decision, retrieval call, and model output. Langfuse is an open-source LLM engineering platform that provides tracing, evaluations, metrics, and prompt management tailored to LLM apps.

### 7.2 Tracing Model

For each query, the bot creates a Langfuse trace:
	•	Trace name: multi_agent_query_processing
	•	Events:
	•	intent_classification
	•	Spans:
	•	rag_retrieval
	•	<Department> Assistant_execution
	•	Metadata:
	•	department
	•	classification confidence
	•	processing time
	•	quality score (if evaluated)

Sample log (simplified):
```text
📝 [Langfuse] Creating trace: multi_agent_query_processing
📅 [Langfuse] Creating event: intent_classification
📊 [Langfuse] Creating span: rag_retrieval
📊 [Langfuse] Creating span: HR Assistant_execution
🔄 [Langfuse] Updating trace with: ['output', 'metadata']
```

This makes it easy to debug:
	•	Misclassifications (wrong department)
	•	Poor retrieval (irrelevant docs)
	•	Hallucinations or low-quality answers

### 7.3 Tracing example

![langfuse example][./docs/langfuse.png]

⸻

## 8. Quality Evaluation (Bonus)

### 8.1 Evaluator Agent

As a bonus, I added an Evaluator Agent that runs after the main RAG answer is generated.

It takes as input:
	•	Original user query
	•	Final answer
	•	(Optionally) retrieved context

It outputs:
	•	overall_score (1–10)
	•	dimension_scores for:
	•	relevance
	•	completeness
	•	accuracy
	•	reasoning
	•	recommendations

These scores are attached to the same Langfuse trace, so I can analyze quality over time, per department, or per query type.

### 8.2 Example Evaluation

For the benefits query, the evaluator produced:
	•	Overall Score: 9.3/10
	•	Dimension Scores:
	•	relevance: 9/10
	•	completeness: 10/10
	•	accuracy: 9/10

This gives me an automated, consistent signal about response quality, and a foundation for future human-in-the-loop review workflows.

⸻

## 9. Usage

The system provides a comprehensive CLI interface for production management:

### Core Commands
```bash
# Initialize the system (load indices, warm up agents)
python3 -m src.main init

# Initialize with force rebuild (useful after document updates)
python3 -m src.main init --force-rebuild

# Initialize without persistent storage (cache-only mode)
python3 -m src.main init --no-persistent

# Process a single query
python3 -m src.main query --query "What benefits am I entitled to as a new employee?"

# Query with user ID for better tracing
python3 -m src.main query --query "How do I reset my password?" --user-id "john.doe"

# Query without quality evaluation (faster processing)
python3 -m src.main query --query "What's our budget?" --no-evaluation

# Run a test suite of queries
python3 -m src.main test --file test_queries.json

# Run tests without quality evaluation (faster)
python3 -m src.main test --file test_queries.json --no-evaluation

# Check system status and component health
python3 -m src.main status
```

### Storage Management Commands
```bash
# View persistent store information
python3 -m src.main store-info

# Clear all persistent stores
python3 -m src.main store-clear

# Clear specific department store
python3 -m src.main store-clear --department hr

# View cache information
python3 -m src.main cache-info

# Clear all caches
python3 -m src.main cache-clear

# Clear specific department cache
python3 -m src.main cache-clear --department tech
```

### Available Flags
- `--force-rebuild`: Force rebuild all FAISS indices
- `--no-persistent`: Disable persistent storage (cache-only mode)
- `--no-evaluation`: Skip quality evaluation for faster processing
- `--user-id <id>`: User ID for better Langfuse tracing
- `--department <dept>`: Target department for cache/store operations (hr, tech, finance)
- `--file <path>`: Test queries file (default: test_queries.json)

⸻

## 10. Results & Metrics

Latest Test Results (from 19 test queries)
	•	Classification Accuracy: 68.4%
	•	Average Confidence: 0.44
	•	Average Processing Time: 4.9s
	•	Average Quality Score: 7.3/10
	•	Total Test Queries: 19 (7 HR, 8 Tech, 4 Finance)

These numbers are based on a comprehensive test set of 19 real-world HR, Tech, and Finance queries. The classification accuracy reflects the current intent classifier performance with semantic similarity, while the quality scores demonstrate strong response generation despite moderate classification confidence.

⸻

## 11. Technical Decisions

### 11.1 Why LangChain

I chose LangChain instead of custom orchestration for several reasons:

	•	Recommendation from instructor to try a state of the art framework
	•	Ready-made abstractions for chains, agents, and retrievers
	•	Cleaner composition of multi-step workflows
	•	Easier to swap models, retrievers, and tools as the system grows

### 11.2 Why FAISS + Sentence Transformers
	•	FAISS gives me fast similarity search over dense vectors, suitable for mid-sized documentation.
	•	all-MiniLM-L6-v2 offers compact, high-quality sentence embeddings with good performance/latency trade-offs.
	•	Together they provide robust semantic retrieval without overcomplicating the infrastructure.

### 11.3 Why Semantic Classification (Not Rules)
	•	Handles paraphrases and non-standard phrasing better than keyword rules.
	•	Easy to extend: to add a department, I just add more prototype examples and documents.
	•	Produces confidence scores, which I can use to control routing and escalation logic.

### 11.4 Why Langfuse for Observability
	•	Purpose-built for LLM observability, tracing and evaluation, not just generic logging.
	•	Direct integration with Python and LangChain.
	•	Gives me a clear view of each query's journey through the multi-agent pipeline.

### 11.5 Why Persistent Storage & Caching
	•	Production Performance: Instant system startup with pre-built FAISS indices and embeddings
	•	Scalability: Store large document collections without memory constraints
	•	Reliability: Persistent indices survive system restarts and crashes
	•	Development Efficiency: Skip document reprocessing during development and testing
	•	Operational Management: CLI commands for cache/store inspection and maintenance

⸻

## 12. Limitations & Future Work

Current limitations:
	•	Only three departments implemented (HR, Tech, Finance).
	•	Small document sets per department (good for demo, but not production-sized).
	•	No explicit escalation to human agents yet when confidence or quality is low.

Planned improvements:
	•	Add Legal and additional specialized agents.
	•	Introduce a fallback “generalist” agent for ambiguous queries.
	•	Integrate with a real ticketing system (e.g., Zendesk/Freshdesk) for end-to-end routing.
	•	Expand test datasets and add continuous evaluation pipelines.

⸻

## 13. Conclusion

Henry Bot M3 is my implementation of a multi-agent intelligent routing system that:
	•	Classifies user intent with semantic similarity
	•	Routes queries to specialized, department-specific RAG agents
	•	Grounds answers in internal documentation
	•	Provides full observability and automated quality evaluation via Langfuse
	•	Follows LangChain-based, production-ready patterns instead of fragile one-off scripts