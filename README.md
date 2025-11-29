# 🤖 Henry Bot M3 - A Multi-Agent Intelligent Routing System 

A multi-agent orchestration system that automatically classifies incoming questions by department and routes them to specialized AI agents for accurate, context-aware answers using company documentation.

The evaluation report can be found at [REPORT.md](./REPORT.md)

## 🌟 Key Features

- **🎯 Intelligent Intent Classification**: Automatically categorizes queries into HR, Tech, or Finance departments
- **🔗 Specialized RAG Agents**: Domain-specific retrieval-augmented generation for accurate, context-aware responses
- **📊 Comprehensive Observability**: Complete workflow tracing with Langfuse for debugging and monitoring
- **🧪 Automated Quality Evaluation**: Response quality scoring across relevance, completeness, and accuracy dimensions
- **🔧 Production-Ready Architecture**: Modular design using LangChain components with clean separation of concerns
- **📈 Real Performance Metrics**: Classification accuracy, confidence scores, processing time tracking

## 🏗️ System Architecture

```
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

## 📋 Technical Implementation

### Core Components

1. **Intent Classifier**: Uses Sentence Transformers for semantic similarity-based classification
2. **Specialized RAG Agents**: Department-specific retrievers with FAISS vector storage
3. **Persistent Storage**: FAISS indices, embeddings, and metadata stored in `./store/` for fast startup
4. **Caching System**: In-memory caching with optional cache management commands
5. **Quality Evaluator**: LLM-based response scoring across multiple dimensions
6. **Langfuse Integration**: Complete workflow tracing and observability
7. **CLI Interface**: Full command-line interface for system management and testing

### Technologies Used

- **Backend**: Python 3.9+ with asyncio
- **LLM Integration**: OpenRouter API with multiple model support
- **RAG System**: FAISS + Sentence Transformers embeddings
- **Observability**: Langfuse for tracing and monitoring
- **Architecture**: LangChain for production-grade components

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenRouter API key (free tier available)
- Langfuse credentials (optional, for observability)

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

```bash
# Required: OpenRouter API key
OPENROUTER_API_KEY=your-openrouter-api-key-here

# Optional: Langfuse for observability
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Model and system configuration
MODEL_NAME=google/gemini-2.0-flash-exp:free
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_TOP_K=3
CONFIDENCE_THRESHOLD=0.7
```

## 💻 Usage

### Initialize the System

```bash
# Initialize system (load FAISS indices, warm up agents)
python3 -m src.main init

# Force rebuild all indices (useful after document updates)
python3 -m src.main init --force-rebuild

# Initialize without persistent storage (cache-only mode)
python3 -m src.main init --no-persistent
```

### Process Single Queries

```bash
# Basic query processing
python3 -m src.main query --query "What benefits am I entitled to as a new employee?"

# Query with user ID for better tracing
python3 -m src.main query --query "How do I reset my password?" --user-id "john.doe"

# Query without quality evaluation (faster processing)
python3 -m src.main query --query "What's our budget?" --no-evaluation
```

### Run Test Suite

```bash
# Run full test suite with quality evaluation
python3 -m src.main test --file test_queries.json

# Run tests without quality evaluation (faster)
python3 -m src.main test --file test_queries.json --no-evaluation
```

### System Management

```bash
# Check system status and available departments
python3 -m src.main status

# Cache management
python3 -m src.main cache-info                    # Show cache information
python3 -m src.main cache-clear                   # Clear all caches
python3 -m src.main cache-clear --department hr   # Clear specific department cache

# Persistent storage management
python3 -m src.main store-info                    # Show persistent store information
python3 -m src.main store-clear                   # Clear all persistent stores
python3 -m src.main store-clear --department tech # Clear specific department store
```

### Programmatic Usage

```python
import asyncio
from src.main import MultiAgentSystem

async def main():
    # Initialize system
    system = MultiAgentSystem()
    await system.initialize()

    # Process query
    response = await system.process_query(
        "How do I request time off?",
        user_id="user123",
        evaluate_quality=True
    )

    print(f"Department: {response['department']}")
    print(f"Answer: {response['answer']}")
    print(f"Quality Score: {response['quality_evaluation']['overall_score']}/10")

    # Shutdown
    await system.shutdown()

asyncio.run(main())
```

## 📊 Response Format

```json
{
  "query": "What benefits am I entitled to as a new employee?",
  "answer": "As a new employee, you are entitled to comprehensive health insurance...",
  "department": "hr",
  "agent_used": "HR Assistant",
  "confidence": 0.85,
  "classification_confidence": 0.92,
  "processing_time": 2.34,
  "source_documents": [
    {
      "content": "Employee benefits include health insurance...",
      "source": "employee_benefits.md",
      "similarity_score": 0.94,
      "metadata": {"file_name": "employee_benefits.md"}
    }
  ],
  "quality_evaluation": {
    "overall_score": 8.5,
    "dimension_scores": {
      "relevance": 9.0,
      "completeness": 8.0,
      "accuracy": 8.5
    },
    "reasoning": "The answer directly addresses the user's question...",
    "recommendations": ["Could include more specific enrollment deadlines"]
  }
}
```

## 📈 Performance Metrics

### Test Results (Latest Run)

- **Classification Accuracy**: 68.4%
- **Average Confidence**: 0.44
- **Average Processing Time**: 4.9 seconds
- **Average Quality Score**: 7.3/10

### Test Queries Format

The system expects test queries in the following JSON format:

```json
{
  "queries": [
    {
      "query": "What benefits am I entitled to as a new employee?",
      "expected_department": "hr",
      "user_id": "test_user_1"
    },
    {
      "query": "My laptop won't connect to the VPN",
      "expected_department": "tech",
      "user_id": "test_user_2"
    }
  ]
}
```

The `expected_department` field is used to calculate classification accuracy, while `user_id` helps with tracing in Langfuse.

### Monitoring with Langfuse

- Complete trace visibility for every query
- Agent performance metrics and error tracking
- Classification confidence and routing decisions
- Quality evaluation scores and recommendations

## 🧪 Testing

```bash
# Run integration tests
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_integration.py::TestMultiAgentIntegration::test_single_query_processing -v
```

## 📁 Project Structure

```
src/
├── main.py                     # System entry point and CLI interface
├── config.py                   # Configuration management with env support
├── agents/                     # Specialized RAG agents
│   ├── base.py                # Base agent interface
│   └── rag_agent.py           # RAG agent implementation
├── retrievers/                 # Document retrieval systems
│   ├── base.py                # Retriever interface
│   ├── faiss_retriever.py     # FAISS-based retrieval
│   └── cached_faiss_retriever.py # Cached FAISS implementation
├── orchestrator/               # Multi-agent coordination
│   ├── intent_classifier.py   # Semantic intent classification
│   └── orchestrator.py        # Main orchestrator logic
├── evaluator/                  # Response quality assessment
│   └── quality_evaluator.py   # Automated quality scoring
└── utils/                      # Utility modules
    ├── langfuse_client.py     # Langfuse observability integration
    └── mock_llm.py           # Mock LLM for testing

data/                           # Department documentation
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

## 🔧 Technical Decisions

### Why LangChain?

- **Production-Grade Components**: Proven, battle-tested implementations
- **Maintainability**: Industry-standard patterns and abstractions
- **Extensibility**: Easy to add new agents and retrieval methods
- **Community Support**: Active development and comprehensive documentation

### Why FAISS + Sentence Transformers?

- **Performance**: Sub-millisecond similarity search at scale
- **Accuracy**: State-of-the-art semantic embeddings
- **Efficiency**: Memory-optimized vector operations
- **Flexibility**: Supports various distance metrics and indexing strategies

### Why Semantic Intent Classification?

- **Accuracy**: Better than keyword-based approaches for nuanced queries
- **Adaptability**: Can handle variations in phrasing and terminology
- **Explainability**: Confidence scores provide routing transparency
- **Performance**: Fast inference with pre-computed embeddings

### Why Automated Quality Evaluation?

- **Consistency**: Objective scoring across all responses
- **Debugging**: Identifies areas for system improvement
- **Monitoring**: Tracks response quality over time
- **Automation**: Reduces manual review overhead

### Why Persistent Storage & Caching?

- **Performance**: Skip document processing on startup with pre-built indices
- **Reliability**: Persistent FAISS indices survive system restarts
- **Efficiency**: In-memory caching for frequently accessed embeddings
- **Scalability**: Store large embeddings and indices without memory constraints

## 🛡️ Production Considerations

### Scalability
- Asynchronous processing for concurrent query handling
- Efficient vector storage with FAISS indexing
- Configurable batch processing for high-volume scenarios

### Reliability
- Comprehensive error handling and fallback mechanisms
- Health check endpoints for monitoring
- Graceful degradation when components fail

### Security
- API key management and secure credential storage
- Input validation and sanitization
- No sensitive data logging

### Observability
- Complete workflow tracing with Langfuse
- Performance metrics and error tracking
- Classification confidence monitoring

## 🚀 Deployment

### Environment Variables for Production

```bash
# Production configuration
DEBUG=false
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_TOP_K=5
CONFIDENCE_THRESHOLD=0.8

# Persistent storage
USE_PERSISTENT_STORAGE=true
STORE_DIR=./store
CACHE_DIR=./cache

# Performance tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Available CLI Commands

The system provides a comprehensive CLI for production management:

- `init` - Initialize system with optional flags (`--force-rebuild`, `--no-persistent`)
- `query` - Process individual queries with tracing (`--user-id`, `--no-evaluation`)
- `test` - Run test suites for quality assurance (`--file`, `--no-evaluation`)
- `status` - System health check and component status
- `cache-info`/`cache-clear` - Cache management and debugging
- `store-info`/`store-clear` - Persistent storage management
- `--department` flag for department-specific operations

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.