
# GraphRAG Implementation

This project implements a Graph Retrieval-Augmented Generation (GraphRAG) system that extracts entities and relationships from documents to build a knowledge graph, enabling complex query answering.

## 🚀 Features

- **Hybrid Processing**: Uses local LLM (Ollama) for initial filtering and cost optimization, then powerful Cloud LLM (OpenAI) for precise graph extraction.
- **Smart Data Ingestion**:
  - Semantic Chunking
  - Domain-specific entity refinement
  - Noise filtering
  - Scoring & Ranking to select top-quality chunks
- **Graph Database**: Neo4j integration for storing and querying complex relationships.
- **Advanced Querying**:
  - Semantic/Fuzzy entity matching using Vector Embeddings.
  - Deterministic Cypher query generation (prevents hallucination).
  - LLM-synthesized answers based on graph context.

## 🛠 Prerequisites

- Python 3.10+
- Docker & Docker Compose
- [Ollama](https://ollama.com/) running locally with `qwen2.5:3b-instruct` model.
- OpenAI API Key (or compatible service).

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/trungthanhnguyenn/graphRAG.git
   cd graph-RAG
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables**
   Copy `.example.env` to `.env` and update your keys:
   ```bash
   cp .example.env .env
   ```
   *Update `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and ensure `OPENAI_MODEL` is set correctly.*

4. **Pull Ollama Model**
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

## 🚀 Usage

### One-Click Start
Use the included script to handle everything (Start DB -> Ingest -> Query):

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh path/to/your/document.docx
```

### Manual Steps

1. **Start Neo4j**
   ```bash
   docker compose up -d
   ```

2. **Ingest Data**
   ```bash
   python dataprocess.py path/to/your/document.docx
   ```

3. **Query the Graph**
   ```bash
   python query_graph.py
   ```

## 📂 Project Structure

- `dataprocess.py`: Main ingestion pipeline (Load -> Chunk -> Filter -> Graph Extract -> Neo4j).
- `preprocessor.py`: Semantic search & Cypher query generation logic.
- `query_graph.py`: CLI tool to ask questions to the Graph.
- `run_pipeline.sh`: Automation script.
- `docker-compose.yml`: Neo4j configuration.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[Apache License 2.0](LICENSE)
