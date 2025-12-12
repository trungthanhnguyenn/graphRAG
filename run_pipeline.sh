#!/bin/bash

# Configuration
# Default input file if not provided
DEFAULT_INPUT_FILE="test.docx"

# Function to print messages
log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

# Check for Input File
INPUT_FILE="${1:-$DEFAULT_INPUT_FILE}"

if [ ! -f "$INPUT_FILE" ]; then
    log_error "Input file not found: $INPUT_FILE"
    echo "Usage: ./script.sh [input_file_path]"
    echo "Example: ./script.sh my_document.docx"
    exit 1
fi

log_info "Using input file: $INPUT_FILE"

# Check Docker Environment (Neo4j)
log_info "Checking Docker containers..."
if docker ps | grep -q "neo4j_graphrag"; then
    log_info "Neo4j container is running."
else
    # Create Neo4j directories if they don't exist
    echo "Setting up Neo4j directories..."
    mkdir -p neo4j_data neo4j_logs
    log_warn "Neo4j container is NOT running."
    log_info "Attempting to start services..."
    docker compose up -d
    
    if [ $? -ne 0 ]; then
        log_error "Failed to start Docker containers. Please check docker-compose.yml and Docker status."
        exit 1
    fi
    
    log_info "Waiting for Neo4j to be ready (30s)..."
    sleep 30
fi

# Run Data Ingestion (dataprocess.py)
log_info "Starting data ingestion pipeline..."
python dataprocess.py "$INPUT_FILE"

if [ $? -eq 0 ]; then
    log_info "Data ingestion completed successfully."
else
    log_error "Data ingestion failed!"
    echo "Possible reasons:"
    echo "- Missing python dependencies (run: pip install -r requirements.txt)"
    echo "- Neo4j connection refused (check localhost:7474)"
    echo "- OpenAI API Key invalid (check .env file)"
    exit 1
fi

# Run Graph Query (Interactive or Automatic Test)
log_info "Starting Graph Query Interactive Mode..."
echo "---------------------------------------------------"
echo "You can now ask questions about your document."
echo "Type 'exit' to quit."
echo "---------------------------------------------------"

python query_graph.py

if [ $? -ne 0 ]; then
    log_error "Query script exited with error."
else
    log_info "Session ended successfully."
fi
