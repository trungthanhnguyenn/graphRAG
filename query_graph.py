
import os
import argparse
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain

# Load environment variables
load_dotenv()

def query_graph(question):
    """
    Queries the Neo4j graph using a GraphCypherQAChain.
    """
    
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0
    )
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("OPENAI_BASE_URL"))
    print(os.getenv("OPENAI_MODEL"))

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Missing Neo4j configuration. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file.")

    graph_db = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )
    
    graph_db.refresh_schema()

    from preprocessor import QueryPreprocessor
    preprocessor = QueryPreprocessor(graph_db)
    
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    print("Extracting entities...")
    entities = preprocessor.extract_entities(question)
    print(f"Found entities: {entities}")
    
    cypher_query = preprocessor.generate_cypher(question, entities)
    print(f"Generated Cypher: {cypher_query}")
    
    try:
        context = graph_db.query(cypher_query)
        
        if not context:
            print("No information found in the knowledge graph.")
            return

        from langchain.prompts import PromptTemplate
        
        ANSWER_PROMPT = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an assistant answering questions based on the provided Knowledge Graph context.
Context:
{context}

Question: {question}

Answer (in Vietnamese):"""
        )
        
        chain = ANSWER_PROMPT | llm
        response = chain.invoke({"question": question, "context": context})
        
        print("\nFinal Answer:")
        print(response.content)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Knowledge Graph")
    parser.add_argument("question", type=str, nargs="?", help="The question to ask the graph")
    args = parser.parse_args()

    if args.question:
        query_graph(args.question)
    else:
        print("Entering interactive mode. Type 'exit' to quit.")
        while True:
            q = input("\nEnter your question: ")
            if q.lower() in ["exit", "quit"]:
                break
            if q.strip():
                query_graph(q)
