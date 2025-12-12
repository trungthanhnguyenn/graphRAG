import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_NAME = 'keepitreal/vietnamese-sbert'
embedder = SentenceTransformer(MODEL_NAME)

class QueryPreprocessor:
    def __init__(self, graph_db):
        self.graph = graph_db
        self.entity_cache = self._fetch_all_entities()
        if self.entity_cache:
            self.entity_embeddings = embedder.encode(self.entity_cache)
        else:
            self.entity_embeddings = None

    def _fetch_all_entities(self):
        """Fetches all unique node IDs or Names from Neo4j."""
        query = "MATCH (n) RETURN DISTINCT toString(n.id) AS name"
        try:
            result = self.graph.query(query)
            entities = [record['name'] for record in result if record['name']]
            return list(set(entities))
        except Exception as e:
            print(f"Error fetching entities: {e}")
            return []

    def extract_entities(self, question, threshold=0.7):
        """
        Extracts potential entities from the question using vector similarity.
        Returns a list of matched entities from the database.
        """
        if not self.entity_cache or self.entity_embeddings is None:
            return []

        words = question.split()
        candidates = []
        for n in range(1, 4): # n-grams from 1 to 3
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i+n])
                candidates.append(gram)
        
        if not candidates:
            return []

        # Encode candidates
        candidate_embeddings = embedder.encode(candidates)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(candidate_embeddings, self.entity_embeddings)
        
        matches = set()
        rows, cols = np.where(similarities > threshold)
        
        for r, c in zip(rows, cols):
            matches.add(self.entity_cache[c])
            
        return list(matches)

    def generate_cypher(self, question, entities):
        """
        Generates a Cypher query dynamically based on detected entities.
        This avoids LLM hallucinations for the exact Cypher syntax.
        """
        if not entities:
            clean_q = re.sub(r'[^\w\s]', '', question).split()
            keywords = [w for w in clean_q if len(w) > 2]
            
            if not keywords:
                 return "MATCH (n) RETURN n LIMIT 10"

            conditions = [f"toLower(toString(n.id)) CONTAINS toLower('{kw}')" for kw in keywords]
            where_clause = " OR ".join(conditions)
            
            return f"""
            MATCH (n)
            WHERE {where_clause}
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m LIMIT 20
            """
            
        entity_conditions = [f"n.id = '{e}'" for e in entities]
        where_clause = " OR ".join(entity_conditions)
        
        return f"""
        MATCH (n)
        WHERE {where_clause}
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m LIMIT 50
        """
