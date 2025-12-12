import os
import json
import re
from langdetect import detect
import spacy
from tabulate import tabulate
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from llm import OllamaLLM
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================
#  Load SpaCy Models (Support English & Vietnamese)
# ============================================================

nlp_en = spacy.load("en_core_web_lg")
nlp_vi = spacy.load("vi_core_news_lg")


EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF]+",
    flags=re.UNICODE,
)

# Noise keyword, advertiser
PR_KEYWORDS = [
    "miễn phí", "ưu đãi", "bạn muốn", "đăng ký", "sở hữu",
    "cơ hội", "giảm giá", "nhanh chóng", "chỉ còn hôm nay",
    "nhận ngay", "đặt mua", "gọi ngay",
]


class OntologyStore:
    """
    Stores canonical entities and synonyms detected from the domain LLM.
    Relations are stored at scoring level.
    """
    def __init__(self):
        self.canonical_entities = set()
        self.synonyms = {}  # { "long name": "canonical entity" }

    def register_entities(self, entities):
        """
        Register canonical entities and their synonyms.
        """
        for e in entities:
            if isinstance(e, str) and e.strip():
                canonical = e.strip()
                self.canonical_entities.add(canonical)

    def register_synonyms(self, synonyms):
        """
        Register synonyms for canonical entities.
        """
        for k, v in synonyms.items():
            if isinstance(k, str) and isinstance(v, str):
                k, v = k.strip(), v.strip()
                if v in self.canonical_entities:
                    self.synonyms[k] = v

    def normalize_entities(self, raw_list):
        """
        Normalize entities by removing duplicates and synonyms.
        """
        normalized = []
        for item in raw_list:
            if isinstance(item, dict):
                candidate = item.get("name") or item.get("value") or None
                if candidate:
                    item_norm = candidate.strip()
                else:
                    continue
            elif isinstance(item, str):
                item_norm = item.strip()
            else:
                continue

            if item_norm in self.synonyms:
                normalized.append(self.synonyms[item_norm])
            else:
                normalized.append(item_norm)

        # remove duplicates, preserve order
        return list(dict.fromkeys(normalized))


ONTOLOGY = OntologyStore()


def detect_language(text):
    try:
        lang = detect(text)
    except Exception:
        return "en"
    return "vi" if lang.startswith("vi") else "en"


def semantic_chunking(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ":"],
    )
    return splitter.split_documents(documents)


def negative_pr_score(text):
    """ Penalizes PR-oriented and emoji-heavy content """
    score = 0.0
    if EMOJI_PATTERN.search(text):
        score -= 0.15
    lower = text.lower()
    if any(kw in lower for kw in PR_KEYWORDS):
        score -= 0.20
    return score


def tfidf_scores(chunks):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([c.page_content for c in chunks])
    return matrix.mean(axis=1)


def centrality_score(chunks):
    """ Simple structural graph centrality scoring between chunks """
    G = nx.Graph()
    for i in range(len(chunks) - 1):
        G.add_edge(i, i + 1)
    return nx.pagerank(G)

def batch_chunks(chunks, max_chars=3500):
    """ Group chunks so each batch fits the local LLM context window """
    batches, batch, current_len = [], [], 0

    for idx, ch in enumerate(chunks):
        length = len(ch.page_content)
        if batch and current_len + length > max_chars:
            batches.append(batch)
            batch = [idx]
            current_len = length
        else:
            batch.append(idx)
            current_len += length

    if batch:
        batches.append(batch)
    return batches


def call_llm(domain_llm, prompt):
    """ Abstracts domain LLM invocation """
    if hasattr(domain_llm, "invoke"):
        return domain_llm.invoke(prompt)
    if callable(domain_llm):
        return domain_llm(prompt)
    raise ValueError("Invalid domain_llm provided")


def extract_json(text):
    """ Extract JSON block from LLM output without generating synthetic data """
    text = str(text)  # Ensure text is string
    
    # Try to find JSON blocks with different strategies
    strategies = [
        lambda t: (t.find("{"), t.rfind("}")),
        lambda t: find_complete_json_block(t)
    ]
    
    for strategy in strategies:
        try:
            start, end = strategy(text)
            if start != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except:
            continue
    
    raise ValueError("No valid JSON object found")

def find_complete_json_block(text):
    """ Find the first complete JSON block by counting braces """
    start = text.find("{")
    if start == -1:
        return -1, -1
    
    brace_count = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return start, i
    
    return -1, -1


def llm_domain_refinement(domain_llm, chunks, ontology: OntologyStore, max_batches=None):
    """
    Iteratively refine ontology:
    - Process batches with incremental ontology info
    - Normalize synonyms
    - Update canonical entity store
    """
    results = {i: {"entities": [], "relations": []} for i in range(len(chunks))}
    batches = batch_chunks(chunks)
    if max_batches is not None:
        batches = batches[:max_batches]

    for batch_idx in batches:
        prompt_parts = [
            "You are a domain analysis assistant.",
            "Use known canonical entities and synonyms to maintain consistency.",
            "If a new entity matches an existing one, reuse canonical naming.",
            "", "Known canonical entities:",
        ]
        for e in ontology.canonical_entities:
            prompt_parts.append(f"- {e}")
        prompt_parts.append("\nText Segments:\n")

        for idx in batch_idx:
            prompt_parts.append(f"ID: {idx}\nTEXT:\n{chunks[idx].page_content}\n---")

        prompt_parts.append(
            'Return only valid JSON:\n'
            '{\n'
            '  "chunks": [\n'
            '    {"id": <int>, "entities": [...], "relations": [...]}'
            '  ],\n'
            '  "synonyms": {"alt_name": "canonical", ...}\n'
            '}'
        )

        prompt = "\n".join(prompt_parts)
        output = call_llm(domain_llm, prompt)

        try:
            data = extract_json(output)
            print(f"Successfully parsed LLM response for batch {batch_idx}")
        except Exception as e:
            print(f"Failed to parse LLM response for batch {batch_idx}: {e}")
            print(f"LLM output: {str(output)[:500]}...")
            continue

        for item in data.get("chunks", []):
            try:
                cid = item.get("id")
                if cid is None:
                    continue
                    
                ents = item.get("entities", [])
                relations = item.get("relations", [])
                
                if not isinstance(ents, list):
                    ents = []
                    
                if not isinstance(relations, list):
                    relations = []
                
                normalized_ents = ontology.normalize_entities(ents)
                ontology.register_entities(normalized_ents)
            except Exception as e:
                print(f"Error processing chunk item {item}: {e}")
                continue
        ontology.register_synonyms(data.get("synonyms", {}))

        for item in data.get("chunks", []):
            try:
                cid = item.get("id")
                if cid is None or cid not in results:
                    continue
                    
                raw_ents = item.get("entities", [])
                relations = item.get("relations", [])
                
                if not isinstance(raw_ents, list):
                    raw_ents = []
                if not isinstance(relations, list):
                    relations = []

                normalized_ents = ontology.normalize_entities(raw_ents)
                results[cid]["entities"].extend(normalized_ents)
                results[cid]["relations"].extend(relations)
            except Exception as e:
                print(f"Error processing result item {item}: {e}")
                continue

    for idx, info in results.items():
        info["entities"] = list(dict.fromkeys(info["entities"]))
        unique_relations = []
        seen_relations = set()
        for rel in info["relations"]:
            if isinstance(rel, dict):
                rel_str = json.dumps(rel, sort_keys=True)
            else:
                rel_str = str(rel)
            
            if rel_str not in seen_relations:
                seen_relations.add(rel_str)
                unique_relations.append(rel)
        
        info["relations"] = unique_relations

    return results

def score_chunks(chunks, entity_relation_info):
    """
    Scoring uses:
    - LLM entity density
    - LLM relation density
    - TF-IDF distribution
    - Chunk centrality
    - Anti-PR penalties
    """
    if not chunks:
        return []

    tfidf = tfidf_scores(chunks)
    centrality = centrality_score(chunks)
    scored = []

    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        info = entity_relation_info.get(i, {"entities": [], "relations": []})
        ents = info["entities"]
        rels = info["relations"]

        token_count = max(1, len(text.split()))
        e_score = len(ents) / token_count
        r_score = len(rels) / max(1, len(ents)) if ents else 0.0

        pr_penalty = negative_pr_score(text)

        final_score = (
            0.35 * e_score +
            0.25 * float(tfidf[i, 0]) +
            0.15 * centrality[i] +
            0.15 * r_score +
            pr_penalty
        )

        scored.append({
            "chunk": chunk,
            "score": float(final_score),
            "entities": ents,
            "relations": rels,
            "lang": detect_language(text)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def print_scored(scored, title, limit=8):
    print(f"\n{title}")
    rows = []
    for item in scored[:min(limit, len(scored))]:
        preview = re.sub(r"\s+", " ", item["chunk"].page_content[:150])
        relation_strs = []
        for rel in item["relations"][:3]:
            if isinstance(rel, dict):
                rel_str = rel.get("type", rel.get("relation", str(rel)))
            else:
                rel_str = str(rel)
            relation_strs.append(rel_str)
        
        rows.append([
            item["lang"], round(item["score"], 3),
            ", ".join(item["entities"][:3]) or "-",
            ", ".join(relation_strs) or "-",
            preview + "..."
        ])
    print(tabulate(rows, headers=["Lang", "Score", "Entities", "Relations", "Content"], tablefmt="grid"))


def filter_chunks(scored, ratio=0.4):
    keep_n = max(1, int(len(scored) * ratio))
    return [s["chunk"] for s in scored[:keep_n]]

def ingest_file_to_graph(file_path, graph_db, llm_transformer, domain_llm=None):
    """
    Full ingest pipeline:
    - Chunk text
    - Iterative domain refinement (if domain_llm provided)
    - Chunk scoring & ranking
    - Filter reduced chunks
    - Final graph extraction (if Neo4j + LLM provided)
    """
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        print("Unsupported file format:", ext)
        return

    documents = loader.load()
    chunks = semantic_chunking(documents)

    print("Running iterative domain refinement...")
    entity_relation_info = {}
    if domain_llm:
        entity_relation_info = llm_domain_refinement(domain_llm, chunks, ONTOLOGY)
    else:
        print("domain_llm is None: skipping domain refinement")

    print("Scoring chunks...")
    scored = score_chunks(chunks, entity_relation_info)
    print_scored(scored, "Top ranked chunks")

    filtered = filter_chunks(scored, ratio=0.45)
    print(f"Reduced chunks: {len(chunks)} → {len(filtered)}")

    if llm_transformer and graph_db:
        graph_docs = llm_transformer.convert_to_graph_documents(filtered)
        graph_db.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
        print("Graph import completed.")
    else:
        print("Skipping final graph extraction (Neo4j or Graph LLM missing)")
        # Save to JSON
        with open("output.json", "w") as f:
            # Convert Document objects to dicts for JSON serialization
            filtered_for_json = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in filtered]
            json.dump(filtered_for_json, f, indent=2, ensure_ascii=False) # For Vietnamese characters


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    args = parser.parse_args()
    
    domain_llm = OllamaLLM(model_name="qwen2.5:3b-instruct", stream=False)

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("OPENAI_MODEL"),
        temperature=0,
        timeout=120,
        max_retries=3,
    )

    # 4. Initialize Neo4j Connection
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

    ingest_file_to_graph(
        file_path=args.file_path,
        graph_db=graph_db,
        llm_transformer=llm_transformer,
        domain_llm=domain_llm
    )