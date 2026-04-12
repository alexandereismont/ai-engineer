"""
generate_data.py
================
Generates all datasets needed for the ai-engineer-learning project.
Run this script once before opening any notebook.

Usage:
    python data/generate_data.py

All files are written to data/raw/. The script is fully deterministic:
random seed 42 is set globally so every run produces identical output.
"""

import csv
import os
import random
from datetime import datetime, timedelta

from faker import Faker

SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def write_csv(filename, fieldnames, rows):
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  v  {filename:40s}  {len(rows):>5} rows  ->  {path}")


def generate_articles():
    categories = ["technology", "science", "health", "finance", "environment"]
    tag_pool = {
        "technology": ["AI", "cloud", "software", "hardware", "cybersecurity",
                       "blockchain", "mobile", "IoT", "open-source", "DevOps"],
        "science":    ["physics", "biology", "chemistry", "astronomy", "genetics",
                       "neuroscience", "climate", "research", "lab", "discovery"],
        "health":     ["nutrition", "fitness", "mental-health", "medicine",
                       "wellness", "vaccine", "diet", "sleep", "cardiology", "public-health"],
        "finance":    ["stocks", "crypto", "investing", "banking", "budget",
                       "insurance", "tax", "fintech", "real-estate", "retirement"],
        "environment":["climate-change", "renewable-energy", "conservation",
                       "wildlife", "ocean", "pollution", "sustainability",
                       "deforestation", "recycling", "carbon"],
    }
    rows = []
    for i in range(1, 201):
        cat = random.choice(categories)
        n_sentences = random.randint(3, 5)
        body = " ".join(fake.sentence() for _ in range(n_sentences))
        tags = random.sample(tag_pool[cat], k=random.randint(2, 4))
        rows.append({
            "id":       i,
            "title":    fake.sentence(nb_words=random.randint(5, 10)).rstrip("."),
            "body":     body,
            "category": cat,
            "tags":     "|".join(tags),
        })
    write_csv("articles.csv", ["id", "title", "body", "category", "tags"], rows)


def generate_products():
    categories = ["electronics", "clothing", "books", "sports", "home"]
    adjectives = ["Premium", "Deluxe", "Essential", "Pro", "Classic",
                  "Ultra", "Smart", "Compact", "Portable", "Advanced"]
    nouns = {
        "electronics": ["Laptop", "Headphones", "Tablet", "Keyboard", "Monitor",
                        "Speaker", "Webcam", "Charger", "SSD", "Router"],
        "clothing":    ["Jacket", "T-Shirt", "Jeans", "Sneakers", "Hoodie",
                        "Dress", "Shorts", "Boots", "Cap", "Socks"],
        "books":       ["Novel", "Handbook", "Guide", "Memoir", "Textbook",
                        "Anthology", "Journal", "Workbook", "Atlas", "Almanac"],
        "sports":      ["Yoga Mat", "Dumbbell", "Running Shoes", "Bicycle Helmet",
                        "Tennis Racket", "Water Bottle", "Jump Rope",
                        "Resistance Band", "Foam Roller", "Gym Bag"],
        "home":        ["Blender", "Coffee Maker", "Desk Lamp", "Pillow",
                        "Curtains", "Cutting Board", "Trash Can",
                        "Wall Clock", "Plant Pot", "Storage Bin"],
    }
    rows = []
    for i in range(1, 501):
        cat = random.choice(categories)
        name = f"{random.choice(adjectives)} {random.choice(nouns[cat])}"
        rows.append({
            "product_id":   i,
            "name":         name,
            "description":  fake.sentence(nb_words=random.randint(10, 18)),
            "category":     cat,
            "price":        round(random.uniform(4.99, 999.99), 2),
            "rating":       round(random.uniform(1.0, 5.0), 1),
            "review_count": random.randint(0, 4800),
        })
    write_csv("products.csv",
              ["product_id", "name", "description", "category", "price", "rating", "review_count"],
              rows)


def generate_employees():
    departments = ["engineering", "marketing", "sales", "HR", "finance"]
    salary_range = {
        "engineering": (75_000, 180_000),
        "marketing":   (55_000, 120_000),
        "sales":       (45_000, 130_000),
        "HR":          (50_000, 100_000),
        "finance":     (65_000, 150_000),
    }
    rows = []
    for i in range(1, 301):
        dept = random.choice(departments)
        lo, hi = salary_range[dept]
        rows.append({
            "employee_id":       i,
            "name":              fake.name(),
            "department":        dept,
            "salary":            round(random.uniform(lo, hi), 2),
            "hire_date":         fake.date_between(start_date="-15y", end_date="today").isoformat(),
            "city":              fake.city(),
            "performance_score": round(random.uniform(1.0, 5.0), 2),
        })
    write_csv("employees.csv",
              ["employee_id", "name", "department", "salary", "hire_date", "city", "performance_score"],
              rows)


def generate_support_tickets():
    statuses   = ["open", "closed", "in_progress"]
    priorities = ["low", "medium", "high", "critical"]
    subjects = [
        "Cannot log in to my account",
        "Password reset email not received",
        "Billing charge I don't recognise",
        "Feature X is not working as expected",
        "How do I export my data?",
        "Integration with third-party tool is broken",
        "Dashboard loads very slowly",
        "Error 500 when submitting the form",
        "My subscription was cancelled incorrectly",
        "Two-factor authentication not working",
        "I need to change the email on my account",
        "Data is not syncing between devices",
        "API rate limit being hit unexpectedly",
        "Missing invoice for last month",
        "How do I add a team member?",
        "Notifications are not arriving",
        "Search results are returning irrelevant items",
        "File upload fails for large files",
        "I forgot my username",
        "Need help migrating data from old account",
    ]
    body_templates = [
        "I have been trying to {action} but keep getting an error that says '{error}'. I have already tried {fix_attempt} but the problem persists. This is blocking my work. Happy to provide any additional information you need.",
        "Since the latest update, {feature} has stopped working correctly. Specifically, when I {action}, the system {bad_outcome}. This was working fine before the update on {date}. Please advise on how to resolve this.",
        "I noticed an unexpected charge of ${amount} on my account dated {date}. I do not recall authorising this payment. Could you either clarify what this charge is for or issue a refund? I have attached a screenshot.",
        "I am unable to {action} from my account. I have followed the documentation steps but the option does not appear in my dashboard. Is this feature available on my current plan? If not, please let me know what I need to upgrade to.",
        "The {feature} is returning incorrect results. For example, when I search for '{query}', it shows items that are completely unrelated. I have cleared my cache and tried on multiple browsers. Any guidance would be appreciated.",
    ]
    actions      = ["log in", "export my data", "connect the API", "submit the form",
                    "update my billing details", "add a new user", "run a report",
                    "access the settings page", "download my invoice", "reset my password"]
    errors       = ["Session expired", "Access denied", "500 Internal Server Error",
                    "Invalid credentials", "Request timeout", "Network error"]
    fix_attempts = ["clearing my browser cache", "using a different browser",
                    "restarting the application", "logging out and back in",
                    "disabling my VPN", "checking my network connection"]
    features     = ["the search filter", "the export function", "two-factor authentication",
                    "the notification centre", "the team management panel",
                    "the data sync feature", "the billing portal", "the API integration"]
    bad_outcomes = ["shows a blank page", "throws an error", "freezes indefinitely",
                    "logs me out immediately", "displays incorrect data"]
    queries      = ["project management", "Q4 report", "invoice #12345",
                    "user settings", "API documentation", "team members"]

    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(1, 401):
        status = random.choice(statuses)
        created_at = base_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        resolved_at = ""
        if status == "closed":
            resolved_at = (created_at + timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%dT%H:%M:%S")
        body = random.choice(body_templates).format(
            action=random.choice(actions),
            error=random.choice(errors),
            fix_attempt=random.choice(fix_attempts),
            feature=random.choice(features),
            bad_outcome=random.choice(bad_outcomes),
            date=fake.date_between(start_date="-6m", end_date="today").isoformat(),
            amount=round(random.uniform(9.99, 299.99), 2),
            query=random.choice(queries),
        )
        rows.append({
            "ticket_id":   f"TKT-{i:04d}",
            "subject":     random.choice(subjects),
            "body":        body,
            "status":      status,
            "priority":    random.choice(priorities),
            "created_at":  created_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "resolved_at": resolved_at,
        })
    write_csv("support_tickets.csv",
              ["ticket_id", "subject", "body", "status", "priority", "created_at", "resolved_at"],
              rows)


def generate_knowledge_graph():
    nodes = [
        {"id": "embedding",           "label": "Embedding",            "category": "representation"},
        {"id": "transformer",          "label": "Transformer",           "category": "architecture"},
        {"id": "attention",            "label": "Attention Mechanism",   "category": "architecture"},
        {"id": "llm",                  "label": "Large Language Model",  "category": "model"},
        {"id": "neural_network",       "label": "Neural Network",        "category": "model"},
        {"id": "fine_tuning",          "label": "Fine-Tuning",           "category": "technique"},
        {"id": "rlhf",                 "label": "RLHF",                  "category": "technique"},
        {"id": "tokenization",         "label": "Tokenization",          "category": "preprocessing"},
        {"id": "prompt_engineering",   "label": "Prompt Engineering",    "category": "technique"},
        {"id": "few_shot",             "label": "Few-Shot Learning",     "category": "technique"},
        {"id": "zero_shot",            "label": "Zero-Shot Learning",    "category": "technique"},
        {"id": "chain_of_thought",     "label": "Chain-of-Thought",      "category": "technique"},
        {"id": "rag",                  "label": "RAG",                   "category": "architecture"},
        {"id": "vector_database",      "label": "Vector Database",       "category": "infrastructure"},
        {"id": "semantic_search",      "label": "Semantic Search",       "category": "retrieval"},
        {"id": "bm25",                 "label": "BM25",                  "category": "retrieval"},
        {"id": "hybrid_search",        "label": "Hybrid Search",         "category": "retrieval"},
        {"id": "reranking",            "label": "Re-Ranking",            "category": "retrieval"},
        {"id": "chunking",             "label": "Chunking",              "category": "preprocessing"},
        {"id": "hyde",                 "label": "HyDE",                  "category": "technique"},
        {"id": "agent",                "label": "AI Agent",              "category": "architecture"},
        {"id": "react",                "label": "ReAct",                 "category": "technique"},
        {"id": "tool_use",             "label": "Tool Use",              "category": "capability"},
        {"id": "memory",               "label": "Agent Memory",          "category": "capability"},
        {"id": "planning",             "label": "Planning",              "category": "capability"},
        {"id": "langchain",            "label": "LangChain",             "category": "framework"},
        {"id": "chromadb",             "label": "ChromaDB",              "category": "infrastructure"},
        {"id": "ollama",               "label": "Ollama",                "category": "infrastructure"},
        {"id": "sentence_transformers","label": "Sentence Transformers", "category": "library"},
        {"id": "dbt",                  "label": "dbt",                   "category": "framework"},
        {"id": "cosine_similarity",    "label": "Cosine Similarity",     "category": "metric"},
        {"id": "hallucination",        "label": "Hallucination",         "category": "problem"},
        {"id": "context_window",       "label": "Context Window",        "category": "constraint"},
        {"id": "ragas",                "label": "RAGAS",                 "category": "evaluation"},
        {"id": "faithfulness",         "label": "Faithfulness",          "category": "metric"},
        {"id": "relevance",            "label": "Answer Relevance",      "category": "metric"},
        {"id": "knowledge_graph",      "label": "Knowledge Graph",       "category": "data_structure"},
        {"id": "neo4j",                "label": "Neo4j",                 "category": "infrastructure"},
        {"id": "graph_rag",            "label": "Graph RAG",             "category": "architecture"},
        {"id": "entity",               "label": "Entity",                "category": "concept"},
        {"id": "relation",             "label": "Relation",              "category": "concept"},
        {"id": "data_pipeline",        "label": "Data Pipeline",         "category": "infrastructure"},
        {"id": "prefect",              "label": "Prefect",               "category": "framework"},
        {"id": "duckdb",               "label": "DuckDB",                "category": "infrastructure"},
        {"id": "etl",                  "label": "ETL",                   "category": "pattern"},
        {"id": "dense_vector",         "label": "Dense Vector",          "category": "representation"},
        {"id": "sparse_vector",        "label": "Sparse Vector",         "category": "representation"},
        {"id": "cross_encoder",        "label": "Cross-Encoder",         "category": "model"},
        {"id": "bi_encoder",           "label": "Bi-Encoder",            "category": "model"},
        {"id": "mmr",                  "label": "MMR",                   "category": "retrieval"},
    ]
    edges = [
        ("transformer","ENABLES","llm"),
        ("attention","IS_PART_OF","transformer"),
        ("tokenization","FEEDS_INTO","transformer"),
        ("llm","REQUIRES","tokenization"),
        ("llm","USED_IN","rag"),
        ("llm","USED_IN","agent"),
        ("fine_tuning","IMPROVES","llm"),
        ("rlhf","IS_TYPE_OF","fine_tuning"),
        ("context_window","CONSTRAINS","llm"),
        ("hallucination","IS_PROBLEM_OF","llm"),
        ("prompt_engineering","GUIDES","llm"),
        ("few_shot","IS_TYPE_OF","prompt_engineering"),
        ("zero_shot","IS_TYPE_OF","prompt_engineering"),
        ("chain_of_thought","IS_TYPE_OF","prompt_engineering"),
        ("embedding","IS_TYPE_OF","dense_vector"),
        ("sentence_transformers","GENERATES","embedding"),
        ("bi_encoder","GENERATES","embedding"),
        ("embedding","ENABLES","semantic_search"),
        ("cosine_similarity","MEASURES","embedding"),
        ("semantic_search","USES","vector_database"),
        ("bm25","IS_TYPE_OF","sparse_vector"),
        ("hybrid_search","COMBINES","semantic_search"),
        ("hybrid_search","COMBINES","bm25"),
        ("reranking","IMPROVES","hybrid_search"),
        ("cross_encoder","USED_FOR","reranking"),
        ("mmr","IS_TYPE_OF","reranking"),
        ("rag","USES","semantic_search"),
        ("rag","USES","llm"),
        ("chunking","PREPARES_DATA_FOR","rag"),
        ("hyde","IMPROVES","rag"),
        ("rag","REDUCES","hallucination"),
        ("ragas","EVALUATES","rag"),
        ("faithfulness","IS_METRIC_IN","ragas"),
        ("relevance","IS_METRIC_IN","ragas"),
        ("chromadb","IS_TYPE_OF","vector_database"),
        ("vector_database","STORES","embedding"),
        ("graph_rag","EXTENDS","rag"),
        ("knowledge_graph","USED_IN","graph_rag"),
        ("neo4j","IMPLEMENTS","knowledge_graph"),
        ("entity","IS_NODE_IN","knowledge_graph"),
        ("relation","IS_EDGE_IN","knowledge_graph"),
        ("agent","USES","tool_use"),
        ("agent","USES","memory"),
        ("agent","USES","planning"),
        ("react","IS_PATTERN_FOR","agent"),
        ("langchain","IMPLEMENTS","agent"),
        ("langchain","IMPLEMENTS","rag"),
        ("ollama","RUNS","llm"),
        ("prefect","ORCHESTRATES","data_pipeline"),
        ("dbt","IS_PART_OF","data_pipeline"),
        ("duckdb","USED_BY","dbt"),
        ("etl","IS_PATTERN_FOR","data_pipeline"),
    ]
    node_ids = {n["id"] for n in nodes}
    write_csv("knowledge_graph_nodes.csv", ["id", "label", "category"],
              [{"id": n["id"], "label": n["label"], "category": n["category"]} for n in nodes])
    write_csv("knowledge_graph_edges.csv", ["source", "relationship", "target"],
              [{"source": s, "relationship": r, "target": t}
               for s, r, t in edges if s in node_ids and t in node_ids])


if __name__ == "__main__":
    print("\nGenerating datasets (seed=42) ...\n")
    generate_articles()
    generate_products()
    generate_employees()
    generate_support_tickets()
    generate_knowledge_graph()
    print("\nAll datasets written to data/raw/  v\n")
