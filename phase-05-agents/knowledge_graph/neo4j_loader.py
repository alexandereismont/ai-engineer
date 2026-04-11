"""
neo4j_loader.py — Load pension regulation entities into Neo4j.

Schema
------
Node labels:
    Regulation   — top-level legislative instrument  (name, full_name, jurisdiction)
    Article      — numbered article within a regulation (id, number, title, summary)
    Concept      — domain concept or term (name, definition)
    PensionFund  — individual pension fund (name, country, type)
    Requirement  — a specific obligation (name, description, mandatory)

Relationship types:
    REFERENCES   — Article REFERENCES Concept  (bidirectional via direction property)
    DEFINES      — Regulation DEFINES Concept
    APPLIES_TO   — Requirement APPLIES_TO PensionFund
    REQUIRES     — Article REQUIRES Requirement

Usage
-----
    python neo4j_loader.py          # loads full sample dataset
    python neo4j_loader.py --clean  # wipe and reload
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from neo4j import Driver, GraphDatabase, Result

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pension_secret")


def get_driver() -> Driver:
    """Return an authenticated Neo4j driver using environment variables."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ---------------------------------------------------------------------------
# Cypher schema & index queries
# ---------------------------------------------------------------------------

CYPHER_CONSTRAINTS = [
    "CREATE CONSTRAINT regulation_name IF NOT EXISTS FOR (r:Regulation) REQUIRE r.name IS UNIQUE",
    "CREATE CONSTRAINT article_id      IF NOT EXISTS FOR (a:Article)    REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name    IF NOT EXISTS FOR (c:Concept)    REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT fund_name       IF NOT EXISTS FOR (f:PensionFund) REQUIRE f.name IS UNIQUE",
    "CREATE CONSTRAINT req_name        IF NOT EXISTS FOR (r:Requirement) REQUIRE r.name IS UNIQUE",
]

CYPHER_CLEAR_ALL = "MATCH (n) DETACH DELETE n"

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGULATIONS: list[dict[str, Any]] = [
    {
        "name": "IORP III",
        "full_name": "Directive on the activities and supervision of institutions for "
                     "occupational retirement provision (Third)",
        "jurisdiction": "European Union",
        "year": 2025,
    },
    {
        "name": "IORP II",
        "full_name": "Directive 2016/2341 on institutions for occupational retirement provision",
        "jurisdiction": "European Union",
        "year": 2016,
    },
    {
        "name": "FTK",
        "full_name": "Financieel Toetsingskader",
        "jurisdiction": "Netherlands",
        "year": 2007,
    },
    {
        "name": "Wet toekomst pensioenen",
        "full_name": "Wet toekomst pensioenen (WTP)",
        "jurisdiction": "Netherlands",
        "year": 2023,
    },
]

# Articles within IORP III
ARTICLES: list[dict[str, Any]] = [
    {
        "id": "IORP3_ART12",
        "number": 12,
        "regulation": "IORP III",
        "title": "Investment conditions — prudent person rule",
        "summary": (
            "Assets shall be invested in the best long-term interest of members and "
            "beneficiaries. The prudent person rule requires risk diversification, "
            "liquidity management, and avoidance of excessive risk concentration."
        ),
    },
    {
        "id": "IORP3_ART14",
        "number": 14,
        "regulation": "IORP III",
        "title": "Risk management and ORSA",
        "summary": (
            "IORPs shall conduct an Own Risk and Solvency Assessment (ORSA) at least "
            "annually and whenever there is a significant change in the risk profile. "
            "Results must be documented and reported to the competent authority."
        ),
    },
    {
        "id": "IORP3_ART19",
        "number": 19,
        "regulation": "IORP III",
        "title": "Coverage ratio and funding requirements",
        "summary": (
            "IORPs must maintain technical provisions fully covered at all times. "
            "The coverage ratio (assets / liabilities) must remain above 100%. "
            "A recovery plan must be submitted within 12 weeks of a funding shortfall."
        ),
    },
    {
        "id": "IORP3_ART23",
        "number": 23,
        "regulation": "IORP III",
        "title": "ESG integration requirements",
        "summary": (
            "IORPs must integrate environmental, social, and governance (ESG) factors "
            "into investment decisions and risk management. Annual sustainability "
            "reporting aligned with SFDR is required."
        ),
    },
    {
        "id": "IORP3_ART31",
        "number": 31,
        "regulation": "IORP III",
        "title": "Governance and key functions",
        "summary": (
            "IORPs shall have an effective system of governance including risk management, "
            "internal audit, and actuarial functions. Board members must satisfy fit-and-proper "
            "requirements. Governance policies must be reviewed annually."
        ),
    },
]

CONCEPTS: list[dict[str, Any]] = [
    {
        "name": "coverage_ratio",
        "definition": (
            "The ratio of a pension fund's assets to its liabilities (technical provisions). "
            "A ratio above 100% indicates the fund can meet all obligations. "
            "Also known as 'dekkingsgraad' in Dutch regulation."
        ),
    },
    {
        "name": "recovery_plan",
        "definition": (
            "A structured remediation plan submitted to the supervisor when a fund's "
            "coverage ratio falls below the required minimum. Must include concrete "
            "measures and a timeline for restoring adequate funding."
        ),
    },
    {
        "name": "prudent_person_principle",
        "definition": (
            "An investment standard requiring pension funds to act with the care, skill, "
            "and diligence of a prudent person managing a fund for the long-term benefit "
            "of its members. Prohibits speculative or excessively concentrated investments."
        ),
    },
    {
        "name": "ORSA",
        "definition": (
            "Own Risk and Solvency Assessment — a forward-looking internal process "
            "used by IORPs to assess their overall solvency needs, risk profile, and "
            "alignment of risk strategy with their investment and liability profile."
        ),
    },
    {
        "name": "ESG",
        "definition": (
            "Environmental, Social, and Governance factors integrated into investment "
            "analysis and decision-making. Mandatory for EU pension funds under IORP III "
            "and the SFDR framework."
        ),
    },
    {
        "name": "technical_provisions",
        "definition": (
            "The present value of future pension obligations, calculated using actuarial "
            "methods. IORPs must hold assets at least equal to technical provisions at all times."
        ),
    },
    {
        "name": "risk_diversification",
        "definition": (
            "Spreading investments across asset classes, geographies, and sectors to "
            "reduce concentration risk. Required under the prudent person rule."
        ),
    },
    {
        "name": "fit_and_proper",
        "definition": (
            "Regulatory requirement that board members and key function holders of an IORP "
            "demonstrate adequate professional qualifications, knowledge, experience, and "
            "personal integrity."
        ),
    },
]

# Article → Concept relationships: which concepts does each article reference?
ARTICLE_CONCEPT_REFS: list[tuple[str, str]] = [
    ("IORP3_ART12", "prudent_person_principle"),
    ("IORP3_ART12", "risk_diversification"),
    ("IORP3_ART14", "ORSA"),
    ("IORP3_ART14", "technical_provisions"),
    ("IORP3_ART19", "coverage_ratio"),
    ("IORP3_ART19", "recovery_plan"),
    ("IORP3_ART19", "technical_provisions"),
    ("IORP3_ART23", "ESG"),
    ("IORP3_ART23", "risk_diversification"),
    ("IORP3_ART31", "fit_and_proper"),
    ("IORP3_ART31", "ORSA"),
]

PENSION_FUNDS: list[dict[str, Any]] = [
    {"name": "ABP", "country": "Netherlands", "type": "industry", "aum_eur_bn": 490},
    {"name": "PFZW", "country": "Netherlands", "type": "industry", "aum_eur_bn": 245},
    {"name": "PMT", "country": "Netherlands", "type": "industry", "aum_eur_bn": 85},
    {"name": "BpfBOUW", "country": "Netherlands", "type": "industry", "aum_eur_bn": 72},
]

REQUIREMENTS: list[dict[str, Any]] = [
    {
        "name": "annual_ORSA",
        "description": "Submit ORSA report to competent authority at least annually.",
        "mandatory": True,
        "article": "IORP3_ART14",
    },
    {
        "name": "coverage_ratio_minimum",
        "description": "Maintain coverage ratio >= 100% at all times.",
        "mandatory": True,
        "article": "IORP3_ART19",
    },
    {
        "name": "recovery_plan_12_weeks",
        "description": "Submit recovery plan within 12 weeks of funding shortfall.",
        "mandatory": True,
        "article": "IORP3_ART19",
    },
    {
        "name": "ESG_integration",
        "description": "Integrate ESG factors into investment and risk management processes.",
        "mandatory": True,
        "article": "IORP3_ART23",
    },
    {
        "name": "board_fit_proper",
        "description": "Board members must pass fit-and-proper assessment.",
        "mandatory": True,
        "article": "IORP3_ART31",
    },
]

# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

CYPHER_CREATE_REGULATION = """
MERGE (r:Regulation {name: $name})
SET r.full_name    = $full_name,
    r.jurisdiction = $jurisdiction,
    r.year         = $year
"""

CYPHER_CREATE_ARTICLE = """
MERGE (a:Article {id: $id})
SET a.number  = $number,
    a.title   = $title,
    a.summary = $summary
WITH a
MATCH (reg:Regulation {name: $regulation})
MERGE (reg)-[:CONTAINS]->(a)
"""

CYPHER_CREATE_CONCEPT = """
MERGE (c:Concept {name: $name})
SET c.definition = $definition
"""

CYPHER_CREATE_ARTICLE_CONCEPT_REF = """
MATCH (a:Article  {id:   $article_id})
MATCH (c:Concept  {name: $concept_name})
MERGE (a)-[:REFERENCES]->(c)
"""

CYPHER_CREATE_FUND = """
MERGE (f:PensionFund {name: $name})
SET f.country    = $country,
    f.type       = $type,
    f.aum_eur_bn = $aum_eur_bn
"""

CYPHER_CREATE_REQUIREMENT = """
MERGE (req:Requirement {name: $name})
SET req.description = $description,
    req.mandatory   = $mandatory
WITH req
MATCH (a:Article {id: $article})
MERGE (a)-[:REQUIRES]->(req)
"""

CYPHER_APPLY_REQUIREMENT_TO_FUNDS = """
MATCH (req:Requirement)
MATCH (f:PensionFund)
MERGE (req)-[:APPLIES_TO]->(f)
"""

CYPHER_REGULATION_DEFINES_CONCEPT = """
MATCH (reg:Regulation {name: 'IORP III'})
MATCH (c:Concept)
MERGE (reg)-[:DEFINES]->(c)
"""


def create_schema(driver: Driver) -> None:
    """Create constraints and indexes."""
    with driver.session() as session:
        for cypher in CYPHER_CONSTRAINTS:
            try:
                session.run(cypher)
            except Exception as exc:  # noqa: BLE001
                print(f"  Warning (constraint may already exist): {exc}")
    print("Schema constraints applied.")


def load_data(driver: Driver) -> None:
    """Insert all sample data nodes and relationships."""
    with driver.session() as session:
        # Regulations
        for reg in REGULATIONS:
            session.run(CYPHER_CREATE_REGULATION, **reg)
        print(f"  Loaded {len(REGULATIONS)} Regulation nodes.")

        # Articles
        for art in ARTICLES:
            session.run(CYPHER_CREATE_ARTICLE, **art)
        print(f"  Loaded {len(ARTICLES)} Article nodes.")

        # Concepts
        for con in CONCEPTS:
            session.run(CYPHER_CREATE_CONCEPT, **con)
        print(f"  Loaded {len(CONCEPTS)} Concept nodes.")

        # Article → Concept references
        for article_id, concept_name in ARTICLE_CONCEPT_REFS:
            session.run(
                CYPHER_CREATE_ARTICLE_CONCEPT_REF,
                article_id=article_id,
                concept_name=concept_name,
            )
        print(f"  Loaded {len(ARTICLE_CONCEPT_REFS)} REFERENCES relationships.")

        # Pension Funds
        for fund in PENSION_FUNDS:
            session.run(CYPHER_CREATE_FUND, **fund)
        print(f"  Loaded {len(PENSION_FUNDS)} PensionFund nodes.")

        # Requirements
        for req in REQUIREMENTS:
            session.run(CYPHER_CREATE_REQUIREMENT, **req)
        print(f"  Loaded {len(REQUIREMENTS)} Requirement nodes.")

        # Cross-links: all requirements apply to all funds (simplified)
        session.run(CYPHER_APPLY_REQUIREMENT_TO_FUNDS)
        print("  Linked requirements to all pension funds.")

        # IORP III DEFINES all concepts
        session.run(CYPHER_REGULATION_DEFINES_CONCEPT)
        print("  Linked IORP III DEFINES all concepts.")


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

CYPHER_QUERY_RELATED_ENTITIES = """
MATCH (n)-[r]-(m)
WHERE toLower(n.name) CONTAINS toLower($entity_name)
   OR toLower(n.id)   CONTAINS toLower($entity_name)
RETURN
    n.name          AS source,
    labels(n)[0]    AS source_type,
    type(r)         AS relationship,
    m.name          AS target,
    labels(m)[0]    AS target_type,
    CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
ORDER BY relationship, target
LIMIT 50
"""

CYPHER_ARTICLES_FOR_CONCEPT = """
MATCH (a:Article)-[:REFERENCES]->(c:Concept {name: $concept_name})
RETURN a.id AS article_id, a.number AS number, a.title AS title, a.summary AS summary
ORDER BY a.number
"""

CYPHER_REQUIREMENTS_FOR_FUND = """
MATCH (req:Requirement)-[:APPLIES_TO]->(f:PensionFund {name: $fund_name})
MATCH (a:Article)-[:REQUIRES]->(req)
RETURN
    req.name        AS requirement,
    req.description AS description,
    req.mandatory   AS mandatory,
    a.id            AS article_id
ORDER BY req.name
"""


def query_related_entities(entity_name: str, driver: Driver | None = None) -> list[dict]:
    """Find all entities connected to ``entity_name`` in the knowledge graph.

    Args:
        entity_name: Partial or full name of any node (case-insensitive substring match).
        driver:      Optional pre-created Neo4j driver; creates a new one if not supplied.

    Returns:
        List of dicts with keys: source, source_type, relationship, target, target_type, direction.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()
    try:
        with driver.session() as session:
            result: Result = session.run(CYPHER_QUERY_RELATED_ENTITIES, entity_name=entity_name)
            return [dict(r) for r in result]
    finally:
        if own_driver:
            driver.close()


def query_articles_for_concept(concept_name: str, driver: Driver | None = None) -> list[dict]:
    """Return all articles that reference a given concept.

    Args:
        concept_name: Exact concept name (e.g. "coverage_ratio").
        driver:       Optional pre-created driver.

    Returns:
        List of article dicts.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(CYPHER_ARTICLES_FOR_CONCEPT, concept_name=concept_name)
            return [dict(r) for r in result]
    finally:
        if own_driver:
            driver.close()


def query_requirements_for_fund(fund_name: str, driver: Driver | None = None) -> list[dict]:
    """Return all regulatory requirements that apply to a pension fund.

    Args:
        fund_name: Exact PensionFund node name (e.g. "ABP").
        driver:    Optional pre-created driver.

    Returns:
        List of requirement dicts.
    """
    own_driver = driver is None
    if own_driver:
        driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(CYPHER_REQUIREMENTS_FOR_FUND, fund_name=fund_name)
            return [dict(r) for r in result]
    finally:
        if own_driver:
            driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load pension regulation graph into Neo4j.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing nodes before loading.",
    )
    parser.add_argument(
        "--query",
        metavar="ENTITY",
        help="After loading, query related entities for ENTITY and print results.",
    )
    args = parser.parse_args()

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    driver = get_driver()
    driver.verify_connectivity()
    print("Connected.\n")

    if args.clean:
        with driver.session() as session:
            session.run(CYPHER_CLEAR_ALL)
        print("All nodes deleted.\n")

    print("Applying schema constraints ...")
    create_schema(driver)

    print("\nLoading data ...")
    load_data(driver)
    print("\nData load complete.")

    if args.query:
        print(f"\nQuerying related entities for '{args.query}' ...")
        records = query_related_entities(args.query, driver=driver)
        if records:
            for rec in records:
                print(
                    f"  {rec['source']} --[{rec['relationship']}]--> "
                    f"{rec['target']} ({rec['target_type']})"
                )
        else:
            print("  No results found.")

    driver.close()


if __name__ == "__main__":
    main()
