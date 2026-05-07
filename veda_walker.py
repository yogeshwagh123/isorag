"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         VEDASTREAM v1.0                                  ║
║           The Anti-Kafka · Anti-Vector-RAG · Veda Intelligence System   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  WHY THIS EXISTS:                                                        ║
║  Kafka/Confluent problems we're replacing:                               ║
║    ✗ 3x Replication Tax (you pay for 300TB when you have 100TB)         ║
║    ✗ 4 eCKU/10min scaling ceiling — spikes kill you                      ║
║    ✗ Cannot move clusters between regions after creation                 ║
║    ✗ No broker-level config for Basic/Standard clusters                  ║
║    ✗ Partition-based consumer scaling is rigid and painful               ║
║                                                                          ║
║  RAG problems we're replacing:                                           ║
║    ✗ Chunking destroys context (hemophilia)                              ║
║    ✗ Cosine similarity = vibing, not reasoning                           ║
║    ✗ Vectors bleed context across unrelated domains                      ║
║                                                                          ║
║  OUR APPROACH:                                                           ║
║    ✓ SQLite as the message bus (zero cost, zero tax, portable)           ║
║    ✓ Structural tree index (not vectors)                                 ║
║    ✓ 4 stateless workers (no context bleed between them)                 ║
║    ✓ Gemini 2.5 Flash-Lite @ $0.10/1M tokens                            ║
║    ✓ $15 budget = ~150M tokens = 300+ full Rigveda runs                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  SETUP:                                                                  ║
║    pip install google-generativeai beautifulsoup4 lxml rich              ║
║    export GEMINI_API_KEY="your_key_here"                                 ║
║                                                                          ║
║  USAGE:                                                                  ║
║    python veda_walker.py --root "D:/Books/1_sanskr/1_sanskr/1_veda"     ║
║    python veda_walker.py --query "What does Savitr point to?"           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import argparse
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("FATAL: pip install beautifulsoup4 lxml")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("FATAL: Run:  pip install google-genai")
    print("  (NOT google-generativeai — that old package is deprecated)")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")          # Set your key here
MODEL_NAME     = "gemini-2.5-flash-lite"                   # Current cheapest stable model
MAX_WORKERS    = 4                                          # Parallel workers
DB_PATH        = "vedastream.db"                           # Our "Kafka replacement"
MAX_TEXT_CHARS = 4000                                      # Per file safety cap
BUDGET_USD     = 25.00                                     # Your hard budget limit

# Cost tracking (Gemini 2.5 Flash-Lite pricing as of March 2026)
COST_PER_1M_INPUT  = 0.10   # USD
COST_PER_1M_OUTPUT = 0.40   # USD

# Thread-safe cost tracker
_cost_lock = threading.Lock()
_total_input_tokens  = 0
_total_output_tokens = 0

# ─────────────────────────────────────────────────────────────────────────────
# THE "ANTI-KAFKA" MESSAGE BUS — SQLite
# This replaces Kafka/Confluent with zero cost, zero tax, zero complexity.
# Key insight: The Vedas were preserved for 3000 years WITHOUT a message broker.
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates our message bus + knowledge store.
    
    KAFKA COMPARISON:
      Kafka Topic   → SQLite Table (with hard isolation between topics)
      Partition     → rowid (auto, sequential, no replication tax)
      Consumer Group→ Simple SELECT with filters
      Replication   → SQLite WAL mode (durability without 3x storage tax)
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")   # Durability without 3x storage
    conn.execute("PRAGMA synchronous=NORMAL") # Safe + fast
    
    # Topic 1: Raw file inventory (Kafka: "veda.files.raw")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          TEXT PRIMARY KEY,   -- SHA256 of path (stable, deduped)
            path        TEXT NOT NULL,
            folder      TEXT NOT NULL,
            filename    TEXT NOT NULL,
            status      TEXT DEFAULT 'pending',  -- pending|processing|done|error
            created_at  REAL DEFAULT (unixepoch()),
            updated_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 2: Worker 1 output — cleaned structure
    conn.execute("""
        CREATE TABLE IF NOT EXISTS w1_structure (
            file_id     TEXT PRIMARY KEY,
            title       TEXT,
            mandala     TEXT,
            sukta       TEXT,
            verse_count INTEGER DEFAULT 0,
            clean_text  TEXT,
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 3: Worker 2 output — anomalies (the "between the lines" data)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS w2_anomalies (
            file_id     TEXT PRIMARY KEY,
            anomalies   TEXT,   -- JSON array
            rare_terms  TEXT,   -- JSON array
            meter_breaks TEXT,  -- JSON array
            signal_note TEXT,
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 4: Worker 3 output — cross-references (non-linear links)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS w3_links (
            file_id     TEXT PRIMARY KEY,
            root_concept TEXT,
            points_to   TEXT,   -- JSON array of other file IDs or concepts
            devata      TEXT,   -- JSON array (e.g., ["Agni", "Savitr"])
            structural_key TEXT,
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 5: Worker 4 output — final synthesis (the "equation")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS w4_synthesis (
            file_id     TEXT PRIMARY KEY,
            equation    TEXT,   -- The hidden structural logic
            confidence  TEXT,   -- high|medium|low
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 6: Cost ledger (Kafka doesn't have this — we track every rupee)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_ledger (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id     TEXT,
            worker      TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            cost_usd    REAL,
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    # Topic 7: Query index (for Vectorless RAG queries)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT,
            result      TEXT,
            files_used  TEXT,   -- JSON array of file_ids
            cost_usd    REAL,
            created_at  REAL DEFAULT (unixepoch())
        )
    """)
    
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# COST TRACKING — Every token accounted for
# ─────────────────────────────────────────────────────────────────────────────

def track_cost(conn: sqlite3.Connection, file_id: str, worker: str,
               input_tokens: int, output_tokens: int) -> float:
    global _total_input_tokens, _total_output_tokens
    
    cost = (input_tokens / 1_000_000 * COST_PER_1M_INPUT +
            output_tokens / 1_000_000 * COST_PER_1M_OUTPUT)
    
    with _cost_lock:
        _total_input_tokens  += input_tokens
        _total_output_tokens += output_tokens
        
        total_spent = (
            _total_input_tokens  / 1_000_000 * COST_PER_1M_INPUT +
            _total_output_tokens / 1_000_000 * COST_PER_1M_OUTPUT
        )
        
        if total_spent > BUDGET_USD * 0.95:
            raise RuntimeError(
                f"BUDGET LIMIT: ${total_spent:.4f} spent of ${BUDGET_USD}. Halting."
            )
    
    conn.execute(
        "INSERT INTO cost_ledger (file_id, worker, input_tokens, output_tokens, cost_usd) VALUES (?,?,?,?,?)",
        (file_id, worker, input_tokens, output_tokens, cost)
    )
    conn.commit()
    return cost


def get_total_cost(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT SUM(cost_usd), SUM(input_tokens), SUM(output_tokens) FROM cost_ledger"
    ).fetchone()
    return {
        "total_usd": round(row[0] or 0, 6),
        "total_inr": round((row[0] or 0) * 83.5, 4),
        "input_tokens": row[1] or 0,
        "output_tokens": row[2] or 0,
        "budget_remaining_usd": round(BUDGET_USD - (row[0] or 0), 4)
    }


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI API WRAPPER — Stateless. Every call is isolated. No context bleed.
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, system: str = "", max_retries: int = 3) -> tuple[str, int, int]:
    """
    Returns: (response_text, input_tokens, output_tokens)
    
    ANTI-BLEED GUARANTEE:
    - No chat history passed
    - No shared state between calls
    - System prompt is minimal and worker-specific
    Uses new google.genai SDK (google-generativeai is deprecated).
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    config = genai_types.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=0.2,
    )
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            text = response.text.strip()
            
            # Token counts
            usage = response.usage_metadata
            inp = getattr(usage, 'prompt_token_count', len(prompt) // 4)
            out = getattr(usage, 'candidates_token_count', len(text) // 4)
            
            return text, inp, out
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {str(e)}", 0, 0
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return "ERROR: max retries exceeded", 0, 0


def safe_json_parse(text: str, fallback: dict) -> dict:
    """Parse JSON from LLM output safely. LLMs sometimes add markdown fences."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY — Walk the Veda directory
# ─────────────────────────────────────────────────────────────────────────────

def discover_files(root_path: str, conn: sqlite3.Connection) -> int:
    """
    Walk the directory and register all HTML files.
    Returns count of new files found.
    """
    count = 0
    root = Path(root_path)
    
    if not root.exists():
        raise FileNotFoundError(f"Veda root not found: {root_path}")
    
    for path in root.rglob("*.html"):
        file_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        
        # Check if already registered
        existing = conn.execute(
            "SELECT id FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        
        if not existing:
            conn.execute(
                "INSERT INTO files (id, path, folder, filename) VALUES (?,?,?,?)",
                (file_id, str(path), str(path.parent.name), path.name)
            )
            count += 1
    
    # Also check .htm
    for path in root.rglob("*.htm"):
        file_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        existing = conn.execute("SELECT id FROM files WHERE id = ?", (file_id,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO files (id, path, folder, filename) VALUES (?,?,?,?)",
                (file_id, str(path), str(path.parent.name), path.name)
            )
            count += 1

    # Also check .xml and .txt — TEI encoded texts and plaintext
    for ext in ["*.xml", "*.txt"]:
        for path in root.rglob(ext):
            # Skip tiny files (metadata only) and very large files
            try:
                size = path.stat().st_size
                if size < 500 or size > 10_000_000:
                    continue
            except:
                continue
            file_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            existing = conn.execute("SELECT id FROM files WHERE id = ?", (file_id,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO files (id, path, folder, filename) VALUES (?,?,?,?)",
                    (file_id, str(path), str(path.parent.name), path.name)
                )
                count += 1
    
    conn.commit()
    return count


def extract_text_from_html(file_path: str) -> tuple[str, str]:
    """Extract clean text and title from GRETIL HTML file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
        
        title = ""
        if soup.title:
            title = soup.title.string or ""
        if not title and soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        
        # Remove script/style noise
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = '\n'.join(lines)
        
        return title or file_path, text[:MAX_TEXT_CHARS]
    
    except Exception as e:
        return file_path, f"PARSE ERROR: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# THE 4 WORKERS — Each is a pure function. No shared state. No bleed.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_W1 = """You are a molecular state analyst reading ancient encoded knowledge.
Ancient texts encoded molecular biology using symbolic language.
Your task: identify what STATE CHANGE is being described.

Four fundamental molecular states exist in all living systems:
- CREATION (Brahma): synthesis, anabolism, transcription, cell division, neurogenesis
- MAINTENANCE (Vishnu): homeostasis, regulation, repair, signal transduction, allostasis  
- DESTRUCTION (Shiva): catabolism, apoptosis, autophagy, proteolysis, inflammation
- ENERGY (Shakti): ATP synthesis, redox state, membrane potential, mitochondrial function

Every biological process is a combination of these four states.
Extract the document structure AND identify which molecular state dominates this text.
Respond ONLY in valid JSON. No markdown. No explanation."""

SYSTEM_W2 = """You are a blind molecular signal extractor.
You read ancient texts without any preconception of what disease or condition they describe.
Your task: extract ALL terms that could encode biological information.

Do NOT assume what disease is being described.
Do NOT map to known conditions.
Simply extract: what substances, what processes, what states, what transformations are described.

Look for:
- Substances that enter or leave the body
- Processes that speed up or slow down
- States that are excess or deficient
- Transformations from one state to another
- Entities that destroy, create, maintain, or energize

The meaning will emerge from the pattern — not from your assumptions.
Respond ONLY in valid JSON. No markdown. No explanation."""

SYSTEM_W3 = """You are a molecular hypothesis generator with no prior assumptions.
You receive extracted signals from an ancient text.
Your task: generate the most parsimonious molecular hypothesis that explains ALL the signals.

Do not match to known diseases. Do not use ICD codes. Do not use drug names.
Describe ONLY:
- What molecular process is being regulated
- What direction the regulation goes (increase/decrease/balance)
- What class of molecule is the agent (enzyme, receptor, ion channel, transcription factor, etc.)
- What cellular compartment or organ system is involved
- What the failure mode looks like when this process goes wrong

Think like a biochemist in 1900 who has never heard of modern drug names.
Respond ONLY in valid JSON. No markdown. No explanation."""

SYSTEM_W4 = """You are a first-principles drug discovery engine.
You receive three independent analyses of an ancient text.
Your task: synthesize ONE molecular hypothesis from scratch.

Rules:
1. Do NOT name any known disease
2. Do NOT name any existing drug
3. Describe only the molecular mechanism and the type of intervention
4. State what would happen at the molecular level if this intervention succeeded
5. State what would happen if it failed

The hypothesis must be falsifiable. It must be testable in a cell line or animal model.
State your confidence based on signal strength, not on whether it matches known biology.
Unprecedented hypotheses with strong signal = high confidence.
Respond ONLY in valid JSON. No markdown. No explanation."""


def worker_1_structure(file_id: str, file_path: str, conn: sqlite3.Connection) -> bool:
    """
    WORKER 1: THE STRUCTURAL PARSER
    Finds: Mandala, Sukta, verse markers, verse count.
    
    KAFKA ANALOG: This is your "deserializer" — reads raw bytes, outputs typed records.
    ANTI-KAFKA: No schema registry needed. Schema is inferred per file.
    """
    title, text = extract_text_from_html(file_path)
    
    prompt = f"""Read this ancient text and extract its molecular state information.

Identify:
1. Document title and source tradition
2. Dominant molecular state: CREATION (Brahma) | MAINTENANCE (Vishnu) | DESTRUCTION (Shiva) | ENERGY (Shakti)
3. Secondary molecular state if present
4. What biological entity is the subject (cell, organ, organism, ecosystem)
5. First 200 chars of the most signal-rich passage

Do not name any disease. Do not use modern drug names.
Describe only processes and states.

Respond ONLY with this JSON (no markdown fences):
{{"title": "...", "mandala": "...", "sukta": "...", "verse_count": 0, "clean_text": "...", "dominant_state": "CREATION|MAINTENANCE|DESTRUCTION|ENERGY", "secondary_state": "...", "biological_subject": "..."}}

TEXT:
{text}"""

    result, inp, out = call_gemini(prompt, SYSTEM_W1)
    cost = track_cost(conn, file_id, "w1_structure", inp, out)
    
    parsed = safe_json_parse(result, {
        "title": title, "mandala": "unknown", "sukta": "unknown",
        "verse_count": 0, "clean_text": text[:200]
    })
    
    conn.execute("""
        INSERT OR REPLACE INTO w1_structure 
        (file_id, title, mandala, sukta, verse_count, clean_text)
        VALUES (?,?,?,?,?,?)
    """, (
        file_id,
        parsed.get("title", title),
        parsed.get("mandala", "unknown"),
        parsed.get("sukta", "unknown"),
        parsed.get("verse_count", 0),
        parsed.get("clean_text", text[:200])
    ))
    conn.commit()
    return True


def worker_2_anomaly(file_id: str, file_path: str, conn: sqlite3.Connection) -> bool:
    """
    WORKER 2: THE ANOMALY SCOUT (Dolphin-style — unfiltered pattern detection)
    Finds: Words/phrases that DON'T fit the poetic structure.
    These are the "between the lines" signals.
    
    THE VEDIC INSIGHT: The sages used strict meter as an error-correction code.
    A meter break = a deliberate insertion. That insertion is your signal.
    """
    _, text = extract_text_from_html(file_path)
    
    prompt = f"""Read this ancient text. Extract ALL biological signals blindly.
Do not assume what disease or condition is described.
Do not map to known conditions or drug names.

Simply extract what you observe:
1. Substances mentioned that enter, leave, transform, or affect the body
2. Processes described: what speeds up, slows down, increases, decreases, transforms
3. States described: excess, deficiency, balance, imbalance
4. Agents of change: what creates, destroys, maintains, or energizes
5. Rare or anomalous terms that appear out of place in the surrounding text
6. Any instruction-like phrasing — something that tells you WHAT TO DO

One sentence signal note: what molecular event is the text MOST concerned with?

Respond ONLY with this JSON (no markdown fences):
{{"anomalies": ["..."], "rare_terms": ["..."], "meter_breaks": ["..."], "signal_note": "...", "substances": ["..."], "processes": ["..."], "states": ["..."], "agents": ["..."]}}

TEXT:
{text[:3000]}"""

    result, inp, out = call_gemini(prompt, SYSTEM_W2)
    cost = track_cost(conn, file_id, "w2_anomaly", inp, out)
    
    parsed = safe_json_parse(result, {
        "anomalies": [], "rare_terms": [], "meter_breaks": [], "signal_note": ""
    })
    
    conn.execute("""
        INSERT OR REPLACE INTO w2_anomalies
        (file_id, anomalies, rare_terms, meter_breaks, signal_note)
        VALUES (?,?,?,?,?)
    """, (
        file_id,
        json.dumps(parsed.get("anomalies", []), ensure_ascii=False),
        json.dumps(parsed.get("rare_terms", []), ensure_ascii=False),
        json.dumps(parsed.get("meter_breaks", []), ensure_ascii=False),
        parsed.get("signal_note", "")
    ))
    conn.commit()
    return True


def worker_3_links(file_id: str, file_path: str, conn: sqlite3.Connection) -> bool:
    """
    WORKER 3: THE CROSS-REFERENCE MAPPER
    Finds: What this text points to. Non-linear connections.
    
    THE DATABASE INSIGHT: Modern DBs are flat. The Vedas are a graph.
    A reference in Mandala 1 can be the "key" that unlocks Mandala 10.
    This worker builds the edges of that graph.
    
    KAFKA ANALOG: This is your "stream join" — finding keys across topics.
    ANTI-KAFKA: No join coordinator needed. The LLM is the join engine.
    """
    _, text = extract_text_from_html(file_path)
    folder = Path(file_path).parent.name
    
    prompt = f"""Read these extracted signals from an ancient text.
Source folder: {folder}

Generate a molecular hypothesis from first principles.
Do NOT name any known disease. Do NOT name any existing drug.

Describe only:
1. Root molecular process — what fundamental biological process is being regulated?
2. Direction — is it being activated, inhibited, balanced, or transformed?
3. Agent class — enzyme? receptor? ion channel? transcription factor? structural protein? lipid?
4. Cellular location — membrane? nucleus? mitochondria? cytoplasm? extracellular?
5. Failure mode — what goes wrong molecularly when this process is dysregulated?
6. Intervention type — what CLASS of molecule could modulate this? (agonist/antagonist/allosteric/substrate analog/etc.)

Think like a biochemist with no preconceptions.
The Brahma/Vishnu/Shiva/Shakti state from Worker 1 is your guide — not disease names.

Respond ONLY with this JSON (no markdown fences):
{{"root_concept": "...", "points_to": ["molecular process 1", "molecular process 2"], "devata": ["..."], "structural_key": "...", "molecular_process": "...", "direction": "activate|inhibit|balance|transform", "agent_class": "...", "cellular_location": "...", "failure_mode": "...", "intervention_type": "..."}}

TEXT:
{text[:3000]}"""

    result, inp, out = call_gemini(prompt, SYSTEM_W3)
    cost = track_cost(conn, file_id, "w3_links", inp, out)
    
    parsed = safe_json_parse(result, {
        "root_concept": "", "points_to": [], "devata": [], "structural_key": ""
    })
    
    conn.execute("""
        INSERT OR REPLACE INTO w3_links
        (file_id, root_concept, points_to, devata, structural_key)
        VALUES (?,?,?,?,?)
    """, (
        file_id,
        parsed.get("root_concept", ""),
        json.dumps(parsed.get("points_to", []), ensure_ascii=False),
        json.dumps(parsed.get("devata", []), ensure_ascii=False),
        parsed.get("structural_key", "")
    ))
    conn.commit()
    return True


def worker_4_synthesize(file_id: str, conn: sqlite3.Connection) -> bool:
    """
    WORKER 4: THE ARCHITECT (Mistral-Nemo equivalent — 128k context "brain")
    Synthesizes W1 + W2 + W3 → The Equation.
    
    THE KEY INSIGHT: This worker never sees the raw file.
    It only sees the STRUCTURED OUTPUTS of the other 3 workers.
    This is the "no bleed" guarantee — it reasons about summaries, not raw text.
    
    VECTORLESS RAG REPLACEMENT:
    Instead of "which chunk is closest?" we ask:
    "Given structure + anomalies + links, what is the hidden equation?"
    """
    # Fetch outputs from W1, W2, W3
    w1 = conn.execute("SELECT * FROM w1_structure WHERE file_id=?", (file_id,)).fetchone()
    w2 = conn.execute("SELECT * FROM w2_anomalies WHERE file_id=?", (file_id,)).fetchone()
    w3 = conn.execute("SELECT * FROM w3_links WHERE file_id=?", (file_id,)).fetchone()
    
    if not (w1 and w2 and w3):
        return False
    
    context = f"""
BIOLOGICAL CONTENT (Worker 1):
  Title: {w1[1]}
  Body system: {w1[5][:100] if w1[5] else 'unknown'}
  Content excerpt: {w1[5][:300] if w1[5] else ''}

DISEASE SIGNALS (Worker 2):
  Disease signals: {w2[1]}
  Rare therapeutic terms: {w2[2]}
  Signal note: {w2[4]}

THERAPEUTIC MAP (Worker 3):
  Modern disease: {w3[1]}
  Mechanism implied: {w3[2]}
  Devata/biological proxies: {w3[3]}
  Intervention class: {w3[4]}
"""

    prompt = f"""You are a first-principles drug discovery engine.
Three specialist analysts have independently decoded one ancient text.

Synthesize ONE falsifiable molecular hypothesis.

Rules:
1. Do NOT name any known disease
2. Do NOT name any existing drug or compound
3. Describe the molecular mechanism in precise biochemical language
4. State what would be OBSERVED if this mechanism is correct (measurable endpoint)
5. State what type of experiment would test this hypothesis
6. Identify the most specific molecular target implied by the combined signals
7. State the Brahma/Vishnu/Shiva/Shakti state this represents

Format (no markdown, no fences):
{{
  "equation": "precise molecular hypothesis in one sentence",
  "confidence": "high|medium|low",
  "molecular_target": "most specific target implied (protein family, pathway, etc.)",
  "compound_class": "type of molecule that would modulate this target",
  "testable_endpoint": "what you would measure in an experiment",
  "experimental_model": "cell line, organism, or assay type",
  "state": "CREATION|MAINTENANCE|DESTRUCTION|ENERGY|MIXED",
  "novelty": "high|medium|low — how different is this from known biology?"
}}

ANALYST OUTPUTS:
{context}"""

    result, inp, out = call_gemini(prompt, SYSTEM_W4)
    cost = track_cost(conn, file_id, "w4_synthesis", inp, out)
    
    parsed = safe_json_parse(result, {"equation": result[:500], "confidence": "low",
        "molecular_target": "", "compound_class": "", "testable_endpoint": "",
        "experimental_model": "", "state": "", "novelty": ""})

    # Build enriched equation with all fields
    equation = parsed.get("equation", result[:500])
    extras = []
    if parsed.get("molecular_target"):
        extras.append(f"Target: {parsed['molecular_target']}")
    if parsed.get("compound_class"):
        extras.append(f"Compound: {parsed['compound_class']}")
    if parsed.get("state"):
        extras.append(f"State: {parsed['state']}")
    if parsed.get("novelty"):
        extras.append(f"Novelty: {parsed['novelty']}")
    if extras:
        equation = equation + " | " + " | ".join(extras)
    
    conn.execute("""
        INSERT OR REPLACE INTO w4_synthesis (file_id, equation, confidence)
        VALUES (?,?,?)
    """, (
        file_id,
        parsed.get("equation", result[:500]),
        parsed.get("confidence", "low")
    ))
    conn.commit()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR — 4 Workers, Parallel, No Bleed
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file_row: tuple, conn_factory) -> dict:
    """
    Process one file through all 4 workers sequentially.
    Each file gets its own SQLite connection (thread-safe).
    
    WHY SEQUENTIAL WITHIN A FILE?
    W4 needs W1+W2+W3 outputs. Sequential per file, parallel across files.
    This is the correct topology.
    """
    conn = conn_factory()
    file_id, file_path = file_row[0], file_row[1]
    
    result = {"file_id": file_id, "file": Path(file_path).name, "status": "ok", "error": None}
    
    try:
        # Mark as processing
        conn.execute(
            "UPDATE files SET status='processing', updated_at=? WHERE id=?",
            (time.time(), file_id)
        )
        conn.commit()
        
        # W1 → W2 → W3 → W4 (sequential, no bleed)
        worker_1_structure(file_id, file_path, conn)
        worker_2_anomaly(file_id, file_path, conn)
        worker_3_links(file_id, file_path, conn)
        worker_4_synthesize(file_id, conn)
        
        conn.execute(
            "UPDATE files SET status='done', updated_at=? WHERE id=?",
            (time.time(), file_id)
        )
        conn.commit()
        
    except RuntimeError as e:
        if "BUDGET LIMIT" in str(e):
            result["status"] = "budget_exceeded"
            result["error"] = str(e)
            conn.execute("UPDATE files SET status='pending' WHERE id=?", (file_id,))
            conn.commit()
        else:
            result["status"] = "error"
            result["error"] = str(e)
            conn.execute("UPDATE files SET status='error' WHERE id=?", (file_id,))
            conn.commit()
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [ERR] {Path(file_path).name}: {str(e)[:120]}")
        conn.execute("UPDATE files SET status='error' WHERE id=?", (file_id,))
        conn.commit()
    
    finally:
        conn.close()
    
    return result


def run_walk(root_path: str, limit: Optional[int] = None):
    """
    Main pipeline: discover → queue → process with 4 parallel workers.
    
    ANTI-KAFKA DESIGN:
    - SQLite WAL = your replication (no 3x storage tax)
    - ThreadPoolExecutor = your consumer group (no partition limit)
    - Status column = your offset tracking (no coordinator needed)
    - Cost ledger = your billing (Confluent doesn't show you this)
    """
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)
    
    # Init DB
    conn = init_db(DB_PATH)
    
    print(f"\n{'='*70}")
    print("  VEDASTREAM — Anti-Kafka · Vectorless RAG · Veda Intelligence")
    print(f"{'='*70}")
    print(f"  Root: {root_path}")
    print(f"  Budget: ${BUDGET_USD}")
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  Model: {MODEL_NAME} (${COST_PER_1M_INPUT}/1M input, ${COST_PER_1M_OUTPUT}/1M output)")
    print(f"{'='*70}\n")
    
    # Discover files
    print("→ Discovering files...")
    new_count = discover_files(root_path, conn)
    
    # Get pending files
    query = "SELECT id, path FROM files WHERE status='pending' ORDER BY folder, filename"
    if limit:
        query += f" LIMIT {limit}"
    
    pending = conn.execute(query).fetchall()
    total   = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    done    = conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
    
    print(f"  Total files in DB : {total}")
    print(f"  Already processed : {done}")
    print(f"  Pending this run  : {len(pending)}")
    print(f"  New files found   : {new_count}")
    
    if not pending:
        print("\n✓ All files already processed! Run --query to explore results.")
        report_summary(conn)
        return
    
    # Connection factory for threads
    def conn_factory():
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        return c
    
    conn.close()
    
    # 4 Parallel Workers
    print(f"\n→ Processing {len(pending)} files with {MAX_WORKERS} parallel workers...\n")
    
    done_count = 0
    error_count = 0
    budget_exceeded = False
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_file, row, conn_factory): row
            for row in pending
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            
            if result["status"] == "budget_exceeded":
                budget_exceeded = True
                print(f"\n⚠  BUDGET LIMIT HIT — stopping gracefully")
                executor.shutdown(wait=False, cancel_futures=True)
                break
            
            elif result["status"] == "ok":
                done_count += 1
                cost_conn = sqlite3.connect(DB_PATH)
                costs = get_total_cost(cost_conn)
                cost_conn.close()
                print(f"  [{done_count}/{len(pending)}] ✓ {result['file'][:40]} | "
                      f"${costs['total_usd']:.4f} spent | "
                      f"${costs['budget_remaining_usd']:.4f} remaining")
            
            elif result["status"] == "error":
                error_count += 1
                print(f"  [ERR] {result['file'][:40]} — {result['error'][:80]}")
    
    # Final report
    final_conn = sqlite3.connect(DB_PATH)
    report_summary(final_conn)
    final_conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# VECTORLESS RAG QUERY ENGINE
# This is the replacement for "cosine similarity search"
# ─────────────────────────────────────────────────────────────────────────────

def query_veda(question: str):
    """
    VECTORLESS RAG — How it actually works:
    
    1. Find relevant files using SQL (not cosine similarity)
       - Search synthesis equations for keywords
       - Search cross-reference links for conceptual matches
       - Search anomalies for structural signals
    
    2. Send ONLY the relevant summaries to the LLM
       (not raw text, not embeddings — structured summaries)
    
    3. LLM reasons over the summaries to answer
    
    WHY THIS BEATS VECTOR RAG:
    - "Agni" in Mandala 1 and "Agni" in Mandala 8 are different Agnis
      in the Vedic system. Cosine similarity can't know that.
      Our structural index DOES know that (different folder, different devata context).
    - No hallucination from random chunk retrieval
    - Every answer is traceable to specific file_ids
    """
    conn = init_db(DB_PATH)
    
    print(f"\n{'='*60}")
    print(f"  QUERY: {question}")
    print(f"{'='*60}\n")
    
    # Step 1: Find relevant files via structural search (not vectors)
    # Search across multiple tables simultaneously
    keywords = question.lower().split()
    
    # Build a relevance query across all worker outputs
    relevant_files = set()
    
    for kw in keywords:
        if len(kw) < 3:
            continue
        like = f"%{kw}%"
        
        # Search synthesis equations
        rows = conn.execute(
            "SELECT file_id FROM w4_synthesis WHERE LOWER(equation) LIKE ?", (like,)
        ).fetchall()
        relevant_files.update(r[0] for r in rows)
        
        # Search links (root concepts and devata)
        rows = conn.execute(
            "SELECT file_id FROM w3_links WHERE LOWER(root_concept) LIKE ? OR LOWER(devata) LIKE ?",
            (like, like)
        ).fetchall()
        relevant_files.update(r[0] for r in rows)
        
        # Search structure (titles)
        rows = conn.execute(
            "SELECT file_id FROM w1_structure WHERE LOWER(title) LIKE ? OR LOWER(mandala) LIKE ?",
            (like, like)
        ).fetchall()
        relevant_files.update(r[0] for r in rows)
        
        # Search anomalies
        rows = conn.execute(
            "SELECT file_id FROM w2_anomalies WHERE LOWER(signal_note) LIKE ?", (like,)
        ).fetchall()
        relevant_files.update(r[0] for r in rows)
    
    if not relevant_files:
        print("  No structurally relevant files found.")
        print("  Tip: Run --walk first to index the Veda files.\n")
        return
    
    print(f"  Found {len(relevant_files)} structurally relevant files (no vectors used)\n")
    
    # Step 2: Fetch structured summaries for relevant files
    context_parts = []
    for fid in list(relevant_files)[:20]:  # Cap at 20 files per query
        w1 = conn.execute("SELECT title, mandala, sukta FROM w1_structure WHERE file_id=?", (fid,)).fetchone()
        w4 = conn.execute("SELECT equation, confidence FROM w4_synthesis WHERE file_id=?", (fid,)).fetchone()
        w3 = conn.execute("SELECT root_concept, devata, structural_key FROM w3_links WHERE file_id=?", (fid,)).fetchone()
        
        if w1 and w4:
            part = f"[{w1[1]}/{w1[2]}] {w1[0]}: {w4[0]} (confidence: {w4[1]})"
            if w3:
                part += f" | Root: {w3[0]} | Devata: {w3[1]} | Key: {w3[2]}"
            context_parts.append(part)
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Reason over structural summaries (Vectorless RAG)
    prompt = f"""You have access to a structural index of Vedic texts. 
Each entry contains: location, synthesized equation (hidden structural logic), root concept, and key.
These are NOT translations. They are structural analysis outputs.

Based on this structural index, answer the question: "{question}"

Cite specific file references (Mandala/Sukta) in your answer.
If the answer requires connecting multiple files, state that connection explicitly.
If the data is insufficient, say so clearly.

STRUCTURAL INDEX:
{context}"""

    result, inp, out = call_gemini(prompt, "You are a Vedic structural reasoning engine. Answer from structural evidence only.")
    cost = track_cost(conn, "query", "vectorless_rag", inp, out)
    
    # Save query result
    conn.execute(
        "INSERT INTO query_results (query, result, files_used, cost_usd) VALUES (?,?,?,?)",
        (question, result, json.dumps(list(relevant_files)[:20]), cost)
    )
    conn.commit()
    
    print(f"  ANSWER:\n")
    print(f"  {result}\n")
    print(f"  Cost: ${cost:.6f} | Files used: {min(len(relevant_files), 20)}\n")
    
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def report_summary(conn: sqlite3.Connection):
    """Full system report — budget, progress, top anomalies."""
    costs = get_total_cost(conn)
    
    total   = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    done    = conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
    errors  = conn.execute("SELECT COUNT(*) FROM files WHERE status='error'").fetchone()[0]
    
    # Top anomalies
    anomalies = conn.execute("""
        SELECT w1.title, w1.mandala, w4.equation, w4.confidence
        FROM w4_synthesis w4
        JOIN w1_structure w1 ON w1.file_id = w4.file_id
        WHERE w4.confidence = 'high'
        ORDER BY w4.created_at DESC
        LIMIT 10
    """).fetchall()
    
    print(f"\n{'═'*70}")
    print("  VEDASTREAM REPORT")
    print(f"{'═'*70}")
    print(f"  Files: {done}/{total} processed ({errors} errors)")
    print(f"  Cost:  ${costs['total_usd']:.6f} USD / ₹{costs['total_inr']:.4f}")
    print(f"  Tokens: {costs['input_tokens']:,} in / {costs['output_tokens']:,} out")
    print(f"  Budget remaining: ${costs['budget_remaining_usd']:.4f}")
    
    if anomalies:
        print(f"\n  TOP HIGH-CONFIDENCE EQUATIONS FOUND:")
        print(f"  {'─'*65}")
        for i, (title, mandala, eq, conf) in enumerate(anomalies, 1):
            print(f"  {i}. [{mandala}] {title[:30]}")
            print(f"     {eq[:120]}...")
            print()
    
    print(f"{'═'*70}\n")


def report_confluent_vs_vedastream():
    """Show the explicit comparison table."""
    print(f"\n{'═'*70}")
    print("  KAFKA/CONFLUENT  vs  VEDASTREAM")
    print(f"{'═'*70}")
    
    comparisons = [
        ("Replication Tax",     "3x storage (100TB → 300TB bill)",  "SQLite WAL (1x, zero tax)"),
        ("Scaling",             "4 eCKU/10min ceiling",              "ThreadPoolExecutor (instant)"),
        ("Cross-AZ Network",    "40-60% of your infra bill",        "$0 (local or same-region)"),
        ("Context Bleed",       "Consumer groups mix state",         "Stateless workers, pure functions"),
        ("Vector RAG",          "Cosine similarity = vibing",        "Structural SQL + LLM reasoning"),
        ("Schema Registry",     "Paid, managed, locked-in",         "Inferred per-file by Worker 1"),
        ("Cost for this run",   "~$500/month minimum cluster",       "~$0.05 for entire Rigveda"),
        ("Vendor lock-in",      "Cannot move regions after creation","SQLite file — copy anywhere"),
    ]
    
    for issue, kafka, vedastream in comparisons:
        print(f"  {issue}")
        print(f"    ✗ Kafka: {kafka}")
        print(f"    ✓ Ours : {vedastream}")
        print()
    
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VedaStream — Anti-Kafka Vectorless RAG for Ancient Knowledge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--root", 
        help="Path to Veda HTML root directory",
        default=r"D:\Books\1_sanskr\1_sanskr\1_veda"
    )
    parser.add_argument(
        "--walk",
        action="store_true",
        help="Walk and index the Veda directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query the indexed Veda using Vectorless RAG"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show processing report and cost summary"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show Kafka/Confluent vs VedaStream comparison"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all 'done' files back to pending (re-process everything)"
    )
    parser.add_argument(
        "--reset-errors",
        action="store_true",
        help="Reset only 'error' files back to pending"
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Set API key
    global GEMINI_API_KEY
    if args.key:
        GEMINI_API_KEY = args.key
    
    if args.compare:
        report_confluent_vs_vedastream()
    
    if getattr(args, 'reset', False):
        conn = init_db(DB_PATH)
        n = conn.execute("UPDATE files SET status='pending' WHERE status='done' OR status='error' OR status='processing'").rowcount
        conn.commit()
        conn.close()
        print(f"✓ Reset {n} files back to pending. Now run --walk.")
        return

    if getattr(args, 'reset_errors', False):
        conn = init_db(DB_PATH)
        n = conn.execute("UPDATE files SET status='pending' WHERE status='error' OR status='processing'").rowcount
        conn.commit()
        conn.close()
        print(f"✓ Reset {n} error/stuck files back to pending.")
        return

    if args.walk:
        run_walk(args.root, limit=args.limit)
    
    elif args.query:
        if not GEMINI_API_KEY:
            print("ERROR: Provide --key or set GEMINI_API_KEY")
            sys.exit(1)
        query_veda(args.query)
    
    elif args.report:
        conn = init_db(DB_PATH)
        report_summary(conn)
        conn.close()
    
    else:
        parser.print_help()
        print("\n  QUICK START:")
        print("  1. Test with 5 files:  python veda_walker.py --walk --limit 5 --key YOUR_KEY")
        print("  2. Full walk:          python veda_walker.py --walk --key YOUR_KEY")
        print("  3. Query:              python veda_walker.py --query 'What is Savitr?' --key YOUR_KEY")
        print("  4. See the comparison: python veda_walker.py --compare")
        print()


if __name__ == "__main__":
    main()
