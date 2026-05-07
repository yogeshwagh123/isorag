"""
veda_complete_pipeline.py  — v5 LOCAL
=======================================
ONE FILE. LOCAL OLLAMA. FREE. NO BUDGET LIMIT.
FULL RESUME SUPPORT — stop anytime, restart exactly where you left off.

Stage 1:  Qwen2.5:7b via Ollama, 4 parallel workers
          41,938 verses, 505 problems, 4 lenses in one call
          Free. Runs overnight.
          Every completed verse saved immediately to disk.
          Ctrl+C anytime. Restart with same command. No work lost.

Stage 2:  Claude Sonnet 4.6 reviews top gold entries
          One batch call, reviews top 50 gold entries together

Stage 3:  Claude Opus writes markdown solution files
          Only what Sonnet flags as genuinely interesting

SETUP:
    pip install anthropic pandas
    ollama serve
    ollama pull qwen2.5:7b

USAGE:
    # Stage 1 only (free, ~25 hours, 4 workers):
    python veda_complete_pipeline.py --walk --workers 4

    # Stage 2+3 after walk completes:
    python veda_complete_pipeline.py --solve

    # Both stages in sequence:
    python veda_complete_pipeline.py --run --workers 4

    # Check progress anytime:
    python veda_complete_pipeline.py --report

    # Read a solution:
    python veda_complete_pipeline.py --read 39
    python veda_complete_pipeline.py --read 40
    python veda_complete_pipeline.py --read 390
"""

import os, sys, json, time, re, sqlite3
import threading, argparse, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

try:
    import anthropic as _anthropic
    CLAUDE_AVAIL = True
except ImportError:
    CLAUDE_AVAIL = False
    print("WARNING: pip install anthropic  (Stage 2+3 disabled)")

try:
    import pandas as pd
except ImportError:
    print("FATAL: pip install pandas"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────
CLAUDE_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-6"

SOURCE_DB    = "vedastream.db"
PIPELINE_DB  = "veda_complete.db"
PROBLEMS_CSV = "pure_sciences_500.csv"
OUTPUT_DIR   = Path("solved_problems")

MIN_VERSE          = 40
MAX_VERSE          = 500
GOLD_REBUILD_EVERY = 200
SKIP_PATTERNS      = ['http', 'gretil', 'www.', 'Input by',
                      'Based on', 'edition by', 'GRETIL', '.pdf']

# Claude pricing
SONNET_IN  = 3.00  / 1_000_000
SONNET_OUT = 15.00 / 1_000_000
OPUS_IN    = 15.00 / 1_000_000
OPUS_OUT   = 75.00 / 1_000_000

_spent = 0.0
_lock  = threading.Lock()

# ── Database ──────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(PIPELINE_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""CREATE TABLE IF NOT EXISTS verses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id TEXT NOT NULL, verse_idx INTEGER NOT NULL,
        mandala TEXT, sukta TEXT, text TEXT NOT NULL,
        n_chars INTEGER, UNIQUE(file_id, verse_idx))""")

    conn.execute("""CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        verse_id INTEGER NOT NULL, lens TEXT NOT NULL,
        provider TEXT NOT NULL, problem_id INTEGER NOT NULL,
        problem_name TEXT, problem_branch TEXT, problem_status TEXT,
        relevance INTEGER DEFAULT 0, equation TEXT,
        reasoning TEXT, testable TEXT,
        is_generic INTEGER DEFAULT 0,
        created_at REAL DEFAULT (unixepoch()),
        UNIQUE(verse_id, lens, problem_id))""")

    conn.execute("""CREATE TABLE IF NOT EXISTS gold (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL, problem_name TEXT,
        problem_branch TEXT, problem_status TEXT,
        equation TEXT, lens TEXT,
        n_sources INTEGER DEFAULT 0, n_verses INTEGER DEFAULT 0,
        avg_relevance REAL DEFAULT 0, sources_list TEXT,
        example_verse TEXT,
        created_at REAL DEFAULT (unixepoch()),
        UNIQUE(problem_id, equation))""")

    conn.execute("""CREATE TABLE IF NOT EXISTS sonnet_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL, problem_name TEXT,
        equation TEXT, n_sources INTEGER, avg_relevance REAL,
        sonnet_verdict TEXT,
        sonnet_interest INTEGER DEFAULT 0,
        sonnet_reasoning TEXT,
        send_to_opus INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0,
        created_at REAL DEFAULT (unixepoch()),
        UNIQUE(problem_id, equation))""")

    conn.execute("""CREATE TABLE IF NOT EXISTS solutions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        problem_id INTEGER NOT NULL, problem_name TEXT,
        model_used TEXT, markdown_path TEXT,
        cost_usd REAL DEFAULT 0,
        created_at REAL DEFAULT (unixepoch()))""")

    # proc_status: written immediately after EVERY verse completes
    # This is the resume table — restart reads this first
    conn.execute("""CREATE TABLE IF NOT EXISTS proc_status (
        verse_id INTEGER PRIMARY KEY,
        done INTEGER DEFAULT 0,
        n_matches INTEGER DEFAULT 0,
        completed_at REAL DEFAULT (unixepoch()))""")

    conn.execute("""CREATE TABLE IF NOT EXISTS cost_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage TEXT, model TEXT,
        inp INTEGER DEFAULT 0, out INTEGER DEFAULT 0,
        cost_usd REAL DEFAULT 0,
        ts REAL DEFAULT (unixepoch()))""")

    conn.commit()
    OUTPUT_DIR.mkdir(exist_ok=True)
    return conn


# ── Verse extraction ──────────────────────────────────────────
def extract_verses(text):
    if not text or len(text) < MIN_VERSE:
        return []
    parts = re.split(r'\s*(?://|॥|\|\|)\s*', text)
    verses = [p.strip() for p in parts
              if MIN_VERSE <= len(p.strip()) <= MAX_VERSE]
    if len(verses) >= 2:
        return verses
    parts = re.split(r'[।.!?]+', text)
    verses = [p.strip() for p in parts
              if MIN_VERSE <= len(p.strip()) <= MAX_VERSE]
    if len(verses) >= 2:
        return verses
    verses, chunk = [], []
    for word in text.split():
        chunk.append(word)
        j = ' '.join(chunk)
        if len(j) >= 150:
            if len(j) <= MAX_VERSE:
                verses.append(j)
            chunk = []
    if chunk:
        j = ' '.join(chunk)
        if MIN_VERSE <= len(j):
            verses.append(j[:MAX_VERSE])
    return verses


def index_verses(conn):
    already = conn.execute(
        "SELECT COUNT(*) FROM verses").fetchone()[0]
    if already > 0:
        print(f"  Using {already:,} indexed verses")
        return already
    if not os.path.exists(SOURCE_DB):
        print(f"FATAL: {SOURCE_DB} not found"); sys.exit(1)
    src = sqlite3.connect(SOURCE_DB)
    rows = src.execute(
        "SELECT file_id, mandala, sukta, clean_text "
        "FROM w1_structure WHERE length(clean_text) > 100").fetchall()
    src.close()
    total = 0
    for fid, mandala, sukta, text in rows:
        for idx, verse in enumerate(extract_verses(text or "")):
            if any(p in verse for p in SKIP_PATTERNS):
                continue
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO verses "
                    "(file_id,verse_idx,mandala,sukta,text,n_chars) "
                    "VALUES (?,?,?,?,?,?)",
                    (fid, idx, mandala or "?",
                     sukta or "?", verse, len(verse)))
                total += 1
            except Exception:
                pass
        if total % 5000 == 0 and total > 0:
            conn.commit()
            print(f"  Indexed {total:,} verses...")
    conn.commit()
    print(f"  Total: {total:,} verses indexed")
    return total


# ── Problems ──────────────────────────────────────────────────
def load_problems():
    if not os.path.exists(PROBLEMS_CSV):
        print(f"FATAL: {PROBLEMS_CSV} not found"); sys.exit(1)
    return pd.read_csv(PROBLEMS_CSV)


def make_problem_block(df):
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"[{int(r['ID'])}]{r['Key_Problem_or_Concept']}:"
            f"{str(r['Description'])[:80]}")
    return "\n".join(lines)


GENERIC_EQ = [
    r'^dX/dt\s*=\s*f\(X\)$', r'^Y\s*\\?propto\s*X\^n$',
    r'^f\(T\(x\)\)\s*=\s*f\(x\)$', r'^dX/dt\s*=\s*0$',
    r'^X_before\s*=\s*X_after$',
]

def is_generic(eq):
    if not eq or len(eq.strip()) < 8:
        return True
    for pat in GENERIC_EQ:
        if re.match(pat, eq.strip(), re.IGNORECASE):
            return True
    return False


# ── Ollama call ───────────────────────────────────────────────
def call_ollama(prompt, system=""):
    """
    Single Ollama call. Returns (text, input_tokens, output_tokens).
    Retries 3 times on failure.
    """
    full_prompt = (system + "\n\n" + prompt) if system else prompt
    data = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  full_prompt,
        "stream":  False,
        "options": {
            "temperature":  0.1,
            "num_predict":  512,   # cap output tokens for speed
            "num_ctx":      4096,  # context window
        }
    }).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            txt = result.get("response", "").strip()
            # Estimate tokens from context info
            inp = result.get("prompt_eval_count",  len(full_prompt) // 4)
            out = result.get("eval_count",          len(txt) // 4)
            return txt, inp, out
        except Exception as e:
            if attempt == 2:
                return f"ERR:{e}", 0, 0
            time.sleep(5 * (attempt + 1))
    return "ERR:retries", 0, 0


def call_claude(prompt, system, use_opus=False):
    if not CLAUDE_AVAIL or not CLAUDE_KEY:
        return "NO_CLAUDE", 0, 0, SONNET
    model  = OPUS if use_opus else SONNET
    client = _anthropic.Anthropic(api_key=CLAUDE_KEY)
    try:
        r = client.messages.create(
            model=model, max_tokens=8000, system=system,
            messages=[{"role": "user", "content": prompt}])
        txt = r.content[0].text
        return txt, r.usage.input_tokens, r.usage.output_tokens, model
    except Exception as e:
        return f"ERR:{e}", 0, 0, model


def safe_json(text, want_list=True):
    empty = [] if want_list else {}
    if not text or str(text).startswith("ERR"):
        return empty
    if text in ("NO_CLAUDE",):
        return empty
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        r = json.loads(text)
        if want_list:
            if isinstance(r, list): return r
            if isinstance(r, dict):
                for v in r.values():
                    if isinstance(v, list): return v
            return []
        else:
            return r if isinstance(r, dict) else {}
    except Exception:
        pat = r'\[.*?\]' if want_list else r'\{.*\}'
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group())
                if want_list: return r if isinstance(r, list) else []
                else:         return r if isinstance(r, dict) else {}
            except Exception:
                pass
    return empty


def track_claude(conn, stage, model, inp, out):
    global _spent
    if "opus" in model:
        cost = inp * OPUS_IN + out * OPUS_OUT
    else:
        cost = inp * SONNET_IN + out * SONNET_OUT
    with _lock:
        _spent += cost
    conn.execute(
        "INSERT INTO cost_log (stage,model,inp,out,cost_usd) "
        "VALUES (?,?,?,?,?)", (stage, model, inp, out, cost))
    conn.commit()
    return cost


# ── Stage 1: Walker prompt ────────────────────────────────────
WALKER_SYSTEM = """You are a mathematical physicist.
Output ONLY valid JSON array. No markdown fences.
Empty array [] if nothing qualifies."""

def make_walker_prompt(verse, prob_block):
    return f"""VERSE:
{verse}

Analyze through 4 mathematical lenses. Find matching open problems.

LENS 1 CONSERVATION: quantity preserved while others change.
  Write: d[X]/dt + nabla·([X]v) = 0  or  sum([X]) = const
  Example: d(rho)/dt + nabla·(rho*u) = 0

LENS 2 SYMMETRY: invariant under named transformation.
  Write: f(T(x)) = f(x) where T is NAMED explicitly.
  Example: u(x,t) = u(x+L,t) [translation by L]

LENS 3 SCALING: power law with SPECIFIC numeric exponent.
  Example: E(k) proportional to k^(-5/3)
  REJECT if exponent is just n without a value.

LENS 4 DYNAMICS: rate equation with specific terms named.
  Example: du/dt = -(u·nabla)u - nabla(p)/rho + nu*nabla^2(u)
  REJECT generic dX/dt = f(X).

RULES:
- relevance >= 5 only
- equations must be specific, no placeholders
- maximum 5 matches total
- return [] if nothing qualifies

JSON array:
[{{"lens":"conservation","problem_id":39,"relevance":7,
"equation":"d(rho)/dt + nabla·(rho*u)=0",
"reasoning":"one sentence","testable":"one computation"}}]

PROBLEMS:
{prob_block}"""


# ── Stage 1: per-verse processor ─────────────────────────────
def process_verse(row, prob_block, problems_df, conn):
    """
    Process one verse. Called by worker thread.
    Saves result to proc_status IMMEDIATELY after completion.
    This is the resume guarantee — if you stop mid-run,
    this verse will be marked done and skipped on restart.
    """
    vid, mandala, sukta, verse_text = row
    matches = 0

    prompt = make_walker_prompt(verse_text, prob_block)
    txt, inp, out = call_ollama(prompt, WALKER_SYSTEM)
    items = safe_json(txt, want_list=True)

    for item in items:
        if not isinstance(item, dict): continue
        pid  = item.get("problem_id")
        eq   = item.get("equation", "").strip()
        rel  = int(item.get("relevance", 0))
        lens = item.get("lens", "combined")
        if not pid or not eq or rel < 5: continue
        prob = problems_df[problems_df['ID'] == pid]
        if prob.empty: continue
        p = prob.iloc[0]
        try:
            conn.execute("""INSERT OR REPLACE INTO matches
                (verse_id,lens,provider,problem_id,problem_name,
                 problem_branch,problem_status,relevance,equation,
                 reasoning,testable,is_generic)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (vid, lens, "ollama", int(pid),
                 p['Key_Problem_or_Concept'],
                 p['Sub-Branch'],
                 p['Status'],
                 rel, eq,
                 str(item.get("reasoning", ""))[:250],
                 str(item.get("testable",  ""))[:250],
                 1 if is_generic(eq) else 0))
            conn.commit()
            matches += 1
        except Exception:
            pass

    # RESUME GUARANTEE: write done status immediately
    # Even if process crashes after this line,
    # this verse will be skipped on restart
    conn.execute("""INSERT OR REPLACE INTO proc_status
        (verse_id, done, n_matches, completed_at)
        VALUES (?, 1, ?, unixepoch())""",
        (vid, matches))
    conn.commit()
    return matches


# ── Gold rebuild ──────────────────────────────────────────────
def rebuild_gold(conn):
    conn.execute("DELETE FROM gold")
    conn.commit()
    rows = conn.execute("""
        SELECT m.problem_id, m.problem_name,
               m.problem_branch, m.problem_status,
               m.equation, m.lens,
               COUNT(DISTINCT v.mandala) as nsrc,
               COUNT(DISTINCT m.verse_id) as nv,
               AVG(m.relevance) as ar,
               GROUP_CONCAT(DISTINCT v.mandala) as srcs,
               MIN(v.text) as ex
        FROM matches m
        JOIN verses v ON v.id = m.verse_id
        WHERE m.is_generic = 0
          AND m.relevance >= 6
          AND v.n_chars >= 60
        GROUP BY m.problem_id, m.equation
        HAVING nsrc >= 2
        ORDER BY nsrc DESC, ar DESC
    """).fetchall()
    n = 0
    for row in rows:
        try:
            conn.execute("""INSERT OR REPLACE INTO gold
                (problem_id,problem_name,problem_branch,problem_status,
                 equation,lens,n_sources,n_verses,avg_relevance,
                 sources_list,example_verse)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", row)
            n += 1
        except Exception:
            pass
    conn.commit()
    return n


# ── Stage 1: walk ─────────────────────────────────────────────
def run_walk(workers=4):
    import concurrent.futures

    # Check Ollama is running
    try:
        urllib.request.urlopen(
            f"{OLLAMA_HOST}/api/tags", timeout=5)
    except Exception:
        print(f"FATAL: Ollama not running at {OLLAMA_HOST}")
        print("Start it with: ollama serve")
        sys.exit(1)

    # Check model is available
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            tags = json.loads(resp.read())
        models = [m['name'] for m in tags.get('models', [])]
        model_ok = any(OLLAMA_MODEL in m for m in models)
        if not model_ok:
            print(f"FATAL: Model {OLLAMA_MODEL} not found in Ollama")
            print(f"Available: {models}")
            print(f"Run: ollama pull {OLLAMA_MODEL}")
            sys.exit(1)
    except Exception:
        pass  # if we can't check, proceed anyway

    conn        = init_db()
    n_verses    = index_verses(conn)
    problems_df = load_problems()
    prob_block  = make_problem_block(problems_df)

    # Resume: get pending verses
    done_ids = set(r[0] for r in conn.execute(
        "SELECT verse_id FROM proc_status WHERE done=1").fetchall())
    all_rows = conn.execute(
        "SELECT id, mandala, sukta, text FROM verses").fetchall()
    pending = [(r[0], r[1], r[2], r[3])
               for r in all_rows if r[0] not in done_ids]

    # Time estimate
    # Qwen2.5 7B: ~25 tokens/sec, ~800 tokens per verse = ~32 sec/verse
    # With N workers: 32/N sec per verse
    sec_per_verse = 32 / max(workers, 1)
    eta_hours = len(pending) * sec_per_verse / 3600

    print(f"\n{'='*65}")
    print(f"  VEDA WALKER — Qwen2.5:7b LOCAL  FREE")
    print(f"{'='*65}")
    print(f"  Verses total:      {n_verses:,}")
    print(f"  Already done:      {len(done_ids):,}")
    print(f"  Pending:           {len(pending):,}")
    print(f"  Workers:           {workers}")
    print(f"  Model:             {OLLAMA_MODEL}")
    print(f"  Cost:              $0.00 (local)")
    print(f"  Est. time:         {eta_hours:.1f} hours")
    print(f"  Resume:            YES — Ctrl+C anytime, restart same command")
    print(f"  DB:                {PIPELINE_DB}")
    print(f"{'='*65}")
    print(f"  Progress saved after EVERY verse.")
    print(f"  Stop and restart anytime with same command.\n")

    start    = time.time()
    done_cnt = 0
    total_m  = 0

    try:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as ex:
            futures = {
                ex.submit(process_verse, row, prob_block,
                          problems_df, conn): row
                for row in pending
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    n        = future.result()
                    done_cnt += 1
                    total_m  += n
                    elapsed  = time.time() - start
                    rate     = done_cnt / elapsed if elapsed > 0 else 1
                    remaining= len(pending) - done_cnt
                    eta_h    = remaining / rate / 3600

                    if done_cnt % GOLD_REBUILD_EVERY == 0:
                        ng = rebuild_gold(conn)
                        total_done = len(done_ids) + done_cnt
                        pct = total_done / n_verses * 100
                        print(
                            f"  [{total_done:>6}/{n_verses}] "
                            f"{pct:.1f}%  "
                            f"matches={total_m:>6}  "
                            f"gold={ng}  "
                            f"ETA={eta_h:.1f}h  "
                            f"rate={rate:.1f}v/s"
                        )
                except Exception as e:
                    print(f"  ERR: {e}")

    except KeyboardInterrupt:
        print(f"\n  Stopped by user.")
        print(f"  Progress saved. Restart with same command to continue.")

    ng = rebuild_gold(conn)
    total_done = len(done_ids) + done_cnt
    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  Walk session complete.")
    print(f"  This session:  {done_cnt:,} verses in {elapsed/3600:.1f}h")
    print(f"  Total done:    {total_done:,} / {n_verses:,}")
    print(f"  Gold entries:  {ng}")
    print(f"  Run --report for full summary")
    print(f"  Run --solve when walk is complete")
    print(f"{'='*65}")
    conn.close()


# ── Stage 2: Sonnet batch review ──────────────────────────────
SONNET_SYS = """You are a mathematical physicist reviewing corpus analysis.
Decide which gold entries deserve deep analysis by Claude Opus.
Be selective. Most entries are pattern-matching noise.
Flag ONLY entries with specific equations, diverse sources,
and plausible structural connection to the mathematics.
Respond ONLY in valid JSON array."""

SONNET_PROMPT = """Review these gold entries from Sanskrit corpus analysis.
Each entry: a mathematical equation found across multiple independent sources.

For each, decide: send to Opus for deep analysis?

{gold_block}

For each entry output JSON object:
{{"problem_id":N,"equation":"...","interest_score":0-10,
  "send_to_opus":true/false,"reasoning":"one sentence"}}

Rules:
- send_to_opus=true only if interest_score >= 7
- Be brutal. Most are noise.
- Flag only genuine mathematical structure.

JSON array only."""


# ── Stage 3: Opus solver ──────────────────────────────────────
OPUS_SYS = """You are the world's most capable mathematical physicist.
Attempt to solve or significantly advance the associated problem.
A rigorous partial result beats a speculative complete solution.
Partial progress is real progress. Nothing is thrown away.
Write structured markdown."""

OPUS_PROMPT = """# Problem #{pid}: {pname}

**Status:** {pstat}
**Corpus equation:** `{eq}`
**Independent sources:** {nsrc}
**Average relevance:** {ar:.1f}/10
**Sources:** {srcs}

**Example verse:**
> {verse}

**Sonnet's assessment:** {sonnet_note}

## Your Task

### 1. Assess the Evidence
Is `{eq}` a meaningful structural signal or pattern-matching noise?
State your reasoning precisely.

### 2. Formalise
If meaningful: write as a theorem statement.
Define the space, measure, operators, domain precisely.

### 3. Attempt Proof
Use: Noether's theorem, Information Bottleneck (Tishby),
Kolmogorov 1941, functional analysis, Lie theory, or whatever applies.
Stop exactly where the proof breaks down.
That breakdown point is the most valuable output.

### 4. Partial Results
What CAN be proven rigorously right now? Even a lemma.

### 5. Millennium Prize Connection
Does this imply anything about Navier-Stokes global regularity?
Yang-Mills mass gap? Riemann Hypothesis?
Logical chain required. Not analogy.

### 6. Falsification
ONE specific numerical computation that would disprove this.
Give exact values.

### 7. Next Steps
Specific papers (author, title, year).
Specific mathematicians working in this area today.

### 8. Honest Assessment
- Signal or noise: signal/noise/unclear
- Mathematical rigour of proposal: X/10
- Probability this leads somewhere: X/10
- Confidence in partial results: X/10"""


def run_solve():
    global _spent

    conn = init_db()
    ng   = rebuild_gold(conn)

    print(f"\n  Gold entries: {ng}")

    if ng == 0:
        print("  No gold entries. Run --walk first.")
        conn.close(); return

    # Get top gold for Sonnet review
    gold = conn.execute("""
        SELECT problem_id, problem_name, problem_status,
               equation, lens, n_sources, avg_relevance,
               sources_list, example_verse
        FROM gold
        WHERE n_sources >= 3 AND avg_relevance >= 7
        ORDER BY n_sources DESC, avg_relevance DESC
        LIMIT 60
    """).fetchall()

    if not gold:
        print("  No gold with n_sources>=3 and rel>=7")
        print("  Try: --report to see what is available")
        conn.close(); return

    # Skip already reviewed
    already_reviewed = set(r[0] for r in conn.execute(
        "SELECT problem_id FROM sonnet_reviews").fetchall())

    new_gold = [(pid,pname,pstat,eq,lens,nsrc,ar,srcs,verse)
                for pid,pname,pstat,eq,lens,nsrc,ar,srcs,verse in gold
                if pid not in already_reviewed]

    print(f"  Sending {len(new_gold)} new entries to Sonnet...")

    if not new_gold:
        print("  All entries already reviewed.")
    else:
        if not CLAUDE_AVAIL or not CLAUDE_KEY:
            print("  ERROR: ANTHROPIC_API_KEY not set")
            conn.close(); return

        # Build gold block
        gold_block = ""
        for pid,pname,pstat,eq,lens,nsrc,ar,srcs,verse in new_gold:
            gold_block += (
                f"\nProblem #{pid}: {pname} ({pstat})\n"
                f"Equation [{lens}]: {eq}\n"
                f"Sources: {nsrc} independent | Avg relevance: {ar:.1f}\n"
                f"From: {(srcs or '')[:100]}\n"
                f"Verse: {(verse or '')[:150]}\n---")

        txt, inp, out, model = call_claude(
            SONNET_PROMPT.format(gold_block=gold_block),
            SONNET_SYS, use_opus=False)
        cost = track_claude(conn, "s2_sonnet", model, inp, out)
        print(f"  Sonnet cost: ${cost:.4f}")

        reviews = safe_json(txt, want_list=True)
        for_opus = []

        for rev in reviews:
            if not isinstance(rev, dict): continue
            pid   = rev.get("problem_id")
            eq    = rev.get("equation", "")
            score = int(rev.get("interest_score", 0))
            send  = rev.get("send_to_opus", False)
            reason= rev.get("reasoning", "")

            gold_row = conn.execute(
                "SELECT problem_name,problem_status,n_sources,"
                "avg_relevance,sources_list,example_verse "
                "FROM gold WHERE problem_id=? LIMIT 1",
                (pid,)).fetchone()
            if not gold_row: continue

            pname = gold_row[0]
            print(f"  #{pid} {(pname or '')[:40]:<40} "
                  f"score={score} "
                  f"{'-> OPUS' if send else 'skip'} "
                  f"— {reason[:50]}")

            try:
                conn.execute("""INSERT OR REPLACE INTO sonnet_reviews
                    (problem_id,problem_name,equation,n_sources,
                     avg_relevance,sonnet_verdict,sonnet_interest,
                     sonnet_reasoning,send_to_opus,cost_usd)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (pid, pname, eq, gold_row[2], gold_row[3],
                     "interesting" if send else "noise",
                     score, reason, 1 if send else 0,
                     cost / max(len(reviews), 1)))
                conn.commit()
            except Exception:
                pass

            if send and score >= 7:
                for_opus.append((pid, pname,
                                 gold_row[1],
                                 eq,
                                 gold_row[2],
                                 gold_row[3],
                                 gold_row[4],
                                 gold_row[5],
                                 reason))

        print(f"\n  Sonnet flagged {len(for_opus)} entries for Opus")

    # Also pick up any previously flagged not yet solved
    flagged = conn.execute("""
        SELECT sr.problem_id, sr.problem_name, sr.equation,
               sr.sonnet_reasoning,
               g.problem_status, g.n_sources, g.avg_relevance,
               g.sources_list, g.example_verse
        FROM sonnet_reviews sr
        JOIN gold g ON g.problem_id = sr.problem_id
        WHERE sr.send_to_opus = 1
          AND sr.problem_id NOT IN (
              SELECT problem_id FROM solutions)
        ORDER BY sr.sonnet_interest DESC
    """).fetchall()

    if not flagged:
        print("  No entries flagged for Opus yet.")
        conn.close(); return

    print(f"  Sending {len(flagged)} entries to Opus...")

    for (pid, pname, eq, note,
         pstat, nsrc, ar, srcs, verse) in flagged:

        print(f"\n  Opus: #{pid} {pname}")

        s_txt, si, so, s_model = call_claude(
            OPUS_PROMPT.format(
                pid=pid, pname=pname, pstat=pstat,
                eq=eq, nsrc=nsrc, ar=ar,
                srcs=(srcs or ""),
                verse=(verse or "")[:300],
                sonnet_note=note),
            OPUS_SYS, use_opus=True)
        s_cost = track_claude(conn, "s3_opus", s_model, si, so)

        safe = re.sub(r'[^\w\s-]', '', pname).strip().replace(' ', '_')
        md_path = OUTPUT_DIR / f"problem_{pid}_{safe}.md"

        md = f"""# Problem #{pid}: {pname}

**Status:** {pstat}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {s_model}
**Cost this analysis:** ${s_cost:.4f}

---

## Corpus Evidence

| Field | Value |
|-------|-------|
| Equation found | `{eq}` |
| Independent sources | {nsrc} |
| Average relevance | {ar:.1f}/10 |

**Sources:** {srcs}

**Example verse:**
> {(verse or 'N/A')[:300]}

**Sonnet review:** {note}

---

## Mathematical Analysis by Claude Opus

{s_txt}

---

*veda_complete_pipeline.py — total cost ${_spent:.4f}*
"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)

        conn.execute("""INSERT INTO solutions
            (problem_id,problem_name,model_used,markdown_path,cost_usd)
            VALUES (?,?,?,?,?)""",
            (pid, pname, s_model, str(md_path), s_cost))
        conn.commit()
        print(f"  Written: {md_path}  (${s_cost:.4f})")

    show_report(conn)
    conn.close()


# ── Report ────────────────────────────────────────────────────
def show_report(conn=None):
    close_after = conn is None
    if conn is None:
        if not os.path.exists(PIPELINE_DB):
            print("No results yet. Run --walk first.")
            return
        conn = sqlite3.connect(PIPELINE_DB)

    print(f"\n{'='*70}")
    print(f"  PIPELINE REPORT")
    print(f"{'='*70}")

    nv   = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
    nd   = conn.execute(
        "SELECT COUNT(*) FROM proc_status WHERE done=1").fetchone()[0]
    nm   = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    ns   = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE is_generic=0").fetchone()[0]
    ng   = conn.execute("SELECT COUNT(*) FROM gold").fetchone()[0]
    nsr  = conn.execute(
        "SELECT COUNT(*) FROM sonnet_reviews "
        "WHERE send_to_opus=1").fetchone()[0]
    nsol = conn.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
    cost = conn.execute(
        "SELECT SUM(cost_usd) FROM cost_log").fetchone()[0] or 0

    pct = nd / nv * 100 if nv else 0
    print(f"\n  Verses indexed:       {nv:,}")
    print(f"  Verses processed:     {nd:,} ({pct:.1f}%)")
    print(f"  Matches found:        {nm:,} ({ns:,} specific)")
    print(f"  Gold entries:         {ng}")
    print(f"  Sonnet flagged:       {nsr}")
    print(f"  Solutions written:    {nsol}")
    print(f"  Claude cost:          ${cost:.4f} (Rs{cost*83.5:.0f})")
    print(f"  Ollama cost:          $0.00 (local)")

    # Resume status
    remaining = nv - nd
    if remaining > 0:
        sec_per_verse = 32 / 4  # estimate
        eta_h = remaining * sec_per_verse / 3600
        print(f"\n  Remaining:            {remaining:,} verses")
        print(f"  Est. time left:       {eta_h:.1f} hours")
        print(f"  Resume:               python veda_complete_pipeline.py --walk --workers 4")

    if nsol > 0:
        print(f"\n  SOLUTIONS:")
        for pid, pname, model, path, scost in conn.execute(
            "SELECT problem_id,problem_name,model_used,"
            "markdown_path,cost_usd FROM solutions "
            "ORDER BY created_at DESC").fetchall():
            ok = "OK" if path and os.path.exists(path) else "MISSING"
            print(f"  [{ok}] #{pid} {pname}")
            print(f"        {model} ${scost:.4f}")
            print(f"        {path}")

    if ng > 0:
        print(f"\n  TOP GOLD (by independent sources):")
        for pid, pname, pstat, nsrc, ar in conn.execute("""
            SELECT problem_id, problem_name, problem_status,
                   n_sources, avg_relevance FROM gold
            ORDER BY n_sources DESC, avg_relevance DESC
            LIMIT 20""").fetchall():
            mark = " *" if pstat and "Unsolved" in str(pstat) else ""
            print(f"  #{pid:>4}  {(pname or ''):<45} "
                  f"src={nsrc}  rel={ar:.1f}{mark}")

    if close_after:
        conn.close()


def read_solution(problem_id):
    if not os.path.exists(PIPELINE_DB):
        print("No results yet."); return
    conn = sqlite3.connect(PIPELINE_DB)
    row  = conn.execute(
        "SELECT markdown_path FROM solutions "
        "WHERE problem_id=? ORDER BY created_at DESC LIMIT 1",
        (problem_id,)).fetchone()
    conn.close()
    if not row or not row[0] or not os.path.exists(row[0]):
        print(f"No solution for problem {problem_id}"); return
    with open(row[0], encoding='utf-8') as f:
        print(f.read())


# ── CLI ───────────────────────────────────────────────────────
def main():
    global CLAUDE_KEY, OLLAMA_MODEL, OLLAMA_HOST

    ap = argparse.ArgumentParser(
        description="Veda Pipeline v5 — Local Ollama, free, full resume",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",          action="store_true",
                    help="Walk then solve automatically")
    ap.add_argument("--walk",         action="store_true",
                    help="Stage 1: walk all verses (free, local)")
    ap.add_argument("--solve",        action="store_true",
                    help="Stage 2+3: Sonnet review then Opus solve")
    ap.add_argument("--report",       action="store_true",
                    help="Show progress and results")
    ap.add_argument("--read",         type=int,
                    help="Print solution markdown for problem ID")
    ap.add_argument("--rebuild-gold", action="store_true",
                    help="Rebuild gold table from existing matches")
    ap.add_argument("--workers",      type=int, default=4,
                    help="Parallel workers (default 4)")
    ap.add_argument("--claude-key",   type=str,
                    help="Anthropic API key")
    ap.add_argument("--ollama-model", type=str,
                    help=f"Ollama model (default: {OLLAMA_MODEL})")
    ap.add_argument("--ollama-host",  type=str,
                    help=f"Ollama host (default: {OLLAMA_HOST})")

    args = ap.parse_args()
    if args.claude_key:  CLAUDE_KEY   = args.claude_key
    if args.ollama_model:OLLAMA_MODEL = args.ollama_model
    if args.ollama_host: OLLAMA_HOST  = args.ollama_host

    if args.run:
        run_walk(workers=args.workers)
        run_solve()
    elif args.walk:
        run_walk(workers=args.workers)
    elif args.solve:
        run_solve()
    elif args.report:
        show_report()
    elif args.read:
        read_solution(args.read)
    elif args.rebuild_gold:
        conn = init_db()
        n = rebuild_gold(conn)
        conn.close()
        print(f"Gold rebuilt: {n} entries")
    else:
        ap.print_help()
        print(f"""
QUICK START:

  # Make sure Ollama is running:
  ollama serve

  # Stage 1 — full walk (free, ~25 hours, resumes anytime):
  python veda_complete_pipeline.py --walk --workers 4

  # Stop anytime with Ctrl+C
  # Restart with the same command — picks up exactly where you left off

  # Stage 2+3 — Sonnet reviews, Opus solves:
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  python veda_complete_pipeline.py --solve

  # Check progress anytime:
  python veda_complete_pipeline.py --report

  # Read solutions:
  python veda_complete_pipeline.py --read 39   # Navier-Stokes
  python veda_complete_pipeline.py --read 40   # Yang-Mills
  python veda_complete_pipeline.py --read 390  # Turbulence

RESUME GUARANTEE:
  Every completed verse is saved to disk immediately.
  Stop with Ctrl+C anytime.
  Restart with same command.
  Zero work lost.

COST:
  Stage 1 Ollama:  $0.00 (local, free)
  Stage 2 Sonnet:  ~$1.00 (one batch call)
  Stage 3 Opus:    ~$3.00 (per flagged problem)
  TOTAL:           ~$4.00

TIME (4 workers, Qwen2.5:7b):
  ~25 hours for full 41,938 verses
  Already done {0} verses — check --report for current count
""")


if __name__ == "__main__":
    main()
