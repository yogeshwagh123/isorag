"""
veda_verse_walker.py
====================
Correct architecture. Verse-level granularity.

THE KEY INSIGHT YOU IDENTIFIED:
    The signal is NOT in the file. It is in the verse.
    One Subhashita. One complete thought. One encoded unit.
    Feed blobs = waste signal. Feed verses = find gold.

ARCHITECTURE:
    Pass 1: All 28,149 verses × 4 Gemini workers × 505 problems
            Cost: ~$23.65
            Output: top N candidates ranked by relevance

    Pass 2: Top 500 candidates × 4 DeepSeek workers (cross-validation)
            Cost: ~$1.80
            Output: consensus table — gold

4 WORKERS, 4 ANGLES, SAME VERSE:
    W1 — Conservation: what is preserved while everything changes?
    W2 — Symmetry: what is invariant under transformation?
    W3 — Scaling: what power law governs the relationship?
    W4 — Dynamics: what differential equation describes evolution?

When 2+ workers flag the same verse independently = structural depth.
When Gemini + DeepSeek agree on same verse + problem = gold.

Usage:
    pip install google-genai openai pandas

    # Test 100 verses (costs $0.08):
    python veda_verse_walker.py --pass1 --limit 100 --key YOUR_KEY

    # Full pass 1 (costs $23.65):
    python veda_verse_walker.py --pass1 --key YOUR_KEY

    # Pass 2 on top 500 (costs $1.80):
    python veda_verse_walker.py --pass2 --top 500 --fw-key YOUR_FW_KEY

    # Results:
    python veda_verse_walker.py --results
    python veda_verse_walker.py --problem 39
"""

import os, sys, json, time, re, sqlite3
import threading, concurrent.futures, argparse
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("FATAL: pip install google-genai"); sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("FATAL: pip install openai"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────
GEMINI_KEY    = os.getenv("GEMINI_API_KEY",    "")
FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY", "")

GEMINI_MODEL    = "gemini-2.5-flash-lite"
DEEPSEEK_MODEL  = "accounts/fireworks/models/deepseek-v3"

SOURCE_DB    = "vedastream.db"
VERSE_DB     = "veda_verses.db"
PROBLEMS_CSV = "pure_sciences_500.csv"

BUDGET_PASS1 = 25.00   # Gemini full corpus
BUDGET_PASS2 = 5.00    # DeepSeek top candidates

GEMINI_IN    = 0.10 / 1_000_000
GEMINI_OUT   = 0.40 / 1_000_000
FW_IN        = 0.90 / 1_000_000
FW_OUT       = 0.90 / 1_000_000

_lock        = threading.Lock()
_spent       = 0.0


# ── Database ──────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(VERSE_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Verse index — the unit of processing
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verses (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    TEXT NOT NULL,
            verse_idx  INTEGER NOT NULL,
            mandala    TEXT,
            sukta      TEXT,
            text       TEXT,
            n_chars    INTEGER,
            UNIQUE(file_id, verse_idx)
        )
    """)

    # Pass 1 results — Gemini, per verse, per worker angle
    conn.execute("""
        CREATE TABLE IF NOT EXISTS p1_matches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id     INTEGER NOT NULL,
            worker       TEXT NOT NULL,
            problem_id   INTEGER NOT NULL,
            problem_name TEXT,
            problem_branch TEXT,
            problem_status TEXT,
            relevance    INTEGER DEFAULT 0,
            equation     TEXT,
            reasoning    TEXT,
            testable     TEXT,
            cost_usd     REAL DEFAULT 0,
            created_at   REAL DEFAULT (unixepoch()),
            UNIQUE(verse_id, worker, problem_id)
        )
    """)

    # Cross-worker consensus — verse flagged by 2+ workers
    conn.execute("""
        CREATE TABLE IF NOT EXISTS p1_consensus (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id     INTEGER NOT NULL,
            problem_id   INTEGER NOT NULL,
            problem_name TEXT,
            problem_status TEXT,
            workers_agreed TEXT,
            n_workers    INTEGER DEFAULT 0,
            mean_relevance REAL DEFAULT 0,
            equations    TEXT,
            verse_text   TEXT,
            mandala      TEXT,
            sukta        TEXT,
            created_at   REAL DEFAULT (unixepoch()),
            UNIQUE(verse_id, problem_id)
        )
    """)

    # Pass 2 results — DeepSeek on top candidates
    conn.execute("""
        CREATE TABLE IF NOT EXISTS p2_matches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id     INTEGER NOT NULL,
            worker       TEXT NOT NULL,
            problem_id   INTEGER NOT NULL,
            relevance    INTEGER DEFAULT 0,
            equation     TEXT,
            reasoning    TEXT,
            testable     TEXT,
            cost_usd     REAL DEFAULT 0,
            created_at   REAL DEFAULT (unixepoch()),
            UNIQUE(verse_id, worker, problem_id)
        )
    """)

    # Final gold table — both Gemini workers AND DeepSeek agree
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id        INTEGER NOT NULL,
            problem_id      INTEGER NOT NULL,
            problem_name    TEXT,
            problem_branch  TEXT,
            problem_status  TEXT,
            equation_gemini TEXT,
            equation_deep   TEXT,
            n_gemini_workers INTEGER,
            relevance_gemini REAL,
            relevance_deep  INTEGER,
            gold_score      REAL,
            verse_text      TEXT,
            mandala         TEXT,
            sukta           TEXT,
            created_at      REAL DEFAULT (unixepoch()),
            UNIQUE(verse_id, problem_id)
        )
    """)

    # Processing status
    conn.execute("""
        CREATE TABLE IF NOT EXISTS p1_status (
            verse_id   INTEGER PRIMARY KEY,
            done       INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS p2_status (
            verse_id   INTEGER PRIMARY KEY,
            done       INTEGER DEFAULT 0
        )
    """)

    # Cost
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id      INTEGER,
            pass_n        INTEGER,
            worker        TEXT,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            cost_usd      REAL,
            created_at    REAL DEFAULT (unixepoch())
        )
    """)

    conn.commit()
    return conn


# ── Verse extraction ──────────────────────────────────────────
def extract_verses(text: str) -> list[str]:
    """
    Split text into verse-level units.
    Priority: Sanskrit verse markers, then sentences, then lines.
    Minimum 40 chars. Maximum 600 chars.
    """
    if not text or len(text) < 40:
        return []

    # Try Sanskrit verse markers first (// ॥ ||)
    parts = re.split(r'\s*(?://|॥|\|\|)\s*', text)
    verses = [p.strip() for p in parts if 40 <= len(p.strip()) <= 600]

    if len(verses) >= 2:
        return verses

    # Try sentence-level (। . ! ?)
    parts = re.split(r'[।.!?]+', text)
    verses = [p.strip() for p in parts if 40 <= len(p.strip()) <= 600]

    if len(verses) >= 2:
        return verses

    # Fall back to paragraph chunks of ~300 chars
    verses = []
    words = text.split()
    chunk = []
    for word in words:
        chunk.append(word)
        joined = ' '.join(chunk)
        if len(joined) >= 200:
            if len(joined) <= 600:
                verses.append(joined)
            chunk = []
    if chunk:
        joined = ' '.join(chunk)
        if len(joined) >= 40:
            verses.append(joined)

    return verses


def load_verses_into_db(conn: sqlite3.Connection,
                        limit: int = None) -> int:
    """Extract all verses from w1_structure and index them."""
    src = sqlite3.connect(SOURCE_DB)
    rows = src.execute(
        "SELECT file_id, mandala, sukta, clean_text "
        "FROM w1_structure WHERE length(clean_text) > 100"
    ).fetchall()
    src.close()

    total = 0
    for file_id, mandala, sukta, text in rows:
        if limit and total >= limit:
            break
        verses = extract_verses(text or "")
        for idx, verse in enumerate(verses):
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO verses
                    (file_id, verse_idx, mandala, sukta, text, n_chars)
                    VALUES (?,?,?,?,?,?)
                """, (file_id, idx, mandala or "?",
                      sukta or "?", verse, len(verse)))
                total += 1
            except Exception:
                pass
        if total % 1000 == 0 and total > 0:
            conn.commit()
            print(f"  Indexed {total} verses...")

    conn.commit()
    return total


# ── Problems ──────────────────────────────────────────────────
def load_problems() -> pd.DataFrame:
    return pd.read_csv(PROBLEMS_CSV)


def make_problem_block(df: pd.DataFrame,
                       unsolved_only: bool = False) -> str:
    """Compact problem list for prompt. ~2000 tokens."""
    if unsolved_only:
        df = df[df['Status'].str.contains(
            'Unsolved|Active research|Active debate', na=False)]
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"[{r['ID']}]{r['Key_Problem_or_Concept']}:"
            f"{r['Description'][:80]}({r['Status']})"
        )
    return "\n".join(lines)


# ── API ───────────────────────────────────────────────────────
def call_gemini(prompt: str, system: str) -> tuple[str, int, int]:
    client = genai.Client(api_key=GEMINI_KEY)
    cfg    = genai_types.GenerateContentConfig(
        system_instruction=system, temperature=0.1)
    for attempt in range(3):
        try:
            r   = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=cfg)
            txt = r.text.strip()
            u   = r.usage_metadata
            return (txt,
                    getattr(u, 'prompt_token_count',    len(prompt)//4),
                    getattr(u, 'candidates_token_count', len(txt)//4))
        except Exception as e:
            if attempt == 2: return f"ERROR:{e}", 0, 0
            time.sleep(2**attempt)
    return "ERROR:retries", 0, 0


def call_deepseek(prompt: str, system: str) -> tuple[str, int, int]:
    if not FIREWORKS_KEY: return "SKIP", 0, 0
    client = OpenAI(api_key=FIREWORKS_KEY,
                    base_url="https://api.fireworks.ai/inference/v1")
    msgs = [{"role":"system","content":system},
            {"role":"user",  "content":prompt}]
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=DEEPSEEK_MODEL, messages=msgs,
                temperature=0.1, max_tokens=2000)
            txt = r.choices[0].message.content.strip()
            return txt, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            if attempt == 2: return f"ERROR:{e}", 0, 0
            time.sleep(2**attempt)
    return "ERROR:retries", 0, 0


def safe_json_list(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip()=="```" else lines[1:])
    try:
        r = json.loads(text)
        return r if isinstance(r, list) else []
    except Exception:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group())
                return r if isinstance(r, list) else []
            except Exception:
                pass
    return []


def track(conn, verse_id, pass_n, worker, inp, out, is_fw=False) -> float:
    global _spent
    ic = FW_IN  if is_fw else GEMINI_IN
    oc = FW_OUT if is_fw else GEMINI_OUT
    c  = inp*ic + out*oc
    budget = BUDGET_PASS2 if pass_n == 2 else BUDGET_PASS1
    with _lock:
        _spent += c
        if _spent > budget * 0.98:
            raise RuntimeError(f"BUDGET: ${_spent:.4f}")
    conn.execute(
        "INSERT INTO cost (verse_id,pass_n,worker,input_tokens,"
        "output_tokens,cost_usd) VALUES (?,?,?,?,?,?)",
        (verse_id, pass_n, worker, inp, out, c))
    conn.commit()
    return c


# ── The 4 worker prompts ──────────────────────────────────────
SYSTEM = ("You are a mathematical physicist. Extract mathematical structure "
          "only. No religion. No metaphor. Pure equations. "
          "Respond ONLY in valid JSON array. No markdown.")

def make_prompt(verse: str, problem_block: str, lens: str) -> str:
    lens_instructions = {
        "conservation": (
            "Find passages where a QUANTITY IS PRESERVED while "
            "other things change. Conservation laws. "
            "Write as: dX/dt=0 or X_before=X_after"
        ),
        "symmetry": (
            "Find passages where something IS UNCHANGED after a "
            "TRANSFORMATION. Symmetry groups, invariants. "
            "Write as: f(T(x))=f(x)"
        ),
        "scaling": (
            "Find passages where one quantity SCALES with another "
            "as a POWER LAW. Exponents, critical behavior. "
            "Write as: Y ∝ X^n, find n"
        ),
        "dynamics": (
            "Find passages describing RATES OF CHANGE, feedback, "
            "attractors, differential equations. "
            "Write as: dX/dt = f(X)"
        ),
    }

    return f"""VERSE:
{verse}

YOUR LENS: {lens_instructions[lens]}

Find ALL problems from the list below where this verse contains
mathematical structure relevant to that problem.

RULES:
- Only include if relevance >= 5
- You MUST write a mathematical equation using standard notation
- Maximum 5 matches
- Empty array [] if nothing qualifies

Respond ONLY with JSON array:
[{{"problem_id":39,"relevance":7,
  "equation":"dX/dt = -gamma*X*(X-Xc)^2",
  "reasoning":"one sentence, pure logic",
  "testable":"one specific computation to verify"}}]

PROBLEMS:
{problem_block}"""


# ── Pass 1: Gemini, all verses, 4 angles ─────────────────────
def process_verse_p1(verse_row: tuple,
                     problem_block: str,
                     problems_df: pd.DataFrame,
                     conn: sqlite3.Connection) -> int:
    vid, file_id, mandala, sukta, text = verse_row
    matches = 0
    worker_hits = {}   # {problem_id: [(worker, relevance, equation)]}

    for lens in ["conservation","symmetry","scaling","dynamics"]:
        prompt = make_prompt(text, problem_block, lens)
        txt, inp, out = call_gemini(prompt, SYSTEM)
        track(conn, vid, 1, f"gemini_{lens}", inp, out, False)

        items = safe_json_list(txt)
        for item in items:
            if not isinstance(item, dict): continue
            pid = item.get("problem_id")
            eq  = item.get("equation","")
            rel = int(item.get("relevance", 0))
            if not pid or not eq or rel < 5: continue

            prob = problems_df[problems_df['ID']==pid]
            if prob.empty: continue
            prob = prob.iloc[0]

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO p1_matches
                    (verse_id,worker,problem_id,problem_name,
                     problem_branch,problem_status,relevance,
                     equation,reasoning,testable)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (vid, f"gemini_{lens}", int(pid),
                      prob['Key_Problem_or_Concept'],
                      prob['Sub-Branch'],
                      prob['Status'],
                      rel, eq,
                      item.get("reasoning","")[:200],
                      item.get("testable","")[:200]))
                conn.commit()
                matches += 1

                if int(pid) not in worker_hits:
                    worker_hits[int(pid)] = []
                worker_hits[int(pid)].append(
                    (lens, rel, eq))
            except Exception:
                pass

    # Build cross-worker consensus for this verse
    for pid, hits in worker_hits.items():
        if len(hits) < 2: continue  # need 2+ workers to agree
        workers   = [h[0] for h in hits]
        relevances= [h[1] for h in hits]
        equations = [h[2] for h in hits]
        mean_rel  = sum(relevances) / len(relevances)

        prob = problems_df[problems_df['ID']==pid]
        if prob.empty: continue
        prob = prob.iloc[0]

        try:
            conn.execute("""
                INSERT OR REPLACE INTO p1_consensus
                (verse_id,problem_id,problem_name,problem_status,
                 workers_agreed,n_workers,mean_relevance,
                 equations,verse_text,mandala,sukta)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (vid, int(pid),
                  prob['Key_Problem_or_Concept'],
                  prob['Status'],
                  json.dumps(workers),
                  len(workers),
                  mean_rel,
                  json.dumps(equations),
                  text[:300],
                  mandala, sukta))
            conn.commit()
        except Exception:
            pass

    conn.execute(
        "INSERT OR REPLACE INTO p1_status (verse_id,done) VALUES (?,1)",
        (vid,))
    conn.commit()
    return matches


def run_pass1(limit: int = None,
              workers: int = 4,
              unsolved_only: bool = False):
    global _spent
    _spent = 0.0

    if not GEMINI_KEY:
        print("ERROR: --key required"); sys.exit(1)

    conn        = init_db()
    problems_df = load_problems()
    prob_block  = make_problem_block(problems_df, unsolved_only)

    # Index verses if not done
    n_indexed = conn.execute(
        "SELECT COUNT(*) FROM verses").fetchone()[0]
    if n_indexed == 0:
        print("Indexing verses from corpus...")
        n_indexed = load_verses_into_db(conn)
        print(f"✓ Indexed {n_indexed:,} verses")
    else:
        print(f"✓ Using {n_indexed:,} indexed verses")

    # Get pending
    done_ids = set(r[0] for r in conn.execute(
        "SELECT verse_id FROM p1_status WHERE done=1").fetchall())

    q = "SELECT id,file_id,mandala,sukta,text FROM verses"
    all_verses = conn.execute(q).fetchall()
    pending = [v for v in all_verses if v[0] not in done_ids]
    if limit:
        pending = pending[:limit]

    # Cost estimate
    tok_per_call  = (len(prob_block)//4 + 100)  # prompt tokens
    total_calls   = len(pending) * 4              # 4 lenses
    est_cost      = total_calls * tok_per_call * GEMINI_IN
    est_cost_out  = total_calls * 200 * GEMINI_OUT

    print(f"\nVerse-level Pass 1 (Gemini, 4 angles)")
    print(f"Verses pending:  {len(pending):,}")
    print(f"Total API calls: {total_calls:,}")
    print(f"Est. input cost: ${est_cost:.2f}")
    print(f"Est. output:     ${est_cost_out:.2f}")
    print(f"Est. TOTAL:      ${est_cost+est_cost_out:.2f}")
    print(f"Budget:          ${BUDGET_PASS1}")
    print(f"Workers:         {workers}")
    print()

    start   = time.time()
    done_n  = 0
    total_m = 0

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as ex:
        futures = {
            ex.submit(process_verse_p1, v, prob_block,
                      problems_df, conn): v
            for v in pending
        }
        for future in concurrent.futures.as_completed(futures):
            v = futures[future]
            try:
                n      = future.result()
                done_n += 1
                total_m += n
                elapsed = time.time() - start
                rate    = done_n / elapsed if elapsed > 0 else 1
                eta     = (len(pending)-done_n) / rate
                c_now   = conn.execute(
                    "SELECT SUM(cost_usd) FROM cost WHERE pass_n=1"
                ).fetchone()[0] or 0
                if done_n % 50 == 0 or n > 0:
                    print(f"[{done_n:>5}/{len(pending)}] "
                          f"matches={n:>2} total={total_m:>4} "
                          f"cost=${c_now:.3f} "
                          f"ETA={eta/60:.1f}m")
            except RuntimeError as e:
                print(f"\nBUDGET STOP: {e}")
                break
            except Exception as e:
                print(f"ERR: {e}")

    # Build consensus table
    build_consensus(conn, problems_df)

    conn.close()
    print(f"\n✓ Pass 1 complete. {total_m} matches. "
          f"Run --results or --pass2")


def build_consensus(conn, problems_df):
    """Compute cross-worker consensus after pass 1."""
    # Already built per-verse above.
    # Now aggregate: which problems appear most across verses?
    print("\nBuilding consensus...")
    n = conn.execute(
        "SELECT COUNT(*) FROM p1_consensus").fetchone()[0]
    print(f"  Cross-worker consensus entries: {n}")

    top = conn.execute("""
        SELECT problem_id, COUNT(*) as n_verses,
               AVG(mean_relevance) as avg_rel,
               MAX(n_workers) as max_workers
        FROM p1_consensus
        GROUP BY problem_id
        ORDER BY n_verses DESC, avg_rel DESC
        LIMIT 20
    """).fetchall()

    print(f"\n  Top problems by verse coverage:")
    for pid, nv, ar, mw in top:
        prob = problems_df[problems_df['ID']==pid]
        pname = prob.iloc[0]['Key_Problem_or_Concept'] if not prob.empty else "?"
        pstat = prob.iloc[0]['Status'] if not prob.empty else "?"
        marker = " ★UNSOLVED" if "Unsolved" in str(pstat) else ""
        print(f"    #{pid} {pname[:40]:<40} "
              f"verses={nv} avg_rel={ar:.1f}{marker}")


# ── Pass 2: DeepSeek on top candidates ───────────────────────
def run_pass2(top_n: int = 500, workers: int = 4):
    global _spent
    _spent = 0.0

    if not FIREWORKS_KEY:
        print("ERROR: --fw-key required for pass 2")
        sys.exit(1)

    conn        = init_db()
    problems_df = load_problems()
    prob_block  = make_problem_block(problems_df)

    # Get top N candidates from pass 1 consensus
    candidates = conn.execute("""
        SELECT DISTINCT c.verse_id, c.problem_id,
               c.n_workers, c.mean_relevance,
               v.text, v.mandala, v.sukta
        FROM p1_consensus c
        JOIN verses v ON v.id = c.verse_id
        ORDER BY c.n_workers DESC, c.mean_relevance DESC
        LIMIT ?
    """, (top_n,)).fetchall()

    done_p2 = set(
        (r[0],r[1]) for r in conn.execute(
            "SELECT verse_id,problem_id FROM p2_status "
            "WHERE done=1"
        ).fetchall()
    )
    pending = [(vid,pid,nw,mr,txt,man,suk)
               for vid,pid,nw,mr,txt,man,suk in candidates
               if (vid,pid) not in done_p2]

    print(f"\nPass 2 (DeepSeek cross-validation)")
    print(f"Candidates:    {len(candidates)}")
    print(f"Already done:  {len(done_p2)}")
    print(f"Pending:       {len(pending)}")
    est = len(pending) * 4 * 2100 * FW_IN
    print(f"Est. cost:     ${est:.2f}")

    start  = time.time()
    done_n = 0
    gold_n = 0

    def process_candidate(row):
        vid, pid, nw, mr, verse_text, mandala, sukta = row
        hits = []

        for lens in ["conservation","symmetry","scaling","dynamics"]:
            prompt   = make_prompt(verse_text, prob_block, lens)
            txt, inp, out = call_deepseek(prompt, SYSTEM)
            track(conn, vid, 2, f"deep_{lens}", inp, out, True)

            items = safe_json_list(txt)
            for item in items:
                if not isinstance(item,dict): continue
                if item.get("problem_id") != pid: continue
                rel = int(item.get("relevance",0))
                eq  = item.get("equation","")
                if rel < 5 or not eq: continue
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO p2_matches
                        (verse_id,worker,problem_id,relevance,
                         equation,reasoning,testable)
                        VALUES (?,?,?,?,?,?,?)
                    """, (vid, f"deep_{lens}", int(pid), rel, eq,
                          item.get("reasoning","")[:200],
                          item.get("testable","")[:200]))
                    conn.commit()
                    hits.append((lens, rel, eq))
                except Exception:
                    pass

        conn.execute(
            "INSERT OR REPLACE INTO p2_status "
            "(verse_id,done) VALUES (?,1)", (vid,))
        conn.commit()
        return hits

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as ex:
        futures = {ex.submit(process_candidate, row): row
                   for row in pending}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            vid, pid, nw, mr, verse_text, mandala, sukta = row
            try:
                hits   = future.result()
                done_n += 1

                if hits:
                    # Write gold entry
                    g_rows = conn.execute("""
                        SELECT equation, mean_relevance, n_workers
                        FROM p1_consensus
                        WHERE verse_id=? AND problem_id=?
                    """, (vid, pid)).fetchone()

                    prob = problems_df[problems_df['ID']==pid]
                    if not prob.empty and g_rows:
                        p = prob.iloc[0]
                        d_rel = max(h[1] for h in hits)
                        d_eq  = hits[0][2]
                        g_eq  = json.loads(g_rows[0])[0] \
                                if g_rows[0] else ""
                        score = (float(g_rows[1]) + d_rel) / 2 \
                                * int(g_rows[2])

                        try:
                            conn.execute("""
                                INSERT OR REPLACE INTO gold
                                (verse_id,problem_id,problem_name,
                                 problem_branch,problem_status,
                                 equation_gemini,equation_deep,
                                 n_gemini_workers,relevance_gemini,
                                 relevance_deep,gold_score,
                                 verse_text,mandala,sukta)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """, (vid, int(pid),
                                  p['Key_Problem_or_Concept'],
                                  p['Sub-Branch'],
                                  p['Status'],
                                  g_eq, d_eq,
                                  int(g_rows[2]),
                                  float(g_rows[1]),
                                  d_rel, score,
                                  verse_text[:300],
                                  mandala, sukta))
                            conn.commit()
                            gold_n += 1
                        except Exception:
                            pass

                c_now = conn.execute(
                    "SELECT SUM(cost_usd) FROM cost "
                    "WHERE pass_n=2").fetchone()[0] or 0
                print(f"[{done_n:>4}/{len(pending)}] "
                      f"gold={gold_n} cost=${c_now:.3f}")

            except RuntimeError as e:
                print(f"\nBUDGET STOP: {e}"); break
            except Exception as e:
                print(f"ERR: {e}")

    conn.close()
    print(f"\n✓ Pass 2 complete. {gold_n} gold entries.")
    print("Run --results to see findings.")


# ── Results ───────────────────────────────────────────────────
def show_results(top_n: int = 30, unsolved_only: bool = False):
    conn        = sqlite3.connect(VERSE_DB)
    problems_df = load_problems()

    print(f"\n{'═'*90}")
    print("  VERSE WALKER RESULTS")
    print(f"{'═'*90}")

    # Gold table first
    q = """
        SELECT problem_id, problem_name, problem_branch,
               problem_status, equation_gemini, equation_deep,
               n_gemini_workers, gold_score,
               verse_text, mandala, sukta
        FROM gold
    """
    if unsolved_only:
        q += " WHERE problem_status LIKE '%Unsolved%'"
    q += " ORDER BY gold_score DESC LIMIT ?"

    gold = conn.execute(q, (top_n,)).fetchall()

    if gold:
        print(f"\n  GOLD TABLE — Gemini + DeepSeek consensus")
        print(f"  {'─'*85}")
        for i, row in enumerate(gold, 1):
            pid,pname,pbranch,pstat = row[0],row[1],row[2],row[3]
            eq_g,eq_d = row[4],row[5]
            nw,score  = row[6],row[7]
            verse,man,suk = row[8],row[9],row[10]
            print(f"\n  [{i}] #{pid} {pname}")
            print(f"       {pbranch} | {pstat}")
            print(f"       Gold score: {score:.1f} | "
                  f"Gemini workers agreed: {nw}")
            print(f"       Source: {man} / {suk}")
            print(f"       Gemini eq:   {eq_g}")
            print(f"       DeepSeek eq: {eq_d}")
            print(f"       Verse: {(verse or '')[:150]}")
    else:
        print("\n  Gold table empty (run --pass2 first)")

    # Pass 1 consensus
    print(f"\n\n  PASS 1 CONSENSUS — 2+ Gemini workers agreed")
    print(f"  {'─'*85}")

    q2 = """
        SELECT c.problem_id, c.problem_name, c.problem_status,
               c.n_workers, c.mean_relevance,
               c.workers_agreed, c.equations,
               c.verse_text, c.mandala, c.sukta,
               COUNT(*) OVER (PARTITION BY c.problem_id) as verse_count
        FROM p1_consensus c
        WHERE c.n_workers >= 2
    """
    if unsolved_only:
        q2 += " AND c.problem_status LIKE '%Unsolved%'"
    q2 += " ORDER BY c.n_workers DESC, c.mean_relevance DESC LIMIT ?"

    p1c = conn.execute(q2, (top_n,)).fetchall()

    if p1c:
        for i, row in enumerate(p1c[:20], 1):
            pid,pname,pstat = row[0],row[1],row[2]
            nw,mr = row[3],row[4]
            workers = json.loads(row[5] or "[]")
            eqs     = json.loads(row[6] or "[]")
            verse,man,suk = row[7],row[8],row[9]
            vcount  = row[10]
            marker  = " ★" if "Unsolved" in str(pstat) else ""
            print(f"\n  [{i}] #{pid} {pname}{marker}")
            print(f"       Status: {pstat}")
            print(f"       Workers agreed: {workers}  "
                  f"Mean relevance: {mr:.1f}  "
                  f"Total verses: {vcount}")
            print(f"       Source: {man} / {suk}")
            for lens, eq in zip(workers, eqs[:3]):
                print(f"       [{lens}] {eq}")
            print(f"       Verse: {(verse or '')[:120]}")
    else:
        print("  No consensus yet. Run --pass1 first.")

    # Summary stats
    n_verses = conn.execute(
        "SELECT COUNT(*) FROM verses").fetchone()[0]
    n_p1     = conn.execute(
        "SELECT COUNT(*) FROM p1_matches").fetchone()[0]
    n_cons   = conn.execute(
        "SELECT COUNT(*) FROM p1_consensus").fetchone()[0]
    n_gold   = conn.execute(
        "SELECT COUNT(*) FROM gold").fetchone()[0]
    cost     = conn.execute(
        "SELECT SUM(cost_usd) FROM cost").fetchone()[0] or 0

    print(f"\n{'═'*90}")
    print(f"  Verses indexed:      {n_verses:,}")
    print(f"  Pass 1 matches:      {n_p1:,}")
    print(f"  Consensus entries:   {n_cons:,}")
    print(f"  Gold entries:        {n_gold}")
    print(f"  Total cost:          ${cost:.4f} (₹{cost*83.5:.0f})")
    print(f"{'═'*90}\n")
    conn.close()


def show_problem(problem_id: int):
    conn        = sqlite3.connect(VERSE_DB)
    problems_df = load_problems()
    prob        = problems_df[problems_df['ID']==problem_id]

    if prob.empty:
        print(f"Problem {problem_id} not found")
        return
    p = prob.iloc[0]

    print(f"\n{'═'*80}")
    print(f"  #{problem_id}: {p['Key_Problem_or_Concept']}")
    print(f"  {p['Description']}")
    print(f"  Status: {p['Status']}")
    print(f"{'═'*80}")

    rows = conn.execute("""
        SELECT m.verse_id, m.worker, m.relevance,
               m.equation, m.reasoning, m.testable,
               v.text, v.mandala, v.sukta
        FROM p1_matches m
        JOIN verses v ON v.id = m.verse_id
        WHERE m.problem_id = ?
        ORDER BY m.relevance DESC
        LIMIT 20
    """, (problem_id,)).fetchall()

    for vid,worker,rel,eq,reason,testable,vtext,man,suk in rows:
        print(f"\n  Worker: {worker}  Relevance: {rel}")
        print(f"  Source: {man} / {suk}")
        print(f"  Equation:  {eq}")
        print(f"  Reasoning: {reason}")
        print(f"  Testable:  {testable}")
        print(f"  Verse:     {(vtext or '')[:150]}")

    conn.close()


# ── CLI ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="VedaStream Verse Walker — "
                    "verse-level corpus × 505 open problems")
    ap.add_argument("--pass1",        action="store_true",
                    help="Run pass 1: Gemini, 4 angles, all verses")
    ap.add_argument("--pass2",        action="store_true",
                    help="Run pass 2: DeepSeek on top candidates")
    ap.add_argument("--results",      action="store_true")
    ap.add_argument("--problem",      type=int,
                    help="Show all matches for one problem ID")
    ap.add_argument("--limit",        type=int, default=None,
                    help="Limit verses for testing")
    ap.add_argument("--top",          type=int, default=500,
                    help="Top N candidates for pass 2")
    ap.add_argument("--workers",      type=int, default=4)
    ap.add_argument("--unsolved-only",action="store_true")
    ap.add_argument("--key",          type=str)
    ap.add_argument("--fw-key",       type=str)

    args = ap.parse_args()
    global GEMINI_KEY, FIREWORKS_KEY
    if args.key:    GEMINI_KEY    = args.key
    if args.fw_key: FIREWORKS_KEY = args.fw_key

    if   args.pass1:   run_pass1(args.limit, args.workers,
                                 args.unsolved_only)
    elif args.pass2:   run_pass2(args.top, args.workers)
    elif args.results: show_results(30, args.unsolved_only)
    elif args.problem: show_problem(args.problem)
    else:
        ap.print_help()
        print("""
QUICK START:
  # Index + test 200 verses (costs $0.17):
  python veda_verse_walker.py --pass1 --limit 200 --key YOUR_KEY

  # Full pass 1 — all 28,149 verses (costs ~$24):
  python veda_verse_walker.py --pass1 --key YOUR_KEY --workers 4

  # Pass 2 — DeepSeek on top 500 candidates (costs ~$1.80):
  python veda_verse_walker.py --pass2 --fw-key YOUR_FW_KEY

  # Results:
  python veda_verse_walker.py --results
  python veda_verse_walker.py --results --unsolved-only

  # Drill into Navier-Stokes specifically:
  python veda_verse_walker.py --problem 39

COST SUMMARY:
  Pass 1 test  (200 verses):  $0.17
  Pass 1 full  (28,149 verses): ~$24.00
  Pass 2 top 500:               ~$1.80
  TOTAL:                        ~$25.80

WHAT MAKES THIS DIFFERENT:
  - Verse level, not file blobs
  - 4 independent mathematical lenses per verse
  - Cross-worker consensus: 2+ lenses agree = structural depth
  - Cross-provider gold: Gemini + DeepSeek agree = publishable
""")


if __name__ == "__main__":
    main()
