"""
veda_physics_4lens.py  — v3 FINAL
==================================
One call per verse. Four lenses in one prompt.
Groq GPT-OSS-120B for conservation + symmetry + scaling.
Ollama Qwen2.5 for dynamics (local, free).
Gemini NOT used here — it is already running your biology walker.

COST (47,899 verses):
    Groq  combined: 47,899 × ~1,200 cached tokens × $0.15/1M = $8.62
    Ollama dynamics: $0.00
    TOTAL: ~$9.00

SETUP:
    pip install groq pandas
    ollama pull qwen2.5:32b-q4_K_M   (20GB, best math)
    ollama pull qwen2.5:14b-q8_0     (14GB, faster)

USAGE:
    # Test 50 verses ($0.04):
    python veda_physics_4lens.py --walk --limit 50 --groq-key YOUR_KEY

    # Full run 8 workers ($9, ~3 hours):
    python veda_physics_4lens.py --walk --workers 8 --groq-key YOUR_KEY

    # Results:
    python veda_physics_4lens.py --results --unsolved-only --min-sources 3

    # Drill into problems:
    python veda_physics_4lens.py --problem 39   # Navier-Stokes
    python veda_physics_4lens.py --problem 40   # Yang-Mills
    python veda_physics_4lens.py --problem 1    # Riemann Hypothesis
"""

import os, sys, json, time, re, sqlite3
import threading, concurrent.futures, argparse
import urllib.request

import pandas as pd

try:
    from groq import Groq as GroqClient
    GROQ_AVAIL = True
except ImportError:
    print("FATAL: pip install groq"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────
GROQ_KEY      = os.getenv("GROQ_API_KEY",   "")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",    "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",   "qwen2.5:32b-q4_K_M")
GROQ_MODEL    = "openai/gpt-oss-120b"
GROQ_REASONING= "medium"

SOURCE_DB     = "vedastream.db"
OUTPUT_DB     = "veda_physics_4lens.db"
PROBLEMS_CSV  = "pure_sciences_500.csv"

MIN_VERSE     = 40
MAX_VERSE     = 500
MIN_RELEVANCE = 5
BUDGET_USD    = 15.00

GROQ_IN  = 0.15 / 1_000_000
GROQ_OUT = 0.75 / 1_000_000

_lock  = threading.Lock()
_spent = 0.0

GENERIC_PATTERNS = [
    r'^dX/dt\s*=\s*f\(X\)$',
    r'^Y\s*\\?propto\s*X\^n$',
    r'^f\(T\(x\)\)\s*=\s*f\(x\)$',
    r'^dX/dt\s*=\s*0$',
    r'^X_before\s*=\s*X_after$',
    r'^Y\s*=\s*a\s*\*\s*X\^n$',
]

def is_generic(eq: str) -> bool:
    if not eq or len(eq.strip()) < 8:
        return True
    for pat in GENERIC_PATTERNS:
        if re.match(pat, eq.strip(), re.IGNORECASE):
            return True
    return False


# ── Database ──────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(OUTPUT_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS verses (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id   TEXT NOT NULL,
            verse_idx INTEGER NOT NULL,
            mandala   TEXT,
            sukta     TEXT,
            text      TEXT NOT NULL,
            n_chars   INTEGER,
            UNIQUE(file_id, verse_idx)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id       INTEGER NOT NULL,
            lens           TEXT NOT NULL,
            provider       TEXT NOT NULL,
            problem_id     INTEGER NOT NULL,
            problem_name   TEXT,
            problem_branch TEXT,
            problem_status TEXT,
            relevance      INTEGER DEFAULT 0,
            equation       TEXT,
            reasoning      TEXT,
            testable       TEXT,
            is_generic     INTEGER DEFAULT 0,
            cost_usd       REAL DEFAULT 0,
            created_at     REAL DEFAULT (unixepoch()),
            UNIQUE(verse_id, lens, problem_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id     INTEGER NOT NULL,
            problem_name   TEXT,
            problem_branch TEXT,
            problem_status TEXT,
            equation       TEXT,
            lens           TEXT,
            n_sources      INTEGER DEFAULT 0,
            n_verses       INTEGER DEFAULT 0,
            avg_relevance  REAL DEFAULT 0,
            sources_list   TEXT,
            example_verse  TEXT,
            created_at     REAL DEFAULT (unixepoch()),
            UNIQUE(problem_id, equation)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS proc_status (
            verse_id  INTEGER PRIMARY KEY,
            done      INTEGER DEFAULT 0,
            n_matches INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_ledger (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id      INTEGER,
            provider      TEXT,
            input_tokens  INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd      REAL DEFAULT 0,
            created_at    REAL DEFAULT (unixepoch())
        )
    """)

    conn.commit()
    return conn


# ── Verse extraction ──────────────────────────────────────────
def extract_verses(text: str) -> list:
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


SKIP_PATTERNS = ['http', 'gretil', 'www.', 'Input by',
                 'Based on', 'edition by', 'GRETIL', '.pdf']

def index_verses(conn: sqlite3.Connection) -> int:
    already = conn.execute(
        "SELECT COUNT(*) FROM verses").fetchone()[0]
    if already > 0:
        print(f"  Using {already:,} indexed verses")
        return already

    src = sqlite3.connect(SOURCE_DB)
    rows = src.execute(
        "SELECT file_id, mandala, sukta, clean_text "
        "FROM w1_structure WHERE length(clean_text) > 100"
    ).fetchall()
    src.close()

    total = 0
    for fid, mandala, sukta, text in rows:
        for idx, verse in enumerate(extract_verses(text or "")):
            if any(p in verse for p in SKIP_PATTERNS):
                continue
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO verses
                    (file_id,verse_idx,mandala,sukta,text,n_chars)
                    VALUES (?,?,?,?,?,?)
                """, (fid, idx, mandala or "?",
                      sukta or "?", verse, len(verse)))
                total += 1
            except Exception:
                pass
        if total % 5000 == 0 and total > 0:
            conn.commit()
            print(f"  Indexed {total:,} verses...")

    conn.commit()
    print(f"  Total: {total:,} verses (metadata excluded)")
    return total


# ── Problems ──────────────────────────────────────────────────
def load_problems() -> pd.DataFrame:
    if not os.path.exists(PROBLEMS_CSV):
        print(f"FATAL: {PROBLEMS_CSV} not found"); sys.exit(1)
    return pd.read_csv(PROBLEMS_CSV)


def make_problem_block(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.iterrows():
        lines.append(
            f"[{int(r['ID'])}]{r['Key_Problem_or_Concept']}:"
            f"{str(r['Description'])[:80]}")
    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────
SYSTEM = (
    "You are a mathematical physicist. "
    "Output ONLY valid JSON array. No markdown. "
    "Empty array [] if nothing qualifies."
)

def make_prompt(verse: str, prob_block: str) -> str:
    return f"""VERSE:
{verse}

Analyze through 4 lenses. Find matching open problems.

LENS 1 CONSERVATION — quantity preserved while others change.
  Write as: d[X]/dt + nabla·([X]v) = 0  or  sum([X])=const
  Example: d(rho)/dt + nabla·(rho*u) = 0

LENS 2 SYMMETRY — invariant under named transformation.
  Write as: f(T(x))=f(x) where T is NAMED explicitly.
  Example: u(x,t)=u(x+L,t) [translation by L]

LENS 3 SCALING — power law with SPECIFIC numeric exponent.
  Write as: Y proportional to X^[number]. Number must be explicit.
  Example: E(k) proportional to k^(-5/3) [Kolmogorov]
  REJECT generic Y proportional to X^n without a value.

LENS 4 DYNAMICS — rate equation with specific terms named.
  Write as: d[X]/dt = [driving_term] - [damping_term]
  Example: du/dt = -(u·nabla)u - nabla(p)/rho + nu*nabla^2(u)
  REJECT generic dX/dt=f(X).

RULES:
- Relevance >= 5 only
- Equation must be specific — no placeholders
- Max 6 matches total
- Return [] if nothing qualifies

JSON only:
[{{"lens":"conservation","problem_id":39,"relevance":7,
  "equation":"d(rho)/dt + nabla·(rho*u)=0",
  "reasoning":"verse describes density conservation in flow",
  "testable":"verify continuity equation for described process"}}]

PROBLEMS:
{prob_block}"""


# ── API ───────────────────────────────────────────────────────
def call_groq(prompt: str) -> tuple:
    if not GROQ_KEY:
        return "SKIP", 0, 0
    client = GroqClient(api_key=GROQ_KEY)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role":"system","content":SYSTEM},
                    {"role":"user",  "content":prompt},
                ],
                temperature=0.1,
                max_completion_tokens=2048,
                reasoning_effort=GROQ_REASONING,
            )
            txt = r.choices[0].message.content.strip()
            return txt, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(15); continue
            if attempt == 2:
                return f"ERR:{e}", 0, 0
            time.sleep(2**attempt)
    return "ERR:retries", 0, 0


def call_ollama(prompt: str) -> tuple:
    try:
        data = json.dumps({
            "model":  OLLAMA_MODEL,
            "prompt": SYSTEM + "\n\n" + prompt,
            "stream": False,
            "options":{"temperature":0.1,"num_predict":2048}
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate", data=data,
            headers={"Content-Type":"application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        txt = result.get("response","").strip()
        return txt, len(prompt)//4, len(txt)//4
    except Exception as e:
        return f"ERR:{e}", 0, 0


def safe_json(text: str) -> list:
    if not text or text.startswith("ERR") or text == "SKIP":
        return []
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1] if lines[-1].strip()=="```" else lines[1:])
    try:
        r = json.loads(text)
        if isinstance(r, list): return r
        if isinstance(r, dict):
            for v in r.values():
                if isinstance(v, list): return v
        return []
    except Exception:
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group())
                return r if isinstance(r, list) else []
            except Exception: pass
    return []


def track_cost(conn, verse_id, provider, inp, out) -> float:
    global _spent
    cost = inp * GROQ_IN + out * GROQ_OUT \
           if provider == "groq" else 0.0
    with _lock:
        _spent += cost
        if _spent > BUDGET_USD * 0.98:
            raise RuntimeError(
                f"BUDGET STOP: ${_spent:.4f} of ${BUDGET_USD}")
    conn.execute("""
        INSERT INTO cost_ledger
        (verse_id,provider,input_tokens,output_tokens,cost_usd)
        VALUES (?,?,?,?,?)
    """, (verse_id, provider, inp, out, cost))
    conn.commit()
    return cost


def save_items(conn, verse_id, items, problems_df,
               provider, total_cost):
    saved = 0
    cpi   = total_cost / max(len(items), 1)
    for item in items:
        if not isinstance(item, dict): continue
        pid  = item.get("problem_id")
        eq   = item.get("equation","").strip()
        rel  = int(item.get("relevance", 0))
        lens = item.get("lens","combined")
        if not pid or not eq or rel < MIN_RELEVANCE: continue
        prob = problems_df[problems_df['ID']==pid]
        if prob.empty: continue
        p = prob.iloc[0]
        try:
            conn.execute("""
                INSERT OR REPLACE INTO matches
                (verse_id,lens,provider,problem_id,
                 problem_name,problem_branch,problem_status,
                 relevance,equation,reasoning,testable,
                 is_generic,cost_usd)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (verse_id, lens, provider, int(pid),
                  p['Key_Problem_or_Concept'],
                  p['Sub-Branch'], p['Status'],
                  rel, eq,
                  str(item.get("reasoning",""))[:250],
                  str(item.get("testable",""))[:250],
                  1 if is_generic(eq) else 0,
                  cpi))
            conn.commit()
            saved += 1
        except Exception: pass
    return saved


# ── Per-verse processor ───────────────────────────────────────
def process_verse(row, prob_block, problems_df,
                  conn, ollama_ok) -> int:
    vid, mandala, sukta, verse_text = row
    prompt  = make_prompt(verse_text, prob_block)
    matches = 0

    # Groq: all 4 lenses in one call
    txt, inp, out = call_groq(prompt)
    cost = track_cost(conn, vid, "groq", inp, out)
    matches += save_items(
        conn, vid, safe_json(txt), problems_df, "groq", cost)

    # Ollama: dynamics lens, free
    if ollama_ok:
        dyn_prompt = (
            "LENS 4 DYNAMICS ONLY. Find differential equations.\n"
            "d[X]/dt=[driving]-[damping], specific terms.\n"
            "Set lens='dynamics' in output.\n\n" + prompt
        )
        txt2, i2, o2 = call_ollama(dyn_prompt)
        track_cost(conn, vid, "ollama", i2, o2)
        items2 = safe_json(txt2)
        for item in items2:
            if isinstance(item, dict):
                item['lens'] = 'dynamics'
        matches += save_items(
            conn, vid, items2, problems_df, "ollama", 0.0)

    conn.execute("""
        INSERT OR REPLACE INTO proc_status
        (verse_id,done,n_matches) VALUES (?,1,?)
    """, (vid, matches))
    conn.commit()
    return matches


# ── Gold builder ──────────────────────────────────────────────
def build_gold(conn, problems_df):
    print("\nBuilding gold table...")
    conn.execute("DELETE FROM gold")
    conn.commit()

    rows = conn.execute("""
        SELECT m.problem_id, m.problem_name,
               m.problem_branch, m.problem_status,
               m.equation, m.lens,
               COUNT(DISTINCT v.mandala) as n_src,
               COUNT(DISTINCT m.verse_id) as n_v,
               AVG(m.relevance) as avg_r,
               GROUP_CONCAT(DISTINCT v.mandala) as srcs,
               MIN(v.text) as ex_verse
        FROM matches m
        JOIN verses v ON v.id = m.verse_id
        WHERE m.is_generic = 0
          AND m.relevance >= 6
          AND v.n_chars >= 60
        GROUP BY m.problem_id, m.equation
        HAVING n_src >= 2
        ORDER BY n_src DESC, avg_r DESC
    """).fetchall()

    n = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO gold
                (problem_id,problem_name,problem_branch,
                 problem_status,equation,lens,n_sources,
                 n_verses,avg_relevance,sources_list,example_verse)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, row)
            n += 1
        except Exception: pass
    conn.commit()
    print(f"  Gold entries: {n}")
    return n


# ── Walk ──────────────────────────────────────────────────────
def run_walk(limit=None, workers=4, use_ollama=True):
    global _spent
    _spent = 0.0

    if not GROQ_KEY:
        print("ERROR: --groq-key required"); sys.exit(1)

    conn        = init_db()
    problems_df = load_problems()
    prob_block  = make_problem_block(problems_df)

    n_verses = index_verses(conn)

    done_ids = set(r[0] for r in conn.execute(
        "SELECT verse_id FROM proc_status WHERE done=1").fetchall())
    all_rows = conn.execute(
        "SELECT id,mandala,sukta,text FROM verses").fetchall()
    pending  = [(r[0],r[1],r[2],r[3])
                for r in all_rows if r[0] not in done_ids]
    if limit:
        pending = pending[:limit]

    ollama_ok = False
    if use_ollama:
        try:
            urllib.request.urlopen(
                f"{OLLAMA_HOST}/api/tags", timeout=3)
            ollama_ok = True
        except Exception:
            print("  ⚠ Ollama not running — dynamics lens skipped")

    # Cost estimate with caching
    prob_tok = len(prob_block) // 4
    eff_inp  = int(prob_tok * 0.5) + 120   # 50% cached + verse
    est      = len(pending) * (eff_inp * GROQ_IN + 400 * GROQ_OUT)

    print(f"\n{'═'*60}")
    print(f"  VEDA PHYSICS 4-LENS  v3")
    print(f"{'═'*60}")
    print(f"  Verses pending:  {len(pending):,}")
    print(f"  Workers:         {workers}")
    print(f"  Groq:            {GROQ_MODEL} ({GROQ_REASONING})")
    print(f"  Ollama:          {'YES '+OLLAMA_MODEL if ollama_ok else 'NO'}")
    print(f"  Calls/verse:     {'2' if ollama_ok else '1'}")
    print(f"  Est. Groq cost:  ${est:.2f} (with caching)")
    print(f"  Budget:          ${BUDGET_USD}")
    print(f"{'═'*60}\n")

    start    = time.time()
    done_cnt = 0
    total_m  = 0

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as ex:
        futures = {
            ex.submit(process_verse, row, prob_block,
                      problems_df, conn, ollama_ok): row
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                n        = future.result()
                done_cnt += 1
                total_m  += n
                elapsed  = time.time() - start
                rate     = done_cnt / elapsed if elapsed > 0 else 1
                eta      = (len(pending) - done_cnt) / rate
                cost_r   = conn.execute(
                    "SELECT SUM(cost_usd) FROM cost_ledger"
                ).fetchone()[0] or 0
                if done_cnt % 100 == 0 or n >= 3:
                    print(f"  [{done_cnt:>6}/{len(pending)}] "
                          f"m={n:>2} total={total_m:>6} "
                          f"cost=${cost_r:.3f} "
                          f"ETA={eta/60:.0f}m")
            except RuntimeError as e:
                print(f"\n  STOPPED: {e}"); break
            except Exception as e:
                print(f"  ERR: {e}")

    build_gold(conn, problems_df)
    cost_f = conn.execute(
        "SELECT SUM(cost_usd) FROM cost_ledger"
    ).fetchone()[0] or 0
    print(f"\n  Done. {done_cnt:,} verses. "
          f"${cost_f:.4f} (₹{cost_f*83.5:.0f})")
    conn.close()


# ── Results ───────────────────────────────────────────────────
def show_results(top_n=30, min_sources=2, unsolved_only=False):
    if not os.path.exists(OUTPUT_DB):
        print("Run --walk first."); return
    conn = sqlite3.connect(OUTPUT_DB)

    print(f"\n{'═'*80}")
    print(f"  PHYSICS 4-LENS RESULTS")
    print(f"{'═'*80}")

    q = """
        SELECT problem_id,problem_name,problem_branch,
               problem_status,equation,lens,
               n_sources,n_verses,avg_relevance,
               sources_list,example_verse
        FROM gold WHERE n_sources >= ?
    """
    params = [min_sources]
    if unsolved_only:
        q += " AND problem_status LIKE '%Unsolved%'"
    q += " ORDER BY n_sources DESC, avg_relevance DESC LIMIT ?"
    params.append(top_n)

    gold = conn.execute(q, params).fetchall()

    if gold:
        print(f"\n  GOLD — {len(gold)} entries, "
              f"{min_sources}+ independent sources\n")
        for i, row in enumerate(gold, 1):
            (pid,pname,pbranch,pstat,eq,lens,
             nsrc,nv,ar,srcs,verse) = row
            mark = " ★ UNSOLVED" if "Unsolved" in str(pstat) else ""
            print(f"  [{i:>3}] #{pid} {pname}{mark}")
            print(f"         {pbranch} | {pstat}")
            print(f"         [{lens}] {eq}")
            print(f"         Sources:{nsrc} Verses:{nv} "
                  f"AvgRel:{ar:.1f}")
            print(f"         {(srcs or '')[:90]}")
            print(f"         {(verse or '')[:110]}")
            print()
    else:
        print(f"\n  No gold entries with {min_sources}+ sources.")
        print("  Try --min-sources 1")

    freq = conn.execute("""
        SELECT m.problem_id,m.problem_name,m.problem_status,
               COUNT(DISTINCT v.mandala)  as nsrc,
               COUNT(DISTINCT m.verse_id) as nv,
               SUM(CASE WHEN m.is_generic=0 THEN 1 ELSE 0 END) as nspec
        FROM matches m
        JOIN verses v ON v.id=m.verse_id
        GROUP BY m.problem_id
        ORDER BY nspec DESC, nsrc DESC LIMIT 30
    """).fetchall()

    print(f"\n  TOP PROBLEMS")
    print(f"  {'ID':>4}  {'Problem':<42} "
          f"{'Src':>4} {'V':>5} {'Spec':>5}  Status")
    for pid,pname,pstat,nsrc,nv,nspec in freq:
        mark = " ★" if pstat and "Unsolved" in str(pstat) else ""
        print(f"  {pid:>4}  {(pname or '')[:42]:<42} "
              f"{nsrc:>4}  {nv:>4}  {nspec:>4}  "
              f"{(pstat or '')[:20]}{mark}")

    costs = conn.execute("""
        SELECT provider,SUM(cost_usd),
               SUM(input_tokens),SUM(output_tokens)
        FROM cost_ledger GROUP BY provider
    """).fetchall()
    total = 0
    print(f"\n  COST")
    for prov,c,inp,out in costs:
        print(f"  {prov:<10} ${c:.4f} "
              f"in={inp:,} out={out:,}")
        total += c or 0
    print(f"  TOTAL: ${total:.4f} (₹{total*83.5:.0f})")
    conn.close()


def show_problem(problem_id: int):
    if not os.path.exists(OUTPUT_DB):
        print("Run --walk first."); return
    conn = sqlite3.connect(OUTPUT_DB)
    df   = load_problems()
    prob = df[df['ID']==problem_id]
    if prob.empty:
        print(f"Problem {problem_id} not found"); return
    p = prob.iloc[0]
    print(f"\n{'═'*70}")
    print(f"  #{problem_id}: {p['Key_Problem_or_Concept']}")
    print(f"  {p['Description']}")
    print(f"  Status: {p['Status']}")
    print(f"{'═'*70}\n")

    rows = conn.execute("""
        SELECT m.lens,m.provider,m.relevance,
               m.equation,m.reasoning,m.testable,
               v.text,v.mandala,v.sukta
        FROM matches m
        JOIN verses v ON v.id=m.verse_id
        WHERE m.problem_id=? AND m.is_generic=0
        ORDER BY m.relevance DESC LIMIT 30
    """, (problem_id,)).fetchall()

    print(f"  Non-generic matches: {len(rows)}\n")
    for lens,prov,rel,eq,reason,test,vtext,man,suk in rows:
        print(f"  [{lens}|{prov}] {rel}/10")
        print(f"   {man}/{suk}")
        print(f"   EQ:  {eq}")
        print(f"   WHY: {reason}")
        print(f"   TEST:{test}")
        print(f"   {(vtext or '')[:110]}")
        print()
    conn.close()


def show_stats():
    if not os.path.exists(OUTPUT_DB):
        print("Run --walk first."); return
    conn = sqlite3.connect(OUTPUT_DB)
    nv  = conn.execute("SELECT COUNT(*) FROM verses").fetchone()[0]
    nd  = conn.execute("SELECT COUNT(*) FROM proc_status WHERE done=1").fetchone()[0]
    nm  = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    ns  = conn.execute("SELECT COUNT(*) FROM matches WHERE is_generic=0").fetchone()[0]
    ng  = conn.execute("SELECT COUNT(*) FROM gold").fetchone()[0]
    nc  = conn.execute("SELECT SUM(cost_usd) FROM cost_ledger").fetchone()[0] or 0
    print(f"\n  Verses:    {nv:,} total / {nd:,} done "
          f"({nd/nv*100:.1f}%)" if nv else "")
    print(f"  Matches:   {nm:,} all / {ns:,} specific")
    print(f"  Gold:      {ng}")
    print(f"  Cost:      ${nc:.4f} (₹{nc*83.5:.0f})")
    conn.close()


# ── CLI ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Veda Physics 4-Lens Walker v3 — $9 full run",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walk",          action="store_true")
    ap.add_argument("--results",       action="store_true")
    ap.add_argument("--stats",         action="store_true")
    ap.add_argument("--problem",       type=int)
    ap.add_argument("--rebuild-gold",  action="store_true")
    ap.add_argument("--limit",         type=int, default=None)
    ap.add_argument("--workers",       type=int, default=4)
    ap.add_argument("--top",           type=int, default=30)
    ap.add_argument("--min-sources",   type=int, default=2)
    ap.add_argument("--unsolved-only", action="store_true")
    ap.add_argument("--no-ollama",     action="store_true")
    ap.add_argument("--groq-key",      type=str)
    ap.add_argument("--ollama-model",  type=str)
    ap.add_argument("--reasoning",     type=str, default="medium",
                    choices=["low","medium","high"])
    ap.add_argument("--budget",        type=float, default=15.00)

    global GROQ_KEY, OLLAMA_MODEL, GROQ_REASONING, BUDGET_USD
    args = ap.parse_args()
    if args.groq_key:    GROQ_KEY      = args.groq_key
    if args.ollama_model:OLLAMA_MODEL  = args.ollama_model
    GROQ_REASONING = args.reasoning
    BUDGET_USD     = args.budget

    if   args.walk:         run_walk(args.limit, args.workers,
                                     not args.no_ollama)
    elif args.results:      show_results(args.top, args.min_sources,
                                         args.unsolved_only)
    elif args.stats:        show_stats()
    elif args.problem:      show_problem(args.problem)
    elif args.rebuild_gold:
        conn = init_db()
        build_gold(conn, load_problems())
        conn.close()
    else:
        ap.print_help()
        print("""
QUICK START:
  pip install groq pandas
  ollama pull qwen2.5:32b-q4_K_M

  # Test 50 verses ($0.04):
  python veda_physics_4lens.py --walk --limit 50 --groq-key KEY

  # Full run ($9, 8 workers):
  python veda_physics_4lens.py --walk --workers 8 --groq-key KEY

  # Results:
  python veda_physics_4lens.py --results --unsolved-only --min-sources 3

  # Drill in:
  python veda_physics_4lens.py --problem 39  # Navier-Stokes
  python veda_physics_4lens.py --problem 40  # Yang-Mills
  python veda_physics_4lens.py --problem 1   # Riemann

COST: ~$9 total for all 47,899 verses
  Old version (4 Gemini calls/verse): $210
  This version (1 Groq call/verse):   $9
""")


if __name__ == "__main__":
    main()
