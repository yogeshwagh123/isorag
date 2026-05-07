"""
veda_bottleneck.py
==================
The Information Bottleneck pipeline.

NOT too wide, NOT too narrow.
Only high-confidence novel structures get through the gate.
What passes gets solved by the best available model.
Output: markdown files. Partial solutions kept. Nothing thrown away.

GATE PASS CRITERIA (the bottleneck):
    1. relevance >= 8/10
    2. non-generic equation (no placeholders)
    3. 3+ independent sources
    4. discoverer confidence >= 7/10
    5. equation appears in 2+ lenses independently

MODELS:
    Gate check:  Groq GPT-OSS-120B (fast, cheap, filters noise)
    Solver:      Claude claude-sonnet-4-6 (default) or claude-opus-4-6 (--opus)
    Cost:        Gate check ~$0.01 per candidate
                 Solver ~$0.05-0.20 per problem (Sonnet)
                         ~$0.15-0.60 per problem (Opus)

PROCESS:
    1. Watch veda_physics_4lens.db for new gold entries
    2. Apply bottleneck filter — strict criteria
    3. Gate check via Groq — is this genuinely novel?
    4. If yes: send to Claude for full solving attempt
    5. Write markdown file regardless of outcome
       (partial solution > no solution)

USAGE:
    # Run alongside the walker:
    python veda_bottleneck.py --run --groq-key KEY --claude-key KEY

    # Use Opus for maximum depth:
    python veda_bottleneck.py --run --opus --groq-key KEY --claude-key KEY

    # See what has been solved:
    python veda_bottleneck.py --report

    # Read one problem's markdown:
    python veda_bottleneck.py --read 39
"""

import os, sys, json, time, re, sqlite3
import argparse
from datetime import datetime
from pathlib import Path

try:
    from groq import Groq as GroqClient
except ImportError:
    print("FATAL: pip install groq"); sys.exit(1)

try:
    import anthropic
except ImportError:
    print("FATAL: pip install anthropic"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────
GROQ_KEY    = os.getenv("GROQ_API_KEY",       "")
CLAUDE_KEY  = os.getenv("ANTHROPIC_API_KEY",  "")
GROQ_MODEL  = "openai/gpt-oss-120b"
SONNET      = "claude-sonnet-4-6"
OPUS        = "claude-opus-4-6"

SOURCE_DB   = "veda_physics_4lens.db"
BOTTLE_DB   = "veda_bottleneck.db"
OUTPUT_DIR  = Path("solved_problems")

POLL_SEC    = 45
BUDGET_USD  = 20.00

# Bottleneck thresholds — the golden medium
MIN_RELEVANCE   = 8
MIN_SOURCES     = 3
MIN_CONFIDENCE  = 7
MIN_LENSES      = 2

GROQ_IN  = 0.15 / 1_000_000
GROQ_OUT = 0.75 / 1_000_000

_spent = 0.0


# ── Database ──────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(BOTTLE_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id    INTEGER NOT NULL,
            problem_name  TEXT,
            problem_status TEXT,
            equation      TEXT,
            n_sources     INTEGER,
            n_lenses      INTEGER,
            avg_relevance REAL,
            sources_list  TEXT,
            example_verse TEXT,
            gate_verdict  TEXT DEFAULT 'pending',
            gate_reason   TEXT,
            gate_confidence INTEGER DEFAULT 0,
            novel_structure TEXT,
            processed     INTEGER DEFAULT 0,
            cost_gate     REAL DEFAULT 0,
            cost_solver   REAL DEFAULT 0,
            created_at    REAL DEFAULT (unixepoch()),
            UNIQUE(problem_id, equation)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS solutions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id    INTEGER NOT NULL,
            problem_id      INTEGER NOT NULL,
            problem_name    TEXT,
            model_used      TEXT,
            verdict         TEXT,
            completeness    INTEGER DEFAULT 0,
            markdown_path   TEXT,
            formalised_eq   TEXT,
            proof_sketch    TEXT,
            open_questions  TEXT,
            next_steps      TEXT,
            cost_usd        REAL DEFAULT 0,
            created_at      REAL DEFAULT (unixepoch())
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            stage     TEXT,
            model     TEXT,
            inp       INTEGER,
            out       INTEGER,
            cost_usd  REAL,
            ts        REAL DEFAULT (unixepoch())
        )
    """)

    conn.commit()
    OUTPUT_DIR.mkdir(exist_ok=True)
    return conn


# ── API calls ─────────────────────────────────────────────────
def call_groq(prompt: str, system: str) -> tuple:
    client = GroqClient(api_key=GROQ_KEY)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role":"system","content":system},
                          {"role":"user",  "content":prompt}],
                temperature=0.1,
                max_completion_tokens=2048,
                reasoning_effort="high",
            )
            txt = r.choices[0].message.content.strip()
            return txt, r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(20); continue
            if attempt == 2: return f"ERR:{e}", 0, 0
            time.sleep(2**attempt)
    return "ERR:retries", 0, 0


def call_claude(prompt: str, system: str,
                use_opus: bool = False) -> tuple:
    model  = OPUS if use_opus else SONNET
    client = anthropic.Anthropic(api_key=CLAUDE_KEY)
    try:
        r = client.messages.create(
            model=model,
            max_tokens=8000,
            system=system,
            messages=[{"role":"user","content":prompt}]
        )
        txt = r.content[0].text
        inp = r.usage.input_tokens
        out = r.usage.output_tokens
        return txt, inp, out, model
    except Exception as e:
        return f"ERR:{e}", 0, 0, model


def safe_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1] if lines[-1].strip()=="```" else lines[1:])
    try:
        r = json.loads(text)
        return r if isinstance(r, dict) else {}
    except Exception:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                r = json.loads(m.group())
                return r if isinstance(r, dict) else {}
            except Exception: pass
    return {}


def track(conn, stage, model, inp, out,
          is_claude=False) -> float:
    global _spent
    if is_claude:
        # Claude Sonnet 4.6: $3/1M input, $15/1M output
        # Claude Opus 4.6:   $15/1M input, $75/1M output
        if "opus" in model:
            cost = inp*15/1e6 + out*75/1e6
        else:
            cost = inp*3/1e6  + out*15/1e6
    else:
        cost = inp*GROQ_IN + out*GROQ_OUT
    _spent += cost
    if _spent > BUDGET_USD * 0.98:
        raise RuntimeError(f"BUDGET: ${_spent:.4f}")
    conn.execute(
        "INSERT INTO cost_log (stage,model,inp,out,cost_usd) "
        "VALUES (?,?,?,?,?)",
        (stage, model, inp, out, cost))
    conn.commit()
    return cost


# ── GATE CHECK prompt ─────────────────────────────────────────
GATE_SYSTEM = """You are a rigorous mathematician acting as a filter.
Your job is to decide if a proposed mathematical structure deserves
deeper investigation by the world's best AI model.

Be extremely strict. Most proposals are noise.
A proposal passes ONLY if it is:
1. Not already known under any name in any field
2. Expressible as a precise mathematical statement
3. Not obviously false
4. Connected to a real open problem in a non-trivial way

Respond ONLY in valid JSON."""

GATE_PROMPT = """PROPOSED STRUCTURE:
Problem: {problem_name} (Status: {problem_status})
Known equation matched: {equation}
Sources: {n_sources} independent texts
Average relevance: {avg_relevance}/10
Example verse: {verse}
Sources: {sources}

GATE QUESTION: Does this verse structure suggest ANY mathematical
relationship that goes BEYOND the known equation above?

Not a restatement. Not a metaphor. An actual mathematical relationship
that could be written as a theorem, conjecture, or inequality that
is not already in the literature.

If yes: write it precisely.
If no: say no clearly.

JSON response:
{{
  "passes_gate": true/false,
  "reason": "precise one sentence",
  "novel_structure": "if passes: write the mathematical statement precisely",
  "mathematical_type": "conservation/symmetry/scaling/dynamics/inequality/other",
  "variables_defined": "define each variable",
  "not_same_as_known_because": "why this differs from {equation}",
  "confidence": 0-10,
  "if_true_implies": "what would follow mathematically if this is correct"
}}"""


# ── SOLVER prompt (Claude) ────────────────────────────────────
SOLVER_SYSTEM = """You are the world's most capable mathematical physicist.
You have been given a proposed novel mathematical structure that has passed
a rigorous gate check. Your job is to attempt to solve or significantly
advance understanding of the associated problem.

Be completely honest about what you can and cannot prove.
A rigorous partial result is worth more than a speculative complete solution.
Do not throw the baby out with the bathwater — partial progress is real progress.

Write your response as a structured mathematical analysis.
This will be saved as a permanent markdown document."""

SOLVER_PROMPT = """# Problem: {problem_name}

## Status
{problem_status}

## Source
Corpus: {sources}
Verse: {verse}

## Known Equation
{known_equation}

## Proposed Novel Structure (passed gate check)
{novel_structure}

Variable definitions: {variables}
Mathematical type: {math_type}

## Your Task

Attempt the following in order. Stop when you reach the limit of
what can be rigorously established. Do not speculate beyond that point.

### 1. Verify the Gate
Is this structure genuinely novel? Is it precisely stated?
If not, state why and stop.

### 2. Formalise
Write this as a precise mathematical statement.
Define: the space, the measure, the operators, the domain.
Write it as: "Theorem (proposed): For all [X] in [space], [relation]"

### 3. Attempt Proof
Try to prove it. Use:
- Noether's theorem if a symmetry is involved
- Tishby's Information Bottleneck if information compression is involved
- Kolmogorov's 1941 theory if turbulence scaling is involved
- Atiyah-Singer if topology is involved
- Whatever is appropriate

If you cannot prove it, state exactly where the proof breaks down.
That breakdown point is the most valuable output.

### 4. Partial Results
What CAN be proven? Even a lemma is valuable.
State it precisely.

### 5. Connection to Unsolved Problems
Does this, if proven, imply anything about:
- Navier-Stokes global regularity?
- Yang-Mills mass gap?
- Riemann Hypothesis?
- Any other Clay Millennium Prize problem?

### 6. Falsification
Give ONE specific numerical computation that would disprove this.
Give the exact values to compute.

### 7. Next Steps
What should a human mathematician do next?
Name the specific mathematical tools they would need.
Name the specific papers they should read.

### 8. Honest Assessment
On a scale 0-10:
- Mathematical rigour of this proposal: X/10
- Probability it leads somewhere: X/10
- Your confidence in the partial results: X/10

Be brutally honest. The person reading this is not a mathematician
but they will show it to one. Precision matters more than encouragement."""


# ── Bottleneck filter ─────────────────────────────────────────
def passes_bottleneck(row: tuple) -> bool:
    """The golden medium. Not too wide, not too narrow."""
    (pid, pname, pstat, eq, nsrc, nlens, ar, srcs, verse) = row

    # Hard filters first (cheap)
    if ar < MIN_RELEVANCE:
        return False
    if nsrc < MIN_SOURCES:
        return False

    # Equation quality check
    if not eq or len(eq.strip()) < 10:
        return False

    # Must not be a pure generic
    generic = [
        'dX/dt = f(X)', 'Y propto X^n', 'f(T(x)) = f(x)',
        'Y ∝ X^n', 'dX/dt=0'
    ]
    if any(eq.strip() == g for g in generic):
        return False

    return True


# ── Main loop ─────────────────────────────────────────────────
def run_pipeline(use_opus: bool = False):
    global _spent

    if not GROQ_KEY:
        print("ERROR: --groq-key required"); sys.exit(1)
    if not CLAUDE_KEY:
        print("ERROR: --claude-key required"); sys.exit(1)

    conn     = init_db()
    model_name = OPUS if use_opus else SONNET

    print(f"\n{'═'*65}")
    print(f"  VEDA INFORMATION BOTTLENECK PIPELINE")
    print(f"{'═'*65}")
    print(f"  Gate model:   Groq GPT-OSS-120B (reasoning=high)")
    print(f"  Solver model: {model_name}")
    print(f"  Output dir:   {OUTPUT_DIR}/")
    print(f"  Budget:       ${BUDGET_USD}")
    print(f"  Bottleneck thresholds:")
    print(f"    Relevance:  >= {MIN_RELEVANCE}/10")
    print(f"    Sources:    >= {MIN_SOURCES} independent")
    print(f"    Confidence: >= {MIN_CONFIDENCE}/10 (gate check)")
    print(f"{'═'*65}\n")
    print(f"  Watching {SOURCE_DB}...")
    print(f"  Press Ctrl+C to stop\n")

    processed_keys = set()

    while True:
        try:
            if not os.path.exists(SOURCE_DB):
                print(f"  Waiting for {SOURCE_DB}...")
                time.sleep(POLL_SEC)
                continue

            src = sqlite3.connect(SOURCE_DB)

            # Pull gold entries that pass hard filters
            gold = src.execute("""
                SELECT g.problem_id, g.problem_name,
                       g.problem_status, g.equation,
                       g.n_sources, g.n_verses,
                       g.avg_relevance, g.sources_list,
                       g.example_verse
                FROM gold g
                WHERE g.n_sources >= ?
                  AND g.avg_relevance >= ?
                ORDER BY g.avg_relevance DESC, g.n_sources DESC
            """, (MIN_SOURCES, MIN_RELEVANCE)).fetchall()

            src.close()

            new_candidates = [
                r for r in gold
                if f"{r[0]}_{r[3][:20]}" not in processed_keys
                and passes_bottleneck(r)
            ]

            if not new_candidates:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"No new candidates. "
                      f"Gold entries checked: {len(gold)}. "
                      f"Waiting {POLL_SEC}s...")
                time.sleep(POLL_SEC)
                continue

            print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{len(new_candidates)} new candidates passed hard filter")

            for row in new_candidates:
                (pid, pname, pstat, eq,
                 nsrc, nv, ar, srcs, verse) = row
                key = f"{pid}_{eq[:20]}"

                print(f"\n  ── Candidate ──────────────────────────")
                print(f"  Problem:  #{pid} {pname}")
                print(f"  Status:   {pstat}")
                print(f"  Equation: {eq}")
                print(f"  Sources:  {nsrc} | AvgRel: {ar:.1f}")

                # ── GATE CHECK ────────────────────────────
                print(f"  Running gate check (Groq)...")
                gate_prompt = GATE_PROMPT.format(
                    problem_name=pname,
                    problem_status=pstat,
                    equation=eq,
                    n_sources=nsrc,
                    avg_relevance=ar,
                    verse=(verse or "")[:200],
                    sources=(srcs or "")[:100]
                )
                gate_txt, g_inp, g_out = call_groq(
                    gate_prompt, GATE_SYSTEM)
                gate_cost = track(
                    conn, "gate", GROQ_MODEL, g_inp, g_out)
                gate = safe_json(gate_txt)

                passes   = gate.get("passes_gate", False)
                g_conf   = int(gate.get("confidence", 0))
                g_reason = gate.get("reason", "")
                novel    = gate.get("novel_structure", "")

                # Save candidate
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO candidates
                        (problem_id,problem_name,problem_status,
                         equation,n_sources,n_lenses,avg_relevance,
                         sources_list,example_verse,gate_verdict,
                         gate_reason,gate_confidence,novel_structure,
                         cost_gate)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (pid, pname, pstat, eq, nsrc, nv, ar,
                          srcs, verse,
                          "pass" if passes else "fail",
                          g_reason, g_conf, novel, gate_cost))
                    conn.commit()
                except Exception:
                    pass

                if not passes or g_conf < MIN_CONFIDENCE:
                    print(f"  ✗ Gate FAIL (conf={g_conf}): {g_reason}")
                    processed_keys.add(key)
                    continue

                print(f"  ✓ Gate PASS (conf={g_conf}/10)")
                print(f"  Novel: {novel[:100]}")

                # ── CLAUDE SOLVER ─────────────────────────
                print(f"  Sending to {model_name}...")

                solver_prompt = SOLVER_PROMPT.format(
                    problem_name=pname,
                    problem_status=pstat,
                    sources=(srcs or ""),
                    verse=(verse or "")[:300],
                    known_equation=eq,
                    novel_structure=novel,
                    variables=gate.get("variables_defined",""),
                    math_type=gate.get("mathematical_type","")
                )

                s_txt, s_inp, s_out, s_model = call_claude(
                    solver_prompt, SOLVER_SYSTEM, use_opus)
                s_cost = track(conn, "solver", s_model,
                               s_inp, s_out, is_claude=True)

                # ── Write markdown ────────────────────────
                safe_name = re.sub(r'[^\w\s-]', '',
                                   pname).strip().replace(' ','_')
                md_path = OUTPUT_DIR / f"problem_{pid}_{safe_name}.md"

                md_content = f"""# Problem #{pid}: {pname}

**Status:** {pstat}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {s_model}
**Cost:** ${s_cost:.4f}

---

## Corpus Evidence

**Equation found:** `{eq}`
**Independent sources:** {nsrc}
**Average relevance:** {ar:.1f}/10
**Sources:** {srcs}

**Example verse:**
> {verse or 'N/A'}

---

## Gate Check Result

**Verdict:** PASSED (confidence {g_conf}/10)
**Novel structure identified:** {novel}
**Gate reasoning:** {g_reason}

---

## Mathematical Analysis

{s_txt}

---

## Metadata

- Walker database: `{SOURCE_DB}`
- Bottleneck database: `{BOTTLE_DB}`
- This file: `{md_path}`
- Total pipeline cost so far: ${_spent:.4f}
"""

                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)

                print(f"  ✓ Saved: {md_path}")

                # Save solution record
                conn.execute("""
                    INSERT INTO solutions
                    (candidate_id,problem_id,problem_name,
                     model_used,markdown_path,cost_usd)
                    VALUES (
                        (SELECT id FROM candidates
                         WHERE problem_id=? AND equation=?),
                        ?,?,?,?,?)
                """, (pid, eq, pid, pname, s_model,
                      str(md_path), s_cost))
                conn.commit()

                processed_keys.add(key)
                print(f"  Total spent so far: ${_spent:.4f}")

            time.sleep(5)

        except RuntimeError as e:
            print(f"\n  STOPPED: {e}")
            break
        except KeyboardInterrupt:
            print("\n  Stopped by user")
            break
        except Exception as e:
            print(f"  ERR: {e}")
            time.sleep(POLL_SEC)

    conn.close()
    print(f"\n  Total cost: ${_spent:.4f}")


# ── Report ────────────────────────────────────────────────────
def show_report():
    if not os.path.exists(BOTTLE_DB):
        print("No results yet. Run --run first."); return

    conn = sqlite3.connect(BOTTLE_DB)

    print(f"\n{'═'*70}")
    print(f"  BOTTLENECK PIPELINE REPORT")
    print(f"{'═'*70}")

    cands = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN gate_verdict='pass' "
        "THEN 1 ELSE 0 END) FROM candidates").fetchone()
    sols  = conn.execute(
        "SELECT COUNT(*) FROM solutions").fetchone()[0]
    cost  = conn.execute(
        "SELECT SUM(cost_usd) FROM cost_log").fetchone()[0] or 0

    print(f"\n  Candidates evaluated: {cands[0]}")
    print(f"  Passed gate:          {cands[1]}")
    print(f"  Solutions generated:  {sols}")
    print(f"  Total cost:           ${cost:.4f} (₹{cost*83.5:.0f})")

    print(f"\n  SOLUTIONS:")
    rows = conn.execute("""
        SELECT s.problem_id, s.problem_name,
               s.model_used, s.markdown_path, s.cost_usd,
               c.gate_confidence, c.novel_structure
        FROM solutions s
        JOIN candidates c ON c.problem_id = s.problem_id
        ORDER BY s.created_at DESC
    """).fetchall()

    for pid, pname, model, path, cost, gconf, novel in rows:
        exists = "✓" if os.path.exists(path or "") else "✗"
        print(f"\n  [{exists}] #{pid} {pname}")
        print(f"       Model: {model} | Cost: ${cost:.4f}")
        print(f"       Gate conf: {gconf}/10")
        print(f"       Novel: {(novel or '')[:80]}")
        print(f"       File: {path}")

    conn.close()


def read_solution(problem_id: int):
    """Print one solution markdown to terminal."""
    if not os.path.exists(BOTTLE_DB):
        print("No results yet."); return
    conn = sqlite3.connect(BOTTLE_DB)
    row = conn.execute(
        "SELECT markdown_path FROM solutions WHERE problem_id=? "
        "ORDER BY created_at DESC LIMIT 1",
        (problem_id,)).fetchone()
    conn.close()
    if not row or not row[0]:
        print(f"No solution for problem {problem_id}")
        return
    path = row[0]
    if not os.path.exists(path):
        print(f"File not found: {path}"); return
    with open(path, encoding='utf-8') as f:
        print(f.read())


# ── CLI ───────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Veda Information Bottleneck Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",        action="store_true",
                    help="Run the pipeline")
    ap.add_argument("--report",     action="store_true",
                    help="Show results report")
    ap.add_argument("--read",       type=int,
                    help="Read solution for problem ID")
    ap.add_argument("--opus",       action="store_true",
                    help="Use Claude Opus instead of Sonnet")
    ap.add_argument("--groq-key",   type=str)
    ap.add_argument("--claude-key", type=str)
    ap.add_argument("--budget",     type=float, default=20.0)
    ap.add_argument("--min-rel",    type=float, default=8.0)
    ap.add_argument("--min-src",    type=int,   default=3)
    ap.add_argument("--min-conf",   type=int,   default=7)

    args = ap.parse_args()
    global GROQ_KEY, CLAUDE_KEY, BUDGET_USD
    global MIN_RELEVANCE, MIN_SOURCES, MIN_CONFIDENCE
    if args.groq_key:   GROQ_KEY   = args.groq_key
    if args.claude_key: CLAUDE_KEY = args.claude_key
    BUDGET_USD     = args.budget
    MIN_RELEVANCE  = args.min_rel
    MIN_SOURCES    = args.min_src
    MIN_CONFIDENCE = args.min_conf

    if   args.run:    run_pipeline(use_opus=args.opus)
    elif args.report: show_report()
    elif args.read:   read_solution(args.read)
    else:
        ap.print_help()
        print(f"""
USAGE:

  # Terminal 1 — walker (already running or start fresh):
  python veda_physics_4lens.py --walk --workers 8 --groq-key KEY

  # Terminal 2 — bottleneck pipeline (start now):
  python veda_bottleneck.py --run \\
      --groq-key KEY --claude-key KEY

  # Terminal 2 with Opus (deeper analysis):
  python veda_bottleneck.py --run --opus \\
      --groq-key KEY --claude-key KEY

  # See what has been solved:
  python veda_bottleneck.py --report

  # Read Navier-Stokes solution:
  python veda_bottleneck.py --read 39

  # Read Yang-Mills solution:
  python veda_bottleneck.py --read 40

BOTTLENECK (the golden medium):
  Relevance >= 8/10        (not too wide)
  Sources >= 3 independent (not too narrow)
  Gate confidence >= 7/10  (Groq checks novelty)
  Only what passes → Claude Sonnet/Opus

OUTPUT:
  solved_problems/problem_39_Navier-Stokes.md
  solved_problems/problem_40_Yang-Mills.md
  ... one file per problem that passes the gate

COST ESTIMATE:
  Gate check (Groq):    ~$0.01 per candidate
  Solver (Sonnet):      ~$0.10 per problem
  Solver (Opus):        ~$0.50 per problem
  Expected candidates:  10-50 from full corpus
  Expected total:       $5-$25 depending on model

NOTHING IS THROWN AWAY:
  Partial solutions are saved.
  Failed gate checks are recorded.
  All evidence stays in {BOTTLE_DB}.
  You can re-run with lower thresholds anytime.
""")


if __name__ == "__main__":
    main()
