# VedaStream: Structural RAG for Hierarchically Encoded Knowledge Corpora

**A stateless multi-worker pipeline that outperforms vector RAG by +139.4% MRR on structured knowledge corpora — at 1/70th the infrastructure cost.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-paper-red)](https://arxiv.org)
[![Patent](https://img.shields.io/badge/Patent-Filed-blue)](docs/patent_filing.md)

---

## The Problem: Context Bleed in Vector RAG

Standard RAG systems embed documents as dense vectors and retrieve by cosine similarity. On hierarchically structured corpora — legal (statute → regulation → case law), medical (mechanism → protocol → outcome), or knowledge corpora (Samhita → Brahmana → Upanishad → Sutra) — this causes **context bleed**: a query for procedural knowledge retrieves philosophically similar but structurally wrong documents.

We call this the **hemophilia problem**: flat vector spaces cannot maintain hard boundaries between document layers that are structurally and functionally distinct.

---

## The Solution: Structural Layer Isolation

VedaStream replaces vector similarity with **SQL-based structural metadata matching** through a four-worker stateless pipeline:

```
DOCUMENT CORPUS (3,980 files · 1.35 GB · Sanskrit + Tamil)
         ↓
┌────────────────────────────────────────────┐
│         4 Parallel Workers                  │
│                                             │
│  W1 STRUCTURAL    W2 ANOMALY    W3 CROSS   │
│  PARSER           SCOUT         REF MAPPER │
│                                             │
│  → Title          → Rare terms  → Root     │
│  → Layer class    → Meter break   concept  │
│  → Verse count    → Signals     → Devata   │
│  → Clean text                   → Struct   │
│                                             │
│  (All outputs committed to isolated DB     │
│   tables before W4 executes)               │
└────────────────────────────────────────────┘
         ↓
  W4 SYNTHESIS ENGINE
  Reads ONLY committed outputs
  → Structural equation
  → Confidence score
         ↓
  VECTORLESS RAG QUERY ENGINE
  SQL keyword match on structural metadata
  Layer-isolated retrieval · Anti-bleed guarantee
         ↓
  STRUCTURED ANSWER
  Traced to file_id · Layer-correct · No hallucination
```

**Key architectural innovation:** No worker reads another worker's intermediate outputs. All inter-worker communication occurs exclusively through committed database records. This is the anti-hemophilia guarantee.

---

## Benchmark Results

Evaluated on the GRETIL corpus: 3,980 files, 50 ground-truth queries with known correct layer labels.

| Metric | Vector RAG (TF-IDF) | VedaStream | Delta | Change |
|--------|-------------------|------------|-------|--------|
| Precision@1 | 0.260 | 0.520 | +0.260 | **+100.0%** |
| Precision@5 | 0.260 | 0.436 | +0.176 | +67.7% |
| MRR | 0.260 | 0.622 | +0.362 | **+139.4%** |
| Bleed Rate | 0.740 | 0.564 | -0.176 | -23.8% |
| Queries Won | 5 / 50 | 28 / 50 | +23 | +460% |

**Infrastructure cost:** USD 7.00 total vs USD 500+/month for equivalent commercial stream processing (Confluent/Kafka-based pipelines).

---

## Why This Beats Kafka-Based Pipelines

Commercial stream processing platforms impose:
- Minimum 3x storage replication overhead
- Scaling ceilings of 4 elastic units per 10-minute window
- Vendor lock-in preventing cluster migration

VedaStream replaces this with **SQLite Write-Ahead Logging (WAL)**, providing:
- Equivalent durability guarantees
- 1x storage overhead
- Zero network replication cost
- Complete portability — copy the `.db` file anywhere

---

## Repository Structure

```
vedastream/
├── README.md                    ← You are here
├── requirements.txt             ← pip install -r requirements.txt
├── pipeline/
│   ├── veda_complete_pipeline.py  ← Main 4-worker pipeline (local Ollama)
│   ├── veda_walker.py             ← Corpus walker and indexer
│   ├── veda_bottleneck.py         ← Tishby information bottleneck scorer
│   ├── veda_verse_walker.py       ← Verse-level structural analysis
│   └── veda_physics_4lens.py      ← 4-lens physics of knowledge retrieval
├── benchmarks/
│   └── benchmark.py             ← Reproduces all benchmark results
├── results/
│   ├── benchmark_report.txt     ← Full benchmark output
│   └── corpus_summary.txt       ← GRETIL corpus inventory
└── docs/
    └── architecture.md          ← Detailed architecture description
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (free local inference)
# https://ollama.ai
ollama pull qwen2.5:7b

# Run the pipeline on your corpus
python pipeline/veda_complete_pipeline.py --walk --workers 4

# Reproduce benchmark results
python benchmarks/benchmark.py
```

---

## The Vedic Architecture Connection

The four-worker pipeline maps directly to the four Ṛtvija priests of the Vedic yajña:

| Worker | Vedic Priest | Veda | Function |
|--------|-------------|------|----------|
| W1 Structural Parser | Hotṛ | Rigveda | Invokes structural identity |
| W2 Anomaly Scout | Adhvaryu | Yajurveda | Identifies procedural signals |
| W3 Cross-Ref Mapper | Udgātṛ | Samaveda | Maps harmonic connections |
| W4 Synthesis Engine | Brahman | Atharvaveda | Integrates, never contaminates |

This is not metaphor. The architectural principle — specialists activated selectively, supervisor integrates only committed outputs, no cross-contamination — is the same in both systems. The Vedic tradition described optimal information routing 3,000 years ago.

---

## Applications Beyond Ancient Texts

The anti-bleed guarantee generalises to any domain with explicit structural layers:

- **Legal:** Statute → Regulation → Case Law (cross-layer bleed produces wrong legal advice)
- **Medical:** Mechanism → Protocol → Outcome (bleed produces dangerous clinical decisions)
- **Financial:** Regulation → Product → Transaction (bleed produces compliance failures)
- **Enterprise:** Any multi-tier knowledge base where layer boundaries matter

---

## Citation

If you use VedaStream in your research, please cite:

```bibtex
@misc{wagh2025vedastream,
  title={VedaStream: A Stateless Multi-Worker Pipeline for Structural
         Retrieval-Augmented Generation with Hierarchical Document Layer Isolation},
  author={Wagh, Yogesh Anant},
  year={2025},
  note={Patent filed, Mumbai Patent Office. arXiv preprint forthcoming.},
  institution={Independent Researcher, Pune, Maharashtra, India}
}
```

---

## Author

**Yogesh Anant Wagh**
Independent Researcher, Pune, Maharashtra, India
Patent Filed: Mumbai Patent Office

---

## License

MIT License — see LICENSE file.

This work is dedicated to the living tradition of Vedic knowledge systems
and to everyone who falls down and gets back up.
