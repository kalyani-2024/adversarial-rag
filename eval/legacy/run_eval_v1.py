"""
HISTORICAL (v1) - kept for reference only; do not run. It deletes the live index on import.
Its numbers are in summary_v1.json. The current harness is `python -m eval.run_eval`.


Adversarial RAG evaluation: single-pass vs full adversarial pipeline.

For each of N auto-generated questions:
  - Run single-pass: Retriever -> Generator
  - Run full:        Retriever -> Generator -> Adversary -> Synthesizer
  - Judge both with Groq LLM-as-judge on faithfulness + completeness (1-5)
  - Time each
Outputs eval/results.json + eval/report.md.
"""

import os
import sys
import json
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_groq import ChatGroq

# -- Reset index so the eval is on this doc only -------------------------------
INDEX_DIR = ROOT / "faiss_index"
for f in ("embeddings.npy", "metadata.json"):
    p = INDEX_DIR / f
    if p.exists():
        p.unlink()

from core.vector_store import vector_store  # imports AFTER reset
from agents.retriever import retriever_node
from agents.generator import generator_node
from agents.adversary import adversary_node
from agents.synthesizer import synthesizer_node

# -- Ingest source -------------------------------------------------------------
SRC = ROOT / "eval" / "source.txt"
text = SRC.read_text(encoding="utf-8")
n_chunks = vector_store.ingest(text)
print(f"[ingest] {n_chunks} chunks indexed")

# -- LLM helpers ---------------------------------------------------------------
judge = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
)

qgen = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
)


def gen_questions(doc_text: str, n: int = 20) -> list[dict]:
    """Generate question/reference-answer pairs grounded in the document."""
    prompt = (
        f"Below is a research paper. Generate exactly {n} factual questions a reader "
        "might ask, each answerable directly from the paper. Mix easy lookup questions "
        "(specific numbers/metrics) with harder synthesis questions. For each, provide "
        "the canonical answer drawn ONLY from the paper.\n\n"
        "Return STRICT JSON: a list of objects with keys 'q' and 'a'. No prose, no markdown.\n\n"
        f"PAPER:\n{doc_text[:14000]}"
    )
    resp = qgen.invoke([{"role": "user", "content": prompt}]).content
    # Extract JSON array
    m = re.search(r"\[.*\]", resp, re.DOTALL)
    if not m:
        raise RuntimeError(f"qgen returned no JSON: {resp[:500]}")
    return json.loads(m.group(0))


# -- Pipelines -----------------------------------------------------------------
def run_single_pass(query: str) -> dict:
    t0 = time.perf_counter()
    s = {"query": query}
    s.update(retriever_node(s))
    s.update(generator_node(s))
    return {"answer": s["initial_answer"], "chunks": s["chunks"], "latency": time.perf_counter() - t0}


def run_full(query: str) -> dict:
    t0 = time.perf_counter()
    s = {"query": query}
    s.update(retriever_node(s))
    s.update(generator_node(s))
    s.update(adversary_node(s))
    s.update(synthesizer_node(s))
    return {"answer": s["final_answer"], "chunks": s["chunks"], "latency": time.perf_counter() - t0}


# -- Judge ---------------------------------------------------------------------
JUDGE_PROMPT = """You are an impartial evaluator scoring an answer to a question.

QUESTION: {q}

REFERENCE ANSWER (ground truth from the source document):
{ref}

CANDIDATE ANSWER:
{cand}

Score the CANDIDATE on three axes from 1 to 5 (5 is best):
- faithfulness: are all claims supported by the reference / does it avoid hallucinated facts?
- completeness: does it cover the key points of the reference?
- correctness: is it factually right where it overlaps with the reference?

Return STRICT JSON only: {{"faithfulness": <int>, "completeness": <int>, "correctness": <int>, "notes": "<one short sentence>"}}
"""


def judge_answer(q: str, ref: str, cand: str) -> dict:
    resp = judge.invoke([{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, cand=cand)}]).content
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return {"faithfulness": 0, "completeness": 0, "correctness": 0, "notes": "judge parse failed"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithfulness": 0, "completeness": 0, "correctness": 0, "notes": "judge parse failed"}


# -- Main loop -----------------------------------------------------------------
def main():
    print("[qgen] generating questions...")
    qs = gen_questions(text, n=20)
    print(f"[qgen] got {len(qs)} questions")

    results = []
    for i, item in enumerate(qs, 1):
        q, ref = item["q"], item["a"]
        print(f"\n[{i}/{len(qs)}] {q[:80]}")

        sp = run_single_pass(q)
        fp = run_full(q)

        sp_score = judge_answer(q, ref, sp["answer"])
        fp_score = judge_answer(q, ref, fp["answer"])

        results.append({
            "q": q, "ref": ref,
            "single_pass": {"answer": sp["answer"], "latency": sp["latency"], **sp_score},
            "full":        {"answer": fp["answer"], "latency": fp["latency"], **fp_score},
        })
        print(f"  single faith={sp_score.get('faithfulness')} comp={sp_score.get('completeness')} corr={sp_score.get('correctness')} ({sp['latency']:.1f}s)")
        print(f"  full   faith={fp_score.get('faithfulness')} comp={fp_score.get('completeness')} corr={fp_score.get('correctness')} ({fp['latency']:.1f}s)")

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out}")

    # -- Aggregate -------------------------------------------------------------
    def avg(xs): return sum(xs) / len(xs) if xs else 0.0

    sp_f = [r["single_pass"]["faithfulness"] for r in results]
    sp_c = [r["single_pass"]["completeness"] for r in results]
    sp_x = [r["single_pass"]["correctness"]  for r in results]
    sp_l = [r["single_pass"]["latency"]      for r in results]
    fp_f = [r["full"]["faithfulness"]        for r in results]
    fp_c = [r["full"]["completeness"]        for r in results]
    fp_x = [r["full"]["correctness"]         for r in results]
    fp_l = [r["full"]["latency"]             for r in results]

    # "Hallucination" proxy: faithfulness <= 3 (i.e., not "good" or "excellent")
    sp_hall = sum(1 for s in sp_f if s <= 3) / len(sp_f)
    fp_hall = sum(1 for s in fp_f if s <= 3) / len(fp_f)

    summary = {
        "n": len(results),
        "single_pass": {
            "faithfulness_avg": avg(sp_f),
            "completeness_avg": avg(sp_c),
            "correctness_avg":  avg(sp_x),
            "latency_avg_s":    avg(sp_l),
            "hallucination_rate": sp_hall,
        },
        "full": {
            "faithfulness_avg": avg(fp_f),
            "completeness_avg": avg(fp_c),
            "correctness_avg":  avg(fp_x),
            "latency_avg_s":    avg(fp_l),
            "hallucination_rate": fp_hall,
        },
    }
    summary["delta"] = {
        "faithfulness_pct": (summary["full"]["faithfulness_avg"] - summary["single_pass"]["faithfulness_avg"]) / max(summary["single_pass"]["faithfulness_avg"], 1e-9) * 100,
        "completeness_pct": (summary["full"]["completeness_avg"] - summary["single_pass"]["completeness_avg"]) / max(summary["single_pass"]["completeness_avg"], 1e-9) * 100,
        "correctness_pct":  (summary["full"]["correctness_avg"]  - summary["single_pass"]["correctness_avg"])  / max(summary["single_pass"]["correctness_avg"], 1e-9) * 100,
        "hallucination_reduction_pct": (sp_hall - fp_hall) / max(sp_hall, 1e-9) * 100,
        "latency_overhead_x": summary["full"]["latency_avg_s"] / max(summary["single_pass"]["latency_avg_s"], 1e-9),
    }

    (ROOT / "eval" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
