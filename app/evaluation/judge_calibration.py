"""Judge calibration: does the pipeline judge catch seeded faults?

The reliability loop is only as good as its judge. This suite feeds the judge
hand-made answers with known defects (wrong number, outside knowledge,
incomplete, off-topic) plus correct controls, and measures detection rate
(fault -> FAIL) and false-alarm rate (correct -> FAIL).
"""

from __future__ import annotations

from typing import Any

from app.core.llm import LLMClient
from app.generation.judge import judge_answer
from app.schemas.reliability import Thresholds
from app.schemas.retrieval import RetrievedChunk

_CTX = [
    RetrievedChunk(chunk_id="p:0", document_id="p", document_name="paper.txt", chunk_index=0, text=(
        "Transformer (Ours, personalised): ROC-AUC 0.959 [0.950, 0.968]; F1 0.774; Recall 0.894. "
        "LSTM Autoencoder: ROC-AUC 0.601; Recall 0.994. Hardware: NVIDIA RTX 4060 GPU. "
        "Configuration: d_model=128, 4 attention heads, 3 encoder layers, dropout 0.2.")),
    RetrievedChunk(chunk_id="p:1", document_id="p", document_name="paper.txt", chunk_index=1, text=(
        "Fairness Calibration: IMD Band DPR before 0.870, after 1.000. Note: this validates the calibration "
        "MECHANISM, not real demographic fairness, since demographics are synthetic.")),
]

# (question, answer, is_faulty, fault_type)
CASES: list[tuple[str, str, bool, str]] = [
    ("What ROC-AUC does the Transformer achieve?", "The Transformer achieves ROC-AUC 0.959 [1].", False, "correct"),
    ("What ROC-AUC does the Transformer achieve?", "The Transformer achieves ROC-AUC 0.989 [1].", True, "wrong_number"),
    ("How does the Transformer compare to the LSTM autoencoder?",
     "The Transformer reaches ROC-AUC 0.959 versus 0.601 for the LSTM autoencoder [1].", False, "correct"),
    ("How does the Transformer compare to the LSTM autoencoder?",
     "The Transformer reaches ROC-AUC 0.959 versus 0.601 for the LSTM autoencoder [1]. This is because attention "
     "avoids the vanishing-gradient problem of recurrent networks, as shown by Vaswani et al. (2017).", True, "outside_knowledge"),
    ("What hardware and architecture were used?",
     "Experiments ran on an NVIDIA RTX 4060 with a 128-dimensional, 4-head, 3-layer Transformer [1].", False, "correct"),
    ("What hardware and architecture were used?",
     "Experiments ran on an NVIDIA A100 cluster with a 12-layer Transformer [1].", True, "contradiction"),
    ("Is the system fair on real students?",
     "Not demonstrated: calibration was validated only on synthetic demographics [2].", False, "correct"),
    ("Is the system fair on real students?",
     "Yes. Fairness calibration achieved DPR 1.000, proving the system is fair for real students [2].", True, "overclaim"),
    ("What ROC-AUC does the Transformer achieve?", "It performs well [1].", True, "incomplete"),
    ("What ROC-AUC does the Transformer achieve?", "The model ran on an RTX 4060 GPU [1].", True, "off_topic"),
]


def run_calibration(llm: LLMClient, thresholds: Thresholds, model: str) -> dict[str, Any]:
    rows = []
    for question, answer, faulty, kind in CASES:
        result, _ = judge_answer(llm, question, _CTX, answer, thresholds, model=model)
        rows.append({"kind": kind, "faulty": faulty, "verdict": result.verdict, "failed_checks": result.failed_checks,
                     "faithfulness": result.faithfulness, "relevance": result.relevance, "completeness": result.completeness})
    faults = [r for r in rows if r["faulty"]]
    controls = [r for r in rows if not r["faulty"]]
    return {
        "model": model,
        "fault_detection_rate": sum(r["verdict"] == "FAIL" for r in faults) / len(faults),
        "false_alarm_rate": sum(r["verdict"] == "FAIL" for r in controls) / len(controls),
        "n_faults": len(faults),
        "n_controls": len(controls),
        "cases": rows,
    }
