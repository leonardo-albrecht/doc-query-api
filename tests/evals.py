import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import logging
from dataclasses import dataclass, field
from app.retrieval import buscar_chunks
from app.generation import gerar_resposta

logging.disable(logging.CRITICAL)


@dataclass
class EvalCase:
    question: str
    expect_refusal: bool = False          # True = esperamos que o sistema recuse
    expected_keywords: list[str] = field(default_factory=list)  # palavras que devem aparecer na resposta


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="O que é engenharia da persuasão?",
        expect_refusal=False,
        expected_keywords=["persuasão", "influenciar"],
    ),
    EvalCase(
        question="Como o cérebro humano toma decisões de compra?",
        expect_refusal=False,
        expected_keywords=["cérebro", "lógica"],
    ),
    EvalCase(
        question="O que é o efeito GAP de Danny Iny?",
        expect_refusal=False,
        expected_keywords=["gap", "danny", "contemplação"],
    ),
    EvalCase(
        question="Qual a capital da França?",
        expect_refusal=True,
        expected_keywords=[],
    ),
    EvalCase(
        question="Como funciona um motor a combustão?",
        expect_refusal=True,
        expected_keywords=[],
    ),
]

REFUSAL_PHRASES = [
    "não encontrei",
    "não há informação",
    "não consta",
    "fora do escopo",
    "não tenho informação",
    "não foi possível",
]


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def run_evals() -> None:
    results = []
    passed = 0
    failed = 0

    print("\n=== EVALS - Doc Query RAG ===\n")

    for i, case in enumerate(EVAL_CASES, start=1):
        t_start = time.perf_counter()

        chunks = buscar_chunks(case.question, top_k=3)
        resultado = gerar_resposta(case.question, chunks)
        answer = resultado["resposta"]
        tokens = resultado["tokens_usados"]

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        refusal_detected = is_refusal(answer)

        if case.expect_refusal:
            ok = refusal_detected
            failure_reason = "esperava recusa mas o modelo respondeu" if not ok else ""
        else:
            missing = [kw for kw in case.expected_keywords if kw.lower() not in answer.lower()]
            ok = not missing and not refusal_detected
            failure_reason = f"keywords ausentes: {missing}" if missing else (
                "recusou mas deveria responder" if refusal_detected else ""
            )

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"[{i}] {status} | {latency_ms}ms | {tokens} tokens")
        print(f"     Q: {case.question}")
        if not ok:
            print(f"     MOTIVO: {failure_reason}")
        print(f"     A: {answer[:120]}...")
        print()

        results.append({
            "question": case.question,
            "passed": ok,
            "latency_ms": latency_ms,
            "tokens_used": tokens,
            "expect_refusal": case.expect_refusal,
            "failure_reason": failure_reason,
        })

    total = len(EVAL_CASES)
    score = round((passed / total) * 100, 1)
    avg_latency = round(sum(r["latency_ms"] for r in results) / total, 2)
    total_tokens = sum(r["tokens_used"] for r in results)

    print("=" * 40)
    print(f"RESULTADO: {passed}/{total} passou ({score}%)")
    print(f"Latência média: {avg_latency}ms")
    print(f"Tokens consumidos: {total_tokens}")
    print("=" * 40)

    os.makedirs("tests/results", exist_ok=True)
    output_path = f"tests/results/eval_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "score_pct": score,
            "passed": passed,
            "failed": failed,
            "avg_latency_ms": avg_latency,
            "total_tokens": total_tokens,
            "cases": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    run_evals()