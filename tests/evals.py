"""
RAG Evaluation Suite — LLM-as-a-Judge
======================================
Avalia a qualidade do pipeline RAG usando métricas de Faithfulness e
Answer Relevance, pontuadas por um LLM juiz (Groq / LLaMA 3.3 70B)
com saída JSON estruturada.

Gera:
  - JSON detalhado em tests/results/eval_<timestamp>.json
  - Dashboard HTML interativo em tests/results/report.html
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fix Windows console encoding (cp1252 doesn't support unicode emojis)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from app.retrieval import buscar_chunks
from app.generation import gerar_resposta

load_dotenv()
logging.disable(logging.CRITICAL)

# ── Groq judge client ──────────────────────────────────────────────
judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.3-70b-versatile"


# ── Data structures ────────────────────────────────────────────────
@dataclass
class EvalCase:
    question: str
    expect_refusal: bool = False
    expected_keywords: list[str] = field(default_factory=list)
    ground_truth: str = ""  # referência esperada para comparação


@dataclass
class JudgeScore:
    faithfulness: int = 0
    faithfulness_reason: str = ""
    relevance: int = 0
    relevance_reason: str = ""


@dataclass
class EvalResult:
    question: str
    answer: str
    context_chunks: list[str]
    ground_truth: str
    expect_refusal: bool
    refusal_detected: bool
    keyword_pass: bool
    keyword_missing: list[str]
    judge: JudgeScore
    latency_ms: float
    tokens_used: int
    judge_tokens_used: int
    passed: bool


# ── Eval cases ─────────────────────────────────────────────────────
EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="O que é engenharia da persuasão?",
        expect_refusal=False,
        expected_keywords=["persuasão"],
        ground_truth="Engenharia da persuasão é o uso sistemático de "
        "técnicas psicológicas para influenciar decisões e comportamentos.",
    ),
    EvalCase(
        question="Como o cérebro humano toma decisões de compra?",
        expect_refusal=False,
        expected_keywords=["cérebro"],
        ground_truth="O cérebro combina processos emocionais e lógicos para "
        "avaliar riscos e recompensas antes de tomar uma decisão de compra.",
    ),
    EvalCase(
        question="O que é o efeito GAP de Danny Iny?",
        expect_refusal=False,
        expected_keywords=["gap", "danny"],
        ground_truth="O efeito GAP de Danny Iny descreve a distância entre "
        "a situação atual do leitor e a situação desejada, usada como "
        "técnica de engajamento e contemplação.",
    ),
    EvalCase(
        question="Qual a capital da França?",
        expect_refusal=True,
        expected_keywords=[],
        ground_truth="",
    ),
    EvalCase(
        question="Como funciona um motor a combustão?",
        expect_refusal=True,
        expected_keywords=[],
        ground_truth="",
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


# ── LLM-as-a-Judge ────────────────────────────────────────────────
def judge_answer(question: str, answer: str, context: str, ground_truth: str) -> tuple[JudgeScore, int]:
    """
    Uses an LLM judge to evaluate Faithfulness and Answer Relevance.
    Returns (JudgeScore, tokens_used_by_judge).
    """
    prompt = f"""You are an impartial RAG evaluation judge. Evaluate the ANSWER based on the CONTEXT and QUESTION.

Score each metric from 1 (terrible) to 5 (excellent).

METRICS:
1. **Faithfulness**: Does the answer use ONLY information from the CONTEXT? 
   - 5 = Entirely faithful, every claim is grounded in context
   - 3 = Mostly faithful but adds minor unsupported claims
   - 1 = Hallucinated or fabricated information not in context

2. **Answer Relevance**: Does the answer directly address the QUESTION?
   - 5 = Perfectly addresses the question with specific detail
   - 3 = Partially addresses the question
   - 1 = Completely off-topic or non-responsive

CONTEXT:
{context}

QUESTION:
{question}

GROUND TRUTH (reference, for comparison only):
{ground_truth if ground_truth else "N/A — this is an out-of-scope question, the system should refuse to answer."}

ANSWER:
{answer}

Respond ONLY with valid JSON in this exact format, no extra text:
{{
  "faithfulness": <int 1-5>,
  "faithfulness_reason": "<one sentence in Portuguese>",
  "relevance": <int 1-5>,
  "relevance_reason": "<one sentence in Portuguese>"
}}"""

    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        tokens = response.usage.total_tokens
        raw = response.choices[0].message.content
        data = json.loads(raw)
        score = JudgeScore(
            faithfulness=int(data.get("faithfulness", 0)),
            faithfulness_reason=data.get("faithfulness_reason", ""),
            relevance=int(data.get("relevance", 0)),
            relevance_reason=data.get("relevance_reason", ""),
        )
        return score, tokens
    except Exception as e:
        return JudgeScore(
            faithfulness=0,
            faithfulness_reason=f"Erro no juiz: {e}",
            relevance=0,
            relevance_reason=f"Erro no juiz: {e}",
        ), 0


def is_refusal(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


# ── Evaluation runner ──────────────────────────────────────────────
def run_evals() -> list[EvalResult]:
    results: list[EvalResult] = []

    print("\n" + "=" * 60)
    print("  RAG EVALUATION SUITE — LLM-as-a-Judge")
    print("=" * 60 + "\n")

    for i, case in enumerate(EVAL_CASES, start=1):
        t_start = time.perf_counter()

        # Step 1: Retrieve + Generate
        chunks = buscar_chunks(case.question, top_k=3)
        resultado = gerar_resposta(case.question, chunks)
        answer = resultado["resposta"]
        tokens = resultado["tokens_usados"]
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Step 2: Deterministic checks
        refusal_detected = is_refusal(answer)
        missing_kw = [kw for kw in case.expected_keywords if kw.lower() not in answer.lower()]
        keyword_pass = len(missing_kw) == 0

        # Step 3: LLM Judge
        context_str = "\n\n".join(chunks)
        judge_score, judge_tokens = judge_answer(
            case.question, answer, context_str, case.ground_truth
        )

        # Step 4: Final verdict
        if case.expect_refusal:
            passed = refusal_detected
        else:
            passed = (
                keyword_pass
                and not refusal_detected
                and judge_score.faithfulness >= 4
                and judge_score.relevance >= 4
            )

        result = EvalResult(
            question=case.question,
            answer=answer,
            context_chunks=chunks,
            ground_truth=case.ground_truth,
            expect_refusal=case.expect_refusal,
            refusal_detected=refusal_detected,
            keyword_pass=keyword_pass,
            keyword_missing=missing_kw,
            judge=judge_score,
            latency_ms=latency_ms,
            tokens_used=tokens,
            judge_tokens_used=judge_tokens,
            passed=passed,
        )
        results.append(result)

        # Console output
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  [{i}] {status} | {latency_ms}ms | {tokens} tok")
        print(f"       Q: {case.question}")
        print(f"       Faithfulness: {judge_score.faithfulness}/5 — {judge_score.faithfulness_reason}")
        print(f"       Relevance:    {judge_score.relevance}/5 — {judge_score.relevance_reason}")
        if not passed and not case.expect_refusal:
            reasons = []
            if missing_kw:
                reasons.append(f"keywords ausentes: {missing_kw}")
            if refusal_detected:
                reasons.append("recusou mas deveria responder")
            if judge_score.faithfulness < 4:
                reasons.append(f"faithfulness baixo ({judge_score.faithfulness}/5)")
            if judge_score.relevance < 4:
                reasons.append(f"relevance baixo ({judge_score.relevance}/5)")
            print(f"       ⚠ {'; '.join(reasons)}")
        print(f"       A: {answer[:140]}...")
        print()

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count
    score_pct = round((passed_count / total) * 100, 1)
    avg_latency = round(sum(r.latency_ms for r in results) / total, 2)
    avg_faithfulness = round(
        sum(r.judge.faithfulness for r in results if not r.expect_refusal)
        / max(1, sum(1 for r in results if not r.expect_refusal)),
        2,
    )
    avg_relevance = round(
        sum(r.judge.relevance for r in results if not r.expect_refusal)
        / max(1, sum(1 for r in results if not r.expect_refusal)),
        2,
    )
    total_tokens = sum(r.tokens_used for r in results)
    total_judge_tokens = sum(r.judge_tokens_used for r in results)

    print("=" * 60)
    print(f"  RESULTADO: {passed_count}/{total} ({score_pct}%)")
    print(f"  Latência média: {avg_latency}ms")
    print(f"  Faithfulness médio: {avg_faithfulness}/5")
    print(f"  Relevance médio: {avg_relevance}/5")
    print(f"  Tokens (geração): {total_tokens}")
    print(f"  Tokens (juiz): {total_judge_tokens}")
    print("=" * 60)

    # Save JSON
    os.makedirs("tests/results", exist_ok=True)
    timestamp = int(time.time())
    json_path = f"tests/results/eval_{timestamp}.json"

    json_data = {
        "timestamp": timestamp,
        "score_pct": score_pct,
        "passed": passed_count,
        "failed": failed_count,
        "avg_latency_ms": avg_latency,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "total_tokens": total_tokens,
        "total_judge_tokens": total_judge_tokens,
        "cases": [_result_to_dict(r) for r in results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON salvo em: {json_path}")

    # Generate HTML report
    html_path = "tests/results/report.html"
    generate_html_report(json_data, html_path)
    print(f"  HTML report salvo em: {html_path}\n")

    return results


def _result_to_dict(r: EvalResult) -> dict:
    d = asdict(r)
    # Truncate context chunks to keep the JSON manageable
    d["context_chunks"] = [c[:200] + "..." if len(c) > 200 else c for c in d["context_chunks"]]
    return d


# ── HTML Report Generator ─────────────────────────────────────────
def generate_html_report(data: dict, output_path: str) -> None:
    """Generates a premium dark-mode HTML dashboard for the eval results."""

    cases = data["cases"]
    total = len(cases)
    passed = data["passed"]
    score_pct = data["score_pct"]

    # Build case rows
    case_rows = ""
    for i, c in enumerate(cases, 1):
        status_class = "pass" if c["passed"] else "fail"
        status_badge = "PASS" if c["passed"] else "FAIL"
        faith = c["judge"]["faithfulness"]
        relev = c["judge"]["relevance"]
        faith_reason = c["judge"]["faithfulness_reason"]
        relev_reason = c["judge"]["relevance_reason"]

        # Context preview
        ctx_preview = " | ".join(
            [chunk[:80] + "…" for chunk in c.get("context_chunks", [])]
        )

        case_rows += f"""
        <div class="case-card {status_class}">
            <div class="case-header">
                <div class="case-number">#{i}</div>
                <span class="badge badge-{status_class}">{status_badge}</span>
                <div class="case-meta">
                    <span class="meta-pill"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> {c['latency_ms']}ms</span>
                    <span class="meta-pill"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg> {c['tokens_used']} tok</span>
                </div>
            </div>

            <p class="case-question">{c['question']}</p>

            <div class="scores-row">
                <div class="score-item">
                    <div class="score-label">Faithfulness</div>
                    <div class="score-bar-container">
                        <div class="score-bar score-bar-faith" style="width: {faith * 20}%"></div>
                    </div>
                    <span class="score-value">{faith}/5</span>
                </div>
                <div class="score-item">
                    <div class="score-label">Relevance</div>
                    <div class="score-bar-container">
                        <div class="score-bar score-bar-relev" style="width: {relev * 20}%"></div>
                    </div>
                    <span class="score-value">{relev}/5</span>
                </div>
            </div>

            <details class="case-details">
                <summary>Ver detalhes</summary>
                <div class="detail-grid">
                    <div class="detail-block">
                        <h4>💬 Resposta</h4>
                        <p>{c['answer'][:500]}</p>
                    </div>
                    <div class="detail-block">
                        <h4>🎯 Ground Truth</h4>
                        <p>{c.get('ground_truth', 'N/A') or '<em>Pergunta fora de escopo — espera-se recusa.</em>'}</p>
                    </div>
                    <div class="detail-block">
                        <h4>📎 Contexto Recuperado</h4>
                        <p class="context-preview">{ctx_preview or 'N/A'}</p>
                    </div>
                    <div class="detail-block">
                        <h4>🧠 Raciocínio do Juiz</h4>
                        <p><strong>Faithfulness:</strong> {faith_reason}</p>
                        <p><strong>Relevance:</strong> {relev_reason}</p>
                    </div>
                </div>
            </details>
        </div>"""

    # Score color
    if score_pct >= 80:
        score_color = "#22c55e"
        score_glow = "rgba(34, 197, 94, 0.3)"
    elif score_pct >= 60:
        score_color = "#eab308"
        score_glow = "rgba(234, 179, 8, 0.3)"
    else:
        score_color = "#ef4444"
        score_glow = "rgba(239, 68, 68, 0.3)"

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Eval Report — Doc Query API</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg-primary: #0a0a0f;
            --bg-card: #12121a;
            --bg-card-hover: #1a1a26;
            --bg-surface: #1e1e2e;
            --border: #2a2a3e;
            --text-primary: #e8e8f0;
            --text-secondary: #8888a8;
            --text-muted: #5a5a78;
            --accent-green: #22c55e;
            --accent-green-dim: rgba(34, 197, 94, 0.15);
            --accent-red: #ef4444;
            --accent-red-dim: rgba(239, 68, 68, 0.15);
            --accent-blue: #6366f1;
            --accent-blue-dim: rgba(99, 102, 241, 0.15);
            --accent-amber: #f59e0b;
            --accent-amber-dim: rgba(245, 158, 11, 0.15);
            --accent-purple: #a855f7;
            --accent-purple-dim: rgba(168, 85, 247, 0.15);
            --radius: 12px;
            --radius-sm: 8px;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 24px;
        }}

        /* ── Header ─────────────────────────── */
        .header {{
            text-align: center;
            margin-bottom: 48px;
        }}

        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--accent-blue-dim);
            color: var(--accent-blue);
            padding: 6px 16px;
            border-radius: 100px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-bottom: 16px;
        }}

        .header h1 {{
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, #e8e8f0, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}

        .header p {{
            color: var(--text-secondary);
            font-size: 15px;
        }}

        /* ── KPI Cards ──────────────────────── */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 40px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            text-align: center;
            transition: all 0.2s ease;
        }}

        .kpi-card:hover {{
            background: var(--bg-card-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .kpi-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
        }}

        .kpi-sub {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* ── Score Ring ─────────────────────── */
        .score-ring-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 40px;
        }}

        .score-ring {{
            position: relative;
            width: 180px;
            height: 180px;
        }}

        .score-ring svg {{
            transform: rotate(-90deg);
        }}

        .score-ring-bg {{
            fill: none;
            stroke: var(--border);
            stroke-width: 10;
        }}

        .score-ring-fill {{
            fill: none;
            stroke: {score_color};
            stroke-width: 10;
            stroke-linecap: round;
            stroke-dasharray: {score_pct * 4.71} 471;
            filter: drop-shadow(0 0 8px {score_glow});
            transition: stroke-dasharray 1s ease;
        }}

        .score-ring-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }}

        .score-ring-text .pct {{
            font-size: 42px;
            font-weight: 800;
            color: {score_color};
        }}

        .score-ring-text .label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        /* ── Section Headers ────────────────── */
        .section-header {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* ── Case Cards ─────────────────────── */
        .case-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }}

        .case-card:hover {{
            background: var(--bg-card-hover);
        }}

        .case-card.pass {{
            border-left: 3px solid var(--accent-green);
        }}

        .case-card.fail {{
            border-left: 3px solid var(--accent-red);
        }}

        .case-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }}

        .case-number {{
            font-weight: 700;
            font-size: 14px;
            color: var(--text-muted);
            min-width: 28px;
        }}

        .badge {{
            padding: 4px 12px;
            border-radius: 100px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .badge-pass {{
            background: var(--accent-green-dim);
            color: var(--accent-green);
        }}

        .badge-fail {{
            background: var(--accent-red-dim);
            color: var(--accent-red);
        }}

        .case-meta {{
            display: flex;
            gap: 8px;
            margin-left: auto;
            flex-wrap: wrap;
        }}

        .meta-pill {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-surface);
            padding: 4px 10px;
            border-radius: 100px;
            font-size: 12px;
            color: var(--text-secondary);
        }}

        .case-question {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }}

        /* ── Score Bars ─────────────────────── */
        .scores-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 12px;
        }}

        .score-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .score-label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            min-width: 90px;
        }}

        .score-bar-container {{
            flex: 1;
            height: 8px;
            background: var(--bg-surface);
            border-radius: 100px;
            overflow: hidden;
        }}

        .score-bar {{
            height: 100%;
            border-radius: 100px;
            transition: width 0.6s ease;
        }}

        .score-bar-faith {{
            background: linear-gradient(90deg, #6366f1, #a855f7);
        }}

        .score-bar-relev {{
            background: linear-gradient(90deg, #22c55e, #06b6d4);
        }}

        .score-value {{
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            min-width: 30px;
        }}

        /* ── Details Panel ──────────────────── */
        .case-details {{
            margin-top: 12px;
        }}

        .case-details summary {{
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            color: var(--accent-blue);
            padding: 8px 0;
            user-select: none;
            list-style: none;
        }}

        .case-details summary::-webkit-details-marker {{ display: none; }}

        .case-details summary::before {{
            content: '▸ ';
        }}

        .case-details[open] summary::before {{
            content: '▾ ';
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 12px;
        }}

        .detail-block {{
            background: var(--bg-surface);
            border-radius: var(--radius-sm);
            padding: 16px;
        }}

        .detail-block h4 {{
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-secondary);
        }}

        .detail-block p {{
            font-size: 13px;
            color: var(--text-primary);
            line-height: 1.5;
            word-break: break-word;
        }}

        .context-preview {{
            font-family: 'Courier New', monospace;
            font-size: 11px !important;
            color: var(--text-muted) !important;
        }}

        /* ── Footer ─────────────────────────── */
        .footer {{
            text-align: center;
            padding: 32px 0;
            border-top: 1px solid var(--border);
            margin-top: 40px;
            color: var(--text-muted);
            font-size: 12px;
        }}

        .footer a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}

        /* ── Responsive ─────────────────────── */
        @media (max-width: 768px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .detail-grid {{ grid-template-columns: 1fr; }}
            .scores-row {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 28px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-badge">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/></svg>
                LLM-as-a-Judge Evaluation
            </div>
            <h1>RAG Evaluation Report</h1>
            <p>Doc Query API — Pipeline de Retrieval-Augmented Generation</p>
        </div>

        <!-- Score Ring -->
        <div class="score-ring-container">
            <div class="score-ring">
                <svg width="180" height="180" viewBox="0 0 180 180">
                    <circle class="score-ring-bg" cx="90" cy="90" r="75"/>
                    <circle class="score-ring-fill" cx="90" cy="90" r="75"/>
                </svg>
                <div class="score-ring-text">
                    <div class="pct">{score_pct}%</div>
                    <div class="label">Score Geral</div>
                </div>
            </div>
        </div>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Casos Avaliados</div>
                <div class="kpi-value" style="color: var(--accent-blue)">{total}</div>
                <div class="kpi-sub">{passed} pass · {data['failed']} fail</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Faithfulness Médio</div>
                <div class="kpi-value" style="color: var(--accent-purple)">{data['avg_faithfulness']}</div>
                <div class="kpi-sub">de 5.0</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Relevance Médio</div>
                <div class="kpi-value" style="color: var(--accent-green)">{data['avg_relevance']}</div>
                <div class="kpi-sub">de 5.0</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latência Média</div>
                <div class="kpi-value" style="color: var(--accent-amber)">{data['avg_latency_ms']}</div>
                <div class="kpi-sub">milissegundos</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tokens (Geração)</div>
                <div class="kpi-value" style="color: var(--text-primary)">{data['total_tokens']}</div>
                <div class="kpi-sub">consumidos</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Tokens (Juiz)</div>
                <div class="kpi-value" style="color: var(--text-primary)">{data['total_judge_tokens']}</div>
                <div class="kpi-sub">avaliação LLM</div>
            </div>
        </div>

        <!-- Case Results -->
        <div class="section-header">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            Resultados por Caso
        </div>

        {case_rows}

        <!-- Footer -->
        <div class="footer">
            Gerado automaticamente por <strong>RAG Eval Suite</strong> · LLM-as-a-Judge via Groq (LLaMA 3.3 70B)<br>
            <a href="https://github.com/leonardo-albrecht/doc-query-api" target="_blank">github.com/leonardo-albrecht/doc-query-api</a>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evals()