"""#T-laya-archdiff — ¿el presupuesto 48/192 de Laya explica su techo en BANKING77?

Sólo CPU y sólo tokenizadores. Reproduce el recorte de opciones de
`TMP/laya/laya/common.py::build_sequence` (Apache-2.0, importado desde el
clon, NO copiado) sobre las 77 etiquetas de BANKING77 tal y como las sirve
`TMP/laya/research/scripts/bench_apps.py` (nombre con `_` → espacio, sin
descripción), y calcula cuántas etiquetas quedan indistinguibles tras el
recorte y el techo de acierto que eso impone. Después mide qué ocupa la
misma pregunta en NUESTRO scorer (`model/ce_scorer.py::length_report`).

Comando (desde la raíz del repo):
    PYTHONPATH=.:TMP/laya .venv-train/bin/python \
        artifacts/gates/T-laya-archdiff/token_budget.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TMP", "laya"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402

from eval.fullspace import full_samples  # noqa: E402
from laya.common import build_sequence, render_options  # noqa: E402
from model import ce_scorer as CE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_budget.json")
INSTRUCTIONS = "Which banking intent does `message` express?"   # bench_apps.py:114
LAYA_CFGS = {  # README.md:983-984; agent.py:655-656 (defaults 512/192)
    "laya (ModernBERT-large, 512/192)": (512, 192),
    "laya-typed-decisions (ModernBERT-large, 1024/256)": (1024, 256),
    "README remedy (1024/512)": (1024, 512),
}


def laya_budget(tok, labels: list[str], state_text: str, max_len: int,
                head_max_len: int, freq: collections.Counter) -> dict:
    q = {"t": "choice", "ins": INSTRUCTIONS, "crit": {lab: None for lab in labels}}
    opts = render_options(q)
    raw_lens = [len(tok(" " + o, add_special_tokens=False)["input_ids"]) for o in opts]
    ids, markers = build_sequence(tok, {"message": state_text}, q, max_len, head_max_len)
    assert len(markers) == len(labels), (len(markers), len(labels))
    # tramo de cada opción: desde su [MASK] hasta el siguiente marcador / el [SEP]
    sep = tok.sep_token_id
    end_of_opts = markers[-1]
    while ids[end_of_opts] != sep:
        end_of_opts += 1
    bounds = markers + [end_of_opts]
    per_opt = []
    for i in range(len(markers)):
        span = ids[bounds[i]:bounds[i + 1]]
        assert span[0] == tok.mask_token_id
        per_opt.append(span[1:])                    # sin el [MASK]
    seen_texts = [tok.decode(s) for s in per_opt]
    groups = collections.defaultdict(list)
    for lab, seen in zip(labels, seen_texts):
        groups[seen].append(lab)
    collided = {k: v for k, v in groups.items() if len(v) > 1}
    n_labels = len(labels)
    # techo: dentro de un grupo indistinguible sólo se puede acertar 1 de g
    ceiling_uniform = len(groups) / n_labels
    total = sum(freq.values())
    ceiling_freq = sum(max(freq[lab] for lab in v) for v in groups.values()) / total
    head_ids_len = markers[0] - 1                   # tokens de instrucción tras [CLS]
    state_room = max_len - (end_of_opts + 1) - 1    # hueco que queda para el estado
    state_len = len(tok(json.dumps({"message": state_text}), add_special_tokens=False)["input_ids"])
    return {
        "max_len": max_len, "head_max_len": head_max_len,
        "n_options": n_labels,
        "option_tokens_raw": {"min": min(raw_lens), "median": sorted(raw_lens)[len(raw_lens) // 2],
                              "max": max(raw_lens), "sum": sum(raw_lens)},
        "per_option_after_cap": {"min": min(len(s) for s in per_opt),
                                 "max": max(len(s) for s in per_opt),
                                 "sum_with_masks": sum(len(s) + 1 for s in per_opt)},
        "budget_rule": "common.py:127-131 — si head_max_len - Σ(1+len) < 16, cada opción se recorta a max(4, (head_max_len-16)//K) tokens INCLUIDO su [MASK]",
        "per_option_cap_incl_mask": max(4, (head_max_len - 16) // n_labels)
        if head_max_len - sum(n + 1 for n in raw_lens) < 16 else None,
        "instruction_tokens_kept": head_ids_len,
        "state_room_tokens": state_room,
        "state_tokens_first_row": state_len,
        "labels_intact": sum(1 for r, s in zip(raw_lens, per_opt) if len(s) == r),
        "distinct_after_cap": len(groups),
        "collision_groups": len(collided),
        "labels_in_a_collision": sum(len(v) for v in collided.values()),
        "largest_group": max((len(v) for v in groups.values()), default=1),
        "collisions": {k.strip(): v for k, v in sorted(collided.items())},
        "accuracy_ceiling_uniform_labels": round(ceiling_uniform, 4),
        "accuracy_ceiling_train_label_freq": round(ceiling_freq, 4),
        "ceiling_reading": ("un grupo de g etiquetas con el mismo texto recortado sólo "
                            "puede acertarse 1/g de las veces; el techo suma el mejor "
                            "miembro de cada grupo. Es un techo del FORMATO, no del modelo"),
    }


def main() -> int:
    t0 = time.time()
    # El espacio de etiquetas y la frecuencia por etiqueta salen del split de
    # TRAIN (corte de desarrollo). El test de BANKING77 es RESERVADO (eval/cuts.py
    # R7) y aquí no se lee: el espacio de etiquetas es el mismo.
    samples = full_samples("banking77", limit=100000, split="train")
    # bench_apps.py:115-116: `label_text.replace("_", " ")`, sin descripción
    labels = [o["text"].replace("_", " ") for o in samples[0].options]
    assert len(labels) == 77, len(labels)
    freq = collections.Counter(s.options[s.gold_index]["text"].replace("_", " ") for s in samples)
    state_text = samples[0].state

    mb_path = os.path.join(ROOT, "artifacts", "weights", "modernbert-base")
    laya_tok = AutoTokenizer.from_pretrained(mb_path)
    out = {
        "task": "T-laya-archdiff",
        "artifact": "token-budget",
        "question": "¿explica el presupuesto de tokens por opción de Laya su 0,425 en BANKING77 K=77?",
        "label_space": {"dataset": "banking77", "split_read": "train (dev cut; el test reservado no se abre)",
                        "rows_read": len(samples), "n_labels": len(labels),
                        "rendering": "bench_apps.py:115 — nombre con `_` → espacio, sin descripción"},
        "laya_tokenizer": {"path": mb_path, "vocab_size": laya_tok.vocab_size,
                           "note": ("tokenizador de ModernBERT-base verificado en artifacts/weights; "
                                    "ModernBERT-large (el encoder de `laya`) comparte vocabulario y "
                                    "tokenizer.json — NO verificado aquí contra el checkpoint de Laya, "
                                    "que no está descargado. mmBERT (laya-multilingual) NO medido")},
        "laya": {name: laya_budget(laya_tok, labels, state_text, ml, hl, freq)
                 for name, (ml, hl) in LAYA_CFGS.items()},
    }

    # Nuestro scorer: 77 pares independientes, la opción NUNCA se recorta
    ours = {}
    for wid in ("minilmv2-l6-mnli-xnli", "modernbert-zeroshot-v2"):
        path = os.path.join(ROOT, "artifacts", "weights", wid)
        if not os.path.exists(os.path.join(path, "tokenizer.json")):
            ours[wid] = {"measured": False, "why": "tokenizer no descargado"}
            continue
        tok = AutoTokenizer.from_pretrained(path)
        cands = tuple(CE.Candidate(f"c{i}", t) for i, t in enumerate(labels))
        per_row = []
        for s in samples[:200]:
            dec = CE.Decision(state=s.state, question=s.question, candidates=cands,
                              family=CE.DESCRIPTION, lang="en")
            pairs = CE.render_pairs(dec)
            rep = CE.length_report(tok, pairs, CE.DEFAULT_MAX_LENGTH)
            per_row.append(rep)
        lens_hyp = [len(tok(CE.HYPOTHESIS["en"].format(option=t), add_special_tokens=False)["input_ids"])
                    for t in labels]
        lens_opt = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in labels]
        dec_ctx = CE.Decision(state=samples[0].state, question=samples[0].question,
                              candidates=cands, family=CE.COMPARISON, lang="en")
        rep_ctx = CE.length_report(tok, CE.render_pairs(dec_ctx), CE.DEFAULT_MAX_LENGTH)
        ours[wid] = {
            "measured": True,
            "rows_measured": len(per_row),
            "pairs_per_row": 77,
            "max_length_per_pair": CE.DEFAULT_MAX_LENGTH,
            "truncation": CE.TRUNCATION_STRATEGY + " — la hipótesis (opción) no se recorta nunca",
            "option_tokens": {"min": min(lens_opt), "max": max(lens_opt),
                              "median": sorted(lens_opt)[len(lens_opt) // 2]},
            "hypothesis_tokens": {"min": min(lens_hyp), "max": max(lens_hyp)},
            "max_pair_tokens_over_rows": max(r["max_tokens"] for r in per_row),
            "rows_with_truncation": sum(1 for r in per_row if r["n_truncated"]),
            "tokens_read_per_decision_approx": "77 × (estado + pregunta + hipótesis); cada opción entera",
            "with_comparative_context_max_tokens": rep_ctx["max_tokens"],
            "with_comparative_context_truncated": rep_ctx["n_truncated"],
            "reading": ("el presupuesto de Laya (≈3 tokens de texto por etiqueta) no existe en el "
                        "scorer: cada etiqueta entra entera en su propio par, a cambio de 77 "
                        "pasadas por decisión"),
        }
    out["ours"] = ours
    # Ritmo medido del scorer NLI en CPU sobre pares reales de BANKING77 (K=77),
    # para costar una pasada completa: 77 × 3 080 = 237 160 pares en el test.
    try:
        scorer = CE.NliPairScorer("minilmv2-l6-mnli-xnli", device="cpu")
        cands = tuple(CE.Candidate(f"c{i}", t) for i, t in enumerate(labels))
        decs = [CE.Decision(state=s.state, question=s.question, candidates=cands,
                            family=CE.DESCRIPTION, lang="en") for s in samples[:4]]
        pairs, _ = CE.flatten_pairs(decs)
        scorer.score_pairs(pairs[:16])                       # calentamiento
        t1 = time.time()
        scorer.score_pairs(pairs)
        dt = time.time() - t1
        rate = len(pairs) / dt
        out["throughput_cpu"] = {
            "scorer": scorer.describe()["id"], "device": "cpu",
            "pairs_timed": len(pairs), "seconds": round(dt, 2),
            "pairs_per_second": round(rate, 1),
            "banking77_test_pairs": 77 * 3080,
            "estimated_minutes_full_test_cpu": round(77 * 3080 / rate / 60, 1),
            "reading": "estimación lineal desde una muestra de 4 filas; no es una medida del test",
        }
    except Exception as exc:  # pesos no descargados u otra causa: se declara, no se inventa
        out["throughput_cpu"] = {"measured": False, "why": repr(exc)[:200]}
    out["seconds"] = round(time.time() - t0, 1)
    out["command"] = ("PYTHONPATH=.:TMP/laya .venv-train/bin/python "
                      "artifacts/gates/T-laya-archdiff/token_budget.py")
    out["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
