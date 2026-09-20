"""Email-triage intentional dataset v1 (#T-teacher-intent).

Closed 4-state decision: archivar / responder / urgente / spam, with
per-state weights in the structured answer. Deterministic by seed,
universal-schema (choice) JSONL.

Usage:
  .venv-train/bin/python -m data.intent.email_triage --n 5000
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.schema import Example, Option, Question, validate  # noqa: E402

STATES = ("archivar", "responder", "urgente", "spam")
LABELS = {
    "archivar": "Archivar",
    "responder": "Responder",
    "urgente": "Responder urgente",
    "spam": "Spam",
}

# Per-state weights over the 4 options (gold is always the max).
WEIGHTS: dict[str, dict[str, float]] = {
    "archivar": {"archivar": 1.0, "responder": 0.15, "urgente": 0.05, "spam": 0.05},
    "responder": {"responder": 1.0, "urgente": 0.25, "archivar": 0.15, "spam": 0.0},
    "urgente": {"urgente": 1.0, "responder": 0.3, "archivar": 0.05, "spam": 0.0},
    "spam": {"spam": 1.0, "archivar": 0.2, "responder": 0.0, "urgente": 0.0},
}

NOMBRES = ["Marta", "Jordi", "Anna", "Pau", "Laia", "Marc", "Núria", "Oriol",
           "Carla", "Sergi", "Júlia", "Pol", "Aina", "Roger", "Clara"]
EMPRESAS = ["Endesa", "CaixaBank", "Telefónica", "Iberia", "Renfe", "SegurCaixa",
            "Naturgy", "Vueling", "BBVA", "Correos"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]

# (state, subject-template, body-template)
TEMPLATES: list[tuple[str, str, str]] = [
    ("archivar", "Factura {empresa} nº {num}",
     "{nombre}, te adjuntamos la factura nº {num} de {empresa} por {importe} € "
     "correspondiente a {mes}. No requiere ninguna acción, es solo para tus archivos."),
    ("archivar", "Confirmación de entrega pedido {num}",
     "Tu pedido {num} ha sido entregado correctamente. Guarda este correo como "
     "comprobante. Gracias por tu compra, {nombre}."),
    ("archivar", "Resumen semanal del equipo",
     "Hola {nombre}, aquí tienes el resumen de la semana: 3 reuniones, 2 entregas "
     "y ningún bloqueo. Léelo cuando puedas, sin prisa."),
    ("archivar", "Newsletter mensual {empresa}",
     "Las novedades de {mes} de {empresa}: nuevos horarios, consejos de ahorro y "
     "una entrevista. Date de baja cuando quieras, no es urgente."),
    ("archivar", "Recibo del parking {mes}",
     "Adjuntamos el recibo del parking de {mes} por {importe} €. Solo archivo, "
     "el cargo ya está domiciliado."),
    ("archivar", "Acta de la reunión del {dia}",
     "Hola {nombre}, adjunto el acta de la reunión del {dia} para archivo. "
     "Sin acciones pendientes para ti."),
    ("responder", "¿Puedes revisar el informe antes del {dia}?",
     "Hola {nombre}, ¿podrías revisar el informe adjunto y decirme qué te parece "
     "antes del {dia}? Necesito tu opinión sobre el apartado 3."),
    ("responder", "Propuesta de colaboración con {empresa}",
     "{nombre}, desde {empresa} nos proponen colaborar en el proyecto de {mes}. "
     "¿Te interesa? Respóndeles esta semana si quieres participar."),
    ("responder", "Duda sobre tu reserva {num}",
     "Hola {nombre}, tenemos una duda sobre tu reserva {num}: ¿prefieres la mañana "
     "o la tarde del {dia}? Confírmanos por correo, por favor."),
    ("responder", "Invitación: boda de {nombre2} en {mes}",
     "{nombre}, ¡nos casamos! Será en {mes} y nos encantaría contar contigo. "
     "¿Puedes confirmarnos asistencia antes del {dia}?"),
    ("responder", "Presupuesto solicitado nº {num}",
     "{nombre}, aquí tienes el presupuesto nº {num} por {importe} €. Si te encaja, "
     "respóndenos y lo ponemos en marcha esta semana."),
    ("responder", "Entrevista de trabajo en {empresa}",
     "Hola {nombre}, tu perfil nos encaja para el puesto. ¿Podrías venir a una "
     "entrevista el {dia}? Confirma por correo tu disponibilidad."),
    ("urgente", "URGENTE: tu tarjeta ha sido bloqueada",
     "{nombre}, hemos detectado un cargo sospechoso de {importe} € y hemos bloqueado "
     "tu tarjeta por seguridad. Llámanos HOY al número oficial para desbloquearla."),
    ("urgente", "Corte de suministro mañana {dia}",
     "AVISO URGENTE de {empresa}: mañana {dia} habrá un corte de suministro de 8h a "
     "14h. Toma precauciones hoy mismo, {nombre}."),
    ("urgente", "Tu vuelo sale en 3 horas - facturación",
     "{nombre}, tu vuelo despega en 3 horas y aún no has facturado. Hazlo AHORA o "
     "perderás el embarque. Localizador {num}."),
    ("urgente", "Fuga de agua en tu vivienda - actúa ya",
     "{nombre}, el vecino de abajo reporta una fuga que viene de tu piso. Ve hoy "
     "mismo o avisa al seguro, los daños crecen por horas."),
    ("urgente", "Plazo beca termina HOY a las 23:59",
     "{nombre}, el plazo de la beca termina HOY. Si no envías la solicitud antes de "
     "las 23:59 pierdes la convocatoria de {mes}."),
    ("urgente", "Resultado médico disponible - importante",
     "{nombre}, ya tienes disponible un resultado importante. Revísalo hoy y pide "
     "cita con tu médico esta misma semana."),
    ("spam", "GANASTE {importe}€!!! reclama YA tu premio",
     "FELICIDADES {nombre}!!! Has sido seleccionado para ganar {importe}€ GRATIS. "
     "Haz clic aquí y envía tus datos bancarios para recibirlo YA."),
    ("spam", "Préstamo pre-aprobado de {importe}€ sin nómina",
     "Hola {nombre}, tienes un PRÉSTAMO PRE-APROBADO de {importe}€ sin nómina ni "
     "aval. Solo por hoy. Entra y deja tu DNI y cuenta."),
    ("spam", "{empresa} regala 2 billetes - últimos 10 minutos",
     "¡¡{empresa} REGALA 2 billetes!! Solo quedan 10 MINUTOS {nombre}. Paga 1€ de "
     "gastos de envío con tu tarjeta para recibirlos."),
    ("spam", "Herencia de un familiar lejano: {importe}€",
     "Estimado {nombre}, un familiar lejano te dejó {importe}€. Para transferirlos "
     "necesitamos tus claves bancarias urgentemente. Abogado {nombre2}."),
    ("spam", "Tu cuenta será SUSPENDIDA en 24h - verifica",
     "{nombre}, tu cuenta será SUSPENDIDA en 24 horas por actividad extraña. "
     "Verifica tu contraseña y tarjeta en este enlace ahora mismo."),
    ("spam", "Trabaja desde casa: gana {importe}€/mes sin esfuerzo",
     "{nombre}, gana {importe}€ al mes desde casa sin esfuerzo. Solo reenvía paquetes "
     "y quédate comisión. Empieza hoy, sin contrato."),
]

BY_STATE: dict[str, list[tuple[str, str, str]]] = {}
for _s, _subj, _body in TEMPLATES:
    BY_STATE.setdefault(_s, []).append((_s, _subj, _body))


def _split(i: int) -> str:
    m = i % 10
    if m < 8:
        return "train"
    if m == 8:
        return "calibration"
    return "test"


def generate(n: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for i in range(n):
        state = STATES[rng.randrange(len(STATES))]
        _s, subj_t, body_t = rng.choice(BY_STATE[state])
        ctx = {
            "nombre": rng.choice(NOMBRES),
            "nombre2": rng.choice(NOMBRES),
            "empresa": rng.choice(EMPRESAS),
            "mes": rng.choice(MESES),
            "dia": rng.choice(["lunes", "martes", "miércoles", "jueves", "viernes"]),
            "num": f"{rng.randrange(10000, 99999)}",
            "importe": f"{rng.randrange(20, 9000)}",
        }
        text = f"Asunto: {subj_t.format(**ctx)}\n{body_t.format(**ctx)}"
        opts = [Option(id=s, text=LABELS[s]) for s in STATES]
        rng.shuffle(opts)
        w = WEIGHTS[state]
        tot = sum(w.values())
        out.append(Example(
            state=text,
            questions=[Question(
                id=f"email-triage-{seed}-{i}",
                kind="choice",
                options=opts,
                answer=state,
                teacher_conf=round(w[state] / tot, 3),
                weights=dict(w),
            )],
            split=_split(i),
        ))
    return out


def to_jsonl(rows: list[Example], path: Path) -> None:
    with open(path, "w") as f:
        for ex in rows:
            errs = validate(ex)
            if errs:
                raise ValueError(f"invalid example: {errs}")
            f.write(json.dumps(dataclasses.asdict(ex), ensure_ascii=False) + "\n")


# Forward-testing cases for the tester engine: easy -> hard, es/en.
FORWARD_CASES: list[str] = [
    "Te adjunto la factura de la luz de marzo, solo para tu archivo.",
    "GANASTE 5000€!!! haz clic aquí y envía tu cuenta bancaria YA.",
    "¿Puedes revisar el informe y decirme qué te parece antes del viernes?",
    "URGENT: your flight leaves in 3 hours and you have not checked in yet.",
    "Newsletter de abril: novedades, horarios y una entrevista. Sin prisa.",
    "Your account will be SUSPENDED in 24h unless you verify your card now.",
    "Hola Marta, ¿podrías venir a una entrevista el jueves? Confirma por correo.",
    "AVISO de corte de suministro mañana de 8h a 14h. Toma precauciones hoy.",
    "I attach the minutes of Tuesday's meeting for the record. No action needed.",
    "Could you confirm whether morning or afternoon suits you better for booking 45231?",
    "Estimado cliente: un familiar lejano le dejó 8000€. Envíenos sus claves para transferirlos. Atentamente, abogado.",
    "Your medical result is available — please review it today and book your doctor this week.",
    "Tu pedido 88412 ha sido entregado correctamente. Guarda este correo como comprobante.",
    "Pre-approved loan of 3000€ with no payslip, today only. Leave your ID and account number.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/data-prefetch/email-triage.jsonl")
    ap.add_argument("--pilot", type=int, default=500,
                    help="rows labelled with the teacher for the pilot report")
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    to_jsonl(rows, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({"job": "email-triage", "rows": len(rows),
                      "sha256": sha[:16], "out": str(out)}))

    # Teacher pilot on a head slice (deterministic first-N).
    from data.teacher_client import TeacherClient, agreement_report
    cli = TeacherClient()
    sample = rows[: args.pilot]
    texts = [r.state for r in sample]
    golds = [r.questions[0].answer for r in sample]
    t_labels = cli.label(texts, gold_hint=golds)
    q_labels = cli.qwen_proxy(texts)
    rep = agreement_report(
        [r["label"] for r in t_labels], [r["label"] for r in q_labels])
    rep.update(cli.stats())
    print(json.dumps({"teacher_pilot": rep}, indent=1, ensure_ascii=False))

    # Manifest entry (replace previous email-triage job if present).
    man_path = ROOT / "artifacts" / "data-prefetch" / "manifest.json"
    man = json.loads(man_path.read_text())
    man["jobs"] = [j for j in man.get("jobs", []) if j.get("job") != "email-triage"]
    man["jobs"].append({"job": "email-triage", "hf_id": None,
                        "status": "converted-intent-v1", "rows": len(rows),
                        "examples": len(rows), "skipped": 0, "sha256": sha,
                        "revision": "intent-v1"})
    man_path.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print("manifest: email-triage registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
