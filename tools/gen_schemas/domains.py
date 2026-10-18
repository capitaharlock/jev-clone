"""Domain catalogue + state rendering for the decision-schema generator.

A domain is DATA, not code: it names its fields and the value family each
field draws from. There is no ``STATES`` list and no ``{}``-placeholder
filter anywhere — a candidate is accepted on whether its answer is derivable
from its state, which is a quality bar, not a string check.

Hard negatives fall out of the family groups: values sharing a group are
near-misses of each other (``critical``/``high``), values in other groups are
easy distractors.
"""
from __future__ import annotations

from dataclasses import dataclass

from .vocab import t

# family -> ((value_key, group), ...). Same group == plausible hard negative.
FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {
    "urgency": (("urgency.critical", "top"), ("urgency.high", "top"),
                ("urgency.elevated", "top"), ("urgency.normal", "mid"),
                ("urgency.low", "bot"), ("urgency.deferred", "bot"),
                ("urgency.scheduled", "mid"), ("urgency.none", "bot")),
    "decision": (("decision.approve", "yes"), ("decision.approve_cond", "yes"),
                 ("decision.reject", "no"), ("decision.reject_final", "no"),
                 ("decision.escalate", "defer"), ("decision.request_info", "defer"),
                 ("decision.defer", "defer"), ("decision.cancel", "no")),
    "team": (("team.billing", "fin"), ("team.payments", "fin"),
             ("team.infra", "tech"), ("team.security", "tech"),
             ("team.support", "ops"), ("team.logistics", "ops"),
             ("team.legal", "gov"), ("team.quality", "gov")),
    "status": (("status.open", "live"), ("status.in_progress", "live"),
               ("status.blocked", "stuck"), ("status.waiting_customer", "stuck"),
               ("status.waiting_supplier", "stuck"), ("status.resolved", "done"),
               ("status.closed", "done"), ("status.reopened", "live")),
    "channel": (("channel.email", "text"), ("channel.chat", "text"),
                ("channel.phone", "voice"), ("channel.web_form", "machine"),
                ("channel.api", "machine"), ("channel.mail", "paper"),
                ("channel.branch", "paper"), ("channel.social", "text")),
    "cause": (("cause.human_error", "internal"), ("cause.config_error", "internal"),
              ("cause.hardware", "infra"), ("cause.network", "infra"),
              ("cause.third_party", "external"), ("cause.no_cause", "external"),
              ("cause.capacity", "infra"), ("cause.software_bug", "internal")),
    "stage": (("stage.canary", "progress"), ("stage.partial", "progress"),
              ("stage.full", "done"), ("stage.rolled_back", "stopped"),
              ("stage.on_hold", "stopped"), ("stage.scheduled", "planned"),
              ("stage.aborted", "stopped"), ("stage.pilot", "planned")),
    "region": (("region.north", "ns"), ("region.south", "ns"),
               ("region.east", "ew"), ("region.west", "ew"),
               ("region.central", "core"), ("region.coast", "edge"),
               ("region.island", "edge"), ("region.border", "edge")),
}

MAX_K = min(len(v) for v in FAMILIES.values())
MIN_K = 3

STATE_FORMATS = ("kv", "prose", "table", "log", "json")


@dataclass(frozen=True)
class Field:
    name: str  # stable field name, used as the record key
    label: str  # vocab key of the displayed label
    family: str  # key into FAMILIES

    def values(self) -> tuple[tuple[str, str], ...]:
        return FAMILIES[self.family]

    def group_of(self, value_key: str) -> str:
        for key, group in self.values():
            if key == value_key:
                return group
        raise KeyError(f"{value_key!r} not in family {self.family!r}")


@dataclass(frozen=True)
class Domain:
    id: str
    fields: tuple[Field, ...]

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"{self.id}: no field {name!r}")


DOMAINS: tuple[Domain, ...] = (
    Domain("support-routing", (
        Field("priority", "lbl.priority", "urgency"),
        Field("queue", "lbl.queue", "team"),
        Field("channel", "lbl.channel", "channel"),
        Field("status", "lbl.status", "status"),
    )),
    Domain("it-incident", (
        Field("severity", "lbl.severity", "urgency"),
        Field("owner_team", "lbl.owner_team", "team"),
        Field("cause", "lbl.cause", "cause"),
        Field("status", "lbl.status", "status"),
    )),
    Domain("logistics", (
        Field("region", "lbl.region", "region"),
        Field("status", "lbl.status", "status"),
        Field("cause", "lbl.cause", "cause"),
        Field("priority", "lbl.priority", "urgency"),
    )),
    Domain("finance-approval", (
        Field("decision", "lbl.decision", "decision"),
        Field("risk", "lbl.risk", "urgency"),
        Field("channel", "lbl.channel", "channel"),
        Field("queue", "lbl.queue", "team"),
    )),
    Domain("content-moderation", (
        Field("action", "lbl.action", "decision"),
        Field("priority", "lbl.priority", "urgency"),
        Field("channel", "lbl.channel", "channel"),
        Field("owner_team", "lbl.owner_team", "team"),
    )),
    Domain("devops-release", (
        Field("stage", "lbl.stage", "stage"),
        Field("decision", "lbl.decision", "decision"),
        Field("owner_team", "lbl.owner_team", "team"),
        Field("cause", "lbl.cause", "cause"),
    )),
)

DOMAIN_IDS = tuple(d.id for d in DOMAINS)


def domain(domain_id: str) -> Domain:
    for d in DOMAINS:
        if d.id == domain_id:
            return d
    raise KeyError(f"unknown domain {domain_id!r}; known: {DOMAIN_IDS}")


def question_for(field: Field, lang: str) -> str:
    return t(f"q.{field.label}", lang)


def render_state(dom: Domain, record: dict[str, str], lang: str, fmt: str,
                 header: str, omit: str | None = None,
                 mention_absent: bool = True) -> str:
    """Render a record in one of STATE_FORMATS.

    ``omit`` drops a field entirely — that is how an `unknown` case is built:
    the state stops carrying the answer, instead of the answer being relabelled.
    """
    if fmt not in STATE_FORMATS:
        raise ValueError(f"unknown state format {fmt!r}")
    rows = [(f, record[f.name]) for f in dom.fields
            if f.name in record and f.name != omit]
    lines: list[str] = []
    if fmt == "kv":
        lines.append(header)
        lines += [f"{t(f.label, lang)}: {t(v, lang)}" for f, v in rows]
    elif fmt == "prose":
        glue = t("glue.is", lang)
        lines.append(header)
        lines += [glue.format(label=t(f.label, lang), value=t(v, lang))
                  for f, v in rows]
    elif fmt == "table":
        lines.append(header)
        lines.append(t("glue.table", lang))
        lines.append("| --- | --- |")
        lines += [f"| {t(f.label, lang)} | {t(v, lang)} |" for f, v in rows]
    elif fmt == "log":
        lines.append(f"# {header}")
        lines += [f"{dom.id}.{f.name}={t(v, lang)}" for f, v in rows]
    else:  # json
        body = ", ".join(f'"{f.name}": "{t(v, lang)}"' for f, v in rows)
        lines.append(f"// {header}")
        lines.append("{" + body + "}")
    if omit is not None and mention_absent:
        lines.append(t("glue.absent", lang).format(
            label=t(dom.field(omit).label, lang)))
    return "\n".join(lines)
