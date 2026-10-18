"""Localized vocabulary for the decision-schema generator (#T-gen-schemas).

Values live in **families**, never in a hardcoded ``STATES`` list: a domain
composes families into fields, so a new domain costs zero new strings and a
new language costs one column. Every entry is a tuple aligned with ``LANGS``,
which is what makes "the same schema in four languages" a real axis instead
of four hand-written templates (audit-2026-09-21, finding D).

Option ids are the value *keys* (language-neutral); only the text is
localized — options are scored by their text, never by their position.
"""
from __future__ import annotations

LANGS = ("es", "en", "fr", "de")

# --- value families: key -> (es, en, fr, de) ------------------------------
VOCAB: dict[str, tuple[str, str, str, str]] = {
    # urgency
    "urgency.critical": ("crítica", "critical", "critique", "kritisch"),
    "urgency.high": ("alta", "high", "élevée", "hoch"),
    "urgency.elevated": ("moderada-alta", "elevated", "moyenne-haute", "erhöht"),
    "urgency.normal": ("normal", "normal", "normale", "normal"),
    "urgency.low": ("baja", "low", "faible", "niedrig"),
    "urgency.deferred": ("aplazada", "deferred", "différée", "zurückgestellt"),
    "urgency.scheduled": ("planificada", "scheduled", "planifiée", "geplant"),
    "urgency.none": ("sin prioridad", "no priority", "sans priorité", "ohne Priorität"),
    # decision
    "decision.approve": ("aprobar", "approve", "approuver", "genehmigen"),
    "decision.approve_cond": ("aprobar con condiciones", "approve with conditions",
                              "approuver sous conditions", "mit Auflagen genehmigen"),
    "decision.reject": ("rechazar", "reject", "refuser", "ablehnen"),
    "decision.reject_final": ("rechazar sin apelación", "reject as final",
                              "refuser définitivement", "endgültig ablehnen"),
    "decision.escalate": ("escalar", "escalate", "escalader", "eskalieren"),
    "decision.request_info": ("pedir más información", "request more information",
                              "demander des informations", "Nachforderung stellen"),
    "decision.defer": ("posponer", "defer", "reporter", "vertagen"),
    "decision.cancel": ("cancelar", "cancel", "annuler", "stornieren"),
    # team
    "team.billing": ("facturación", "billing", "facturation", "Rechnungswesen"),
    "team.payments": ("pagos", "payments", "paiements", "Zahlungen"),
    "team.infra": ("infraestructura", "infrastructure", "infrastructure", "Infrastruktur"),
    "team.security": ("seguridad", "security", "sécurité", "Sicherheit"),
    "team.support": ("soporte", "support", "assistance", "Support"),
    "team.logistics": ("logística", "logistics", "logistique", "Logistik"),
    "team.legal": ("jurídico", "legal", "juridique", "Rechtsabteilung"),
    "team.quality": ("calidad", "quality", "qualité", "Qualität"),
    # status
    "status.open": ("abierto", "open", "ouvert", "offen"),
    "status.in_progress": ("en curso", "in progress", "en cours", "in Bearbeitung"),
    "status.blocked": ("bloqueado", "blocked", "bloqué", "blockiert"),
    "status.waiting_customer": ("esperando al cliente", "waiting on customer",
                                "en attente du client", "wartet auf Kunden"),
    "status.waiting_supplier": ("esperando al proveedor", "waiting on supplier",
                                "en attente du fournisseur", "wartet auf Lieferant"),
    "status.resolved": ("resuelto", "resolved", "résolu", "gelöst"),
    "status.closed": ("cerrado", "closed", "clôturé", "geschlossen"),
    "status.reopened": ("reabierto", "reopened", "rouvert", "wiedereröffnet"),
    # channel
    "channel.email": ("correo electrónico", "email", "courriel", "E-Mail"),
    "channel.chat": ("chat", "chat", "tchat", "Chat"),
    "channel.phone": ("teléfono", "phone", "téléphone", "Telefon"),
    "channel.web_form": ("formulario web", "web form", "formulaire web", "Webformular"),
    "channel.api": ("API", "API", "API", "API"),
    "channel.mail": ("carta postal", "postal mail", "courrier postal", "Briefpost"),
    "channel.branch": ("oficina", "branch office", "agence", "Filiale"),
    "channel.social": ("redes sociales", "social media", "réseaux sociaux",
                       "soziale Medien"),
    # cause
    "cause.human_error": ("error humano", "human error", "erreur humaine",
                          "menschliches Versagen"),
    "cause.config_error": ("error de configuración", "configuration error",
                           "erreur de configuration", "Konfigurationsfehler"),
    "cause.hardware": ("fallo de hardware", "hardware failure", "panne matérielle",
                       "Hardwaredefekt"),
    "cause.network": ("incidencia de red", "network incident", "incident réseau",
                      "Netzwerkstörung"),
    "cause.third_party": ("proveedor externo", "third-party provider",
                          "prestataire externe", "Drittanbieter"),
    "cause.no_cause": ("sin causa identificada", "no identified cause",
                       "aucune cause identifiée", "keine erkannte Ursache"),
    "cause.capacity": ("falta de capacidad", "capacity shortfall",
                       "manque de capacité", "Kapazitätsengpass"),
    "cause.software_bug": ("defecto de software", "software defect",
                           "défaut logiciel", "Softwarefehler"),
    # stage
    "stage.canary": ("canario", "canary", "canari", "Canary"),
    "stage.partial": ("despliegue parcial", "partial rollout", "déploiement partiel",
                      "Teil-Rollout"),
    "stage.full": ("despliegue completo", "full rollout", "déploiement complet",
                   "vollständiges Rollout"),
    "stage.rolled_back": ("revertido", "rolled back", "annulé", "zurückgerollt"),
    "stage.on_hold": ("en pausa", "on hold", "en pause", "pausiert"),
    "stage.scheduled": ("programado", "scheduled", "programmé", "terminiert"),
    "stage.aborted": ("abortado", "aborted", "interrompu", "abgebrochen"),
    "stage.pilot": ("piloto", "pilot", "pilote", "Pilot"),
    # region
    "region.north": ("norte", "north", "nord", "Nord"),
    "region.south": ("sur", "south", "sud", "Süd"),
    "region.east": ("este", "east", "est", "Ost"),
    "region.west": ("oeste", "west", "ouest", "West"),
    "region.central": ("centro", "central", "centre", "Zentrum"),
    "region.coast": ("costa", "coast", "littoral", "Küste"),
    "region.island": ("islas", "islands", "îles", "Inseln"),
    "region.border": ("frontera", "border", "frontière", "Grenze"),
    # --- field labels ----------------------------------------------------
    "lbl.priority": ("prioridad", "priority", "priorité", "Priorität"),
    "lbl.severity": ("severidad", "severity", "sévérité", "Schweregrad"),
    # feminine in es/fr so the urgency values agree with the label.
    "lbl.risk": ("escala de riesgo", "risk level", "\u00e9chelle de risque",
                 "Risikostufe"),
    "lbl.queue": ("cola de destino", "destination queue", "file de destination",
                  "Zielwarteschlange"),
    "lbl.owner_team": ("equipo responsable", "owning team", "équipe responsable",
                       "zuständiges Team"),
    "lbl.channel": ("canal de entrada", "intake channel", "canal d'entrée",
                    "Eingangskanal"),
    "lbl.status": ("estado", "status", "statut", "Status"),
    "lbl.cause": ("causa raíz", "root cause", "cause racine", "Grundursache"),
    "lbl.region": ("zona", "zone", "zone", "Zone"),
    "lbl.decision": ("decisión", "decision", "décision", "Entscheidung"),
    "lbl.action": ("acción de revisión", "review action", "action de révision",
                   "Prüfmaßnahme"),
    "lbl.stage": ("fase de despliegue", "rollout stage", "phase de déploiement",
                  "Rollout-Phase"),
    # --- question templates ----------------------------------------------
    "q.lbl.priority": ("¿Qué prioridad corresponde a este caso?",
                       "Which priority applies to this case?",
                       "Quelle priorité s'applique à ce dossier ?",
                       "Welche Priorität gilt für diesen Fall?"),
    "q.lbl.severity": ("¿Qué severidad tiene la incidencia?",
                       "What severity does this incident have?",
                       "Quelle est la sévérité de cet incident ?",
                       "Welchen Schweregrad hat dieser Vorfall?"),
    "q.lbl.risk": ("\u00bfQu\u00e9 escala de riesgo se ha asignado?",
                   "What risk level has been assigned?",
                   "Quelle \u00e9chelle de risque a \u00e9t\u00e9 attribu\u00e9e ?",
                   "Welche Risikostufe wurde vergeben?"),
    "q.lbl.queue": ("¿A qué cola debe ir este caso?",
                    "Which queue should this case go to?",
                    "Vers quelle file ce dossier doit-il aller ?",
                    "In welche Warteschlange gehört dieser Fall?"),
    "q.lbl.owner_team": ("¿Qué equipo es responsable?", "Which team owns this?",
                         "Quelle équipe en est responsable ?",
                         "Welches Team ist zuständig?"),
    "q.lbl.channel": ("¿Por qué canal entró la solicitud?",
                      "Through which channel did the request arrive?",
                      "Par quel canal la demande est-elle arrivée ?",
                      "Über welchen Kanal kam die Anfrage?"),
    "q.lbl.status": ("¿En qué estado se encuentra?", "What state is it in?",
                     "Dans quel état se trouve-t-il ?", "In welchem Status ist er?"),
    "q.lbl.cause": ("¿Cuál es la causa raíz registrada?",
                    "What is the recorded root cause?",
                    "Quelle est la cause racine enregistrée ?",
                    "Was ist die erfasste Grundursache?"),
    "q.lbl.region": ("¿A qué zona pertenece el envío?",
                     "Which zone does the shipment belong to?",
                     "À quelle zone appartient l'envoi ?",
                     "Zu welcher Zone gehört die Sendung?"),
    "q.lbl.decision": ("¿Qué decisión se ha tomado?", "Which decision was taken?",
                       "Quelle décision a été prise ?",
                       "Welche Entscheidung wurde getroffen?"),
    "q.lbl.action": ("¿Qué acción de revisión corresponde?",
                     "Which review action applies?",
                     "Quelle action de révision s'applique ?",
                     "Welche Prüfmaßnahme gilt?"),
    "q.lbl.stage": ("¿En qué fase de despliegue está?",
                    "Which rollout stage is it in?",
                    "À quelle phase de déploiement en est-on ?",
                    "In welcher Rollout-Phase befindet es sich?"),
    # --- state headers (one per domain), {ref} {who} {day} {num} ----------
    "hdr.support-routing": (
        "Ticket {ref} abierto por {who} el día {day}; {num} mensajes en el hilo.",
        "Ticket {ref} opened by {who} on day {day}; {num} messages in the thread.",
        "Ticket {ref} ouvert par {who} le jour {day} ; {num} messages dans le fil.",
        "Ticket {ref} von {who} am Tag {day} eröffnet; {num} Nachrichten im Verlauf."),
    "hdr.it-incident": (
        "Incidencia {ref} detectada por {who} en el minuto {num} del día {day}.",
        "Incident {ref} detected by {who} at minute {num} of day {day}.",
        "Incident {ref} détecté par {who} à la minute {num} du jour {day}.",
        "Vorfall {ref} von {who} in Minute {num} des Tages {day} erkannt."),
    "hdr.logistics": (
        "Envío {ref} de {who}, {num} bultos, salida prevista el día {day}.",
        "Shipment {ref} from {who}, {num} packages, departure on day {day}.",
        "Envoi {ref} de {who}, {num} colis, départ prévu le jour {day}.",
        "Sendung {ref} von {who}, {num} Packstücke, Abfahrt am Tag {day}."),
    "hdr.finance-approval": (
        "Solicitud {ref} de {who} por {num} unidades, registrada el día {day}.",
        "Request {ref} from {who} for {num} units, filed on day {day}.",
        "Demande {ref} de {who} pour {num} unités, déposée le jour {day}.",
        "Antrag {ref} von {who} über {num} Einheiten, eingereicht am Tag {day}."),
    "hdr.content-moderation": (
        "Reporte {ref} sobre contenido de {who}, {num} denuncias, día {day}.",
        "Report {ref} about content from {who}, {num} flags, day {day}.",
        "Signalement {ref} sur un contenu de {who}, {num} plaintes, jour {day}.",
        "Meldung {ref} zu Inhalten von {who}, {num} Hinweise, Tag {day}."),
    "hdr.devops-release": (
        "Versión {ref} preparada por {who}, {num} servicios, corte el día {day}.",
        "Release {ref} prepared by {who}, {num} services, cut on day {day}.",
        "Version {ref} préparée par {who}, {num} services, gel le jour {day}.",
        "Release {ref} von {who} vorbereitet, {num} Dienste, Stichtag {day}."),
    # --- prose glue -------------------------------------------------------
    # The label is quoted as a term on purpose: it keeps every sentence
    # grammatical in four languages without a gender table per label.
    "glue.is": ("«{label}» consta como {value}.",
                "\u201c{label}\u201d is recorded as {value}.",
                "\u00ab {label} \u00bb est enregistr\u00e9 comme {value}.",
                "\u201e{label}\u201c ist als {value} erfasst."),
    "glue.absent": ("«{label}» no consta en el expediente.",
                    "\u201c{label}\u201d is not recorded in the file.",
                    "\u00ab {label} \u00bb ne figure pas au dossier.",
                    "\u201e{label}\u201c ist in der Akte nicht vermerkt."),
    "glue.table": ("| campo | valor |", "| field | value |",
                   "| champ | valeur |", "| Feld | Wert |"),
}

# Proper nouns and refs. They are masked out of the skeleton, so varying them
# must NOT look like diversity — that is exactly the trap finding C fell into.
PEOPLE = ("Marta", "Jordi", "Anna", "Pau", "Laia", "Marc", "Nuria", "Oriol",
          "Ingrid", "Tomas", "Sofie", "Lucien", "Aurelie", "Bastian", "Carla",
          "Sergi", "Klaus", "Margit", "Helena", "Pol")
ORGS = ("Delta Norte", "Casbridge", "Vallmar", "Orbitel", "Nordfracht",
        "Lumiere SA", "Kestrel Labs", "Ponent Group", "Arcadia BV", "Tramuntana")
REF_PREFIXES = ("REQ", "INC", "TCK", "SHP", "REL", "CASE", "OPS", "MOD")


def t(key: str, lang: str) -> str:
    """Localized text for a vocab key. Raises on an unknown key or language."""
    try:
        col = LANGS.index(lang)
    except ValueError:
        raise KeyError(f"unknown language {lang!r}; known: {LANGS}") from None
    try:
        return VOCAB[key][col]
    except KeyError:
        raise KeyError(f"unknown vocab key {key!r}") from None
