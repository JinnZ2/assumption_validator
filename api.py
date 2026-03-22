# assumption_validator/api.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Universal REST + SSE API.
# No dependency on earth-systems-physics.
# Any adapter plugs in at startup or via /v1/adapter.
# Any client — AI model, dashboard, operator console —
# queries live assumption validity and gets honest uncertainty back.
#
# Requires: pip install flask flask-cors

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import json
import time
from datetime import datetime
from typing import Optional, Dict

from assumption_validator.registry import (
    REGISTRY,
    RiskLevel,
    full_report,
    detect_cascade_risk,
    global_confidence_multiplier,
)
from assumption_validator.monitors import (
    UniversalMonitor,
    print_alert,
    Alert,
)
from assumption_validator.vector_risk import (
    VectorRiskForecaster,
)


# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────

app  = Flask(__name__)
CORS(app)

_monitor    : Optional[UniversalMonitor]   = None
_forecaster : Optional[VectorRiskForecaster] = None
_monitor_lock = threading.Lock()


def get_monitor() -> UniversalMonitor:
    global _monitor, _forecaster
    with _monitor_lock:
        if _monitor is None:
            # Default: generic adapter with empty values
            # Replace at startup or via /v1/adapter endpoint
            from assumption_validator.adapters.generic import GenericAdapter
            adapter    = GenericAdapter({})
            _monitor   = UniversalMonitor(
                adapter         = adapter,
                poll_interval_s = 3600.0,
                alert_callbacks = [print_alert],
            )
            _monitor.start()
        if _forecaster is None:
            _forecaster = VectorRiskForecaster()
        return _monitor


def get_forecaster() -> VectorRiskForecaster:
    get_monitor()   # ensures _forecaster is initialized
    return _forecaster


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _json(obj, status: int = 200) -> Response:
    return Response(
        json.dumps(obj, indent=2, default=str),
        status   = status,
        mimetype = "application/json",
    )


def _error(msg: str, status: int = 400) -> Response:
    return _json({"error": msg}, status)


def _current_report() -> Dict:
    m = get_monitor()
    r = m.current_report()
    if r is None:
        r = m.poll_once()
    return r


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

# ── HEALTH ───────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Service liveness and monitor status."""
    m = get_monitor()
    r = m.current_report()
    return _json({
        "status":       "operational",
        "service":      "assumption-validator",
        "version":      "0.1.0",
        "adapter":      type(m.adapter).__name__,
        "poll_count":   r.get("poll_count", 0) if r else 0,
        "last_poll":    r.get("timestamp")     if r else None,
        "assumptions":  len(REGISTRY),
        "timestamp":    datetime.utcnow().isoformat(),
    })


# ── FULL VALIDITY REPORT ─────────────────────

@app.route("/v1/validity", methods=["GET"])
def validity():
    """
    Full assumption validity report.

    Query params:
        refresh=true        force new poll
        domain=<str>        filter by domain (atmosphere, ocean, energy, ...)
        status=yellow|red   filter to degraded only
    """
    m = get_monitor()

    if request.args.get("refresh", "false").lower() == "true":
        m.poll_once()

    r = _current_report()

    assumptions  = dict(r.get("assumptions", {}))
    domain_filter  = request.args.get("domain", "").lower()
    status_filter  = request.args.get("status", "").upper()

    if domain_filter:
        assumptions = {
            k: v for k, v in assumptions.items()
            if v.get("domain", "").lower() == domain_filter
        }

    if status_filter in ("YELLOW", "RED"):
        assumptions = {
            k: v for k, v in assumptions.items()
            if v.get("status") == status_filter
        }

    return _json({
        "timestamp":                    r.get("timestamp"),
        "global_confidence_multiplier": r.get("global_confidence_multiplier"),
        "summary":                      r.get("summary"),
        "cascade":                      r.get("cascade"),
        "assumptions":                  assumptions,
    })


# ── SINGLE ASSUMPTION ────────────────────────

@app.route("/v1/validity/<assumption_id>", methods=["GET"])
def validity_single(assumption_id: str):
    """
    Status + trend for one assumption.

    Path:
        assumption_id : key from registry (e.g. co2_concentration)
    """
    if assumption_id not in REGISTRY:
        return _error(
            f"Unknown assumption '{assumption_id}'. "
            f"Available: {sorted(REGISTRY.keys())}",
            status=404,
        )

    r          = _current_report()
    assessment = r.get("assumptions", {}).get(assumption_id, {})
    trend      = r.get("trends",      {}).get(assumption_id, {})
    boundary   = REGISTRY[assumption_id]

    return _json({
        "assumption_id":      assumption_id,
        "name":               boundary.name,
        "domain":             boundary.domain,
        "parameter":          boundary.parameter,
        "units":              boundary.units,
        "layer_key":          boundary.layer_key,
        "current_value":      assessment.get("value"),
        "status":             assessment.get("status"),
        "confidence_penalty": assessment.get("confidence_penalty"),
        "proximity_to_red":   assessment.get("proximity_to_red"),
        "green_range":        boundary.green_range,
        "yellow_range":       boundary.yellow_range,
        "red_threshold":      boundary.red_threshold,
        "couplings":          boundary.couplings,
        "notes":              boundary.notes,
        "trend": {
            "drift_rate_per_poll":  trend.get("drift_rate_per_poll"),
            "acceleration":         trend.get("acceleration"),
            "hours_to_red":         trend.get("hours_to_red"),
            "consecutive_degraded": trend.get("consecutive_degraded"),
            "status_history_24":    trend.get("status_history_24"),
        },
        "timestamp": r.get("timestamp"),
    })


# ── DOMAIN SUMMARY ───────────────────────────

@app.route("/v1/domains", methods=["GET"])
def domains():
    """
    Validity grouped by domain.
    Shows worst status and all assumptions per domain.
    """
    r           = _current_report()
    assumptions = r.get("assumptions", {})
    status_rank = {"GREEN": 0, "YELLOW": 1, "RED": 2, "UNKNOWN": -1}

    domain_map: Dict[str, list] = {}
    for aid, data in assumptions.items():
        d = data.get("domain", "general")
        domain_map.setdefault(d, []).append((aid, data))

    out = {}
    for domain, items in sorted(domain_map.items()):
        worst_status = max(
            (v.get("status", "UNKNOWN") for _, v in items),
            key = lambda s: status_rank.get(s, -1),
        )
        out[domain] = {
            "domain":           domain,
            "worst_status":     worst_status,
            "assumption_count": len(items),
            "assumptions": {
                aid: {
                    "name":   v.get("name"),
                    "status": v.get("status"),
                    "value":  v.get("value"),
                    "units":  v.get("units"),
                }
                for aid, v in items
            },
        }

    return _json({
        "timestamp": r.get("timestamp"),
        "domains":   out,
        "cascade":   r.get("cascade"),
    })


# ── ADJUST PREDICTION ────────────────────────

@app.route("/v1/adjust", methods=["POST"])
def adjust():
    """
    Submit any prediction; receive confidence-adjusted version.

    Request body (JSON):
    {
        "prediction":        { ...any dict... },
        "base_confidence":   0.85,
        "assumptions":       ["co2_concentration", "amoc_strength"],
        "model_name":        "optional label",
        "derivation_regime": "holocene"
    }

    If "assumptions" is omitted, all registry assumptions are used.
    """
    data = request.get_json(silent=True)
    if not data:
        return _error("JSON body required")

    prediction   = data.get("prediction", {})
    base_conf    = data.get("base_confidence", 0.5)
    required     = data.get("assumptions")
    model_name   = data.get("model_name",        "unnamed")
    deriv_regime = data.get("derivation_regime", "holocene")

    if not (0.0 <= base_conf <= 1.0):
        return _error("base_confidence must be 0.0 – 1.0")

    if required:
        unknown = [a for a in required if a not in REGISTRY]
        if unknown:
            return _error(f"Unknown assumption IDs: {unknown}")

    r           = _current_report()
    assumptions = r.get("assumptions", {})
    subset      = (
        {k: v for k, v in assumptions.items() if k in required}
        if required else assumptions
    )

    # Confidence multiplier
    multiplier = 1.0
    for v in subset.values():
        p = v.get("confidence_penalty", 0.0)
        if isinstance(p, (int, float)):
            multiplier *= (1.0 - p)

    adjusted_conf = base_conf * multiplier

    reds    = [k for k, v in subset.items() if v.get("status") == "RED"]
    yellows = [k for k, v in subset.items() if v.get("status") == "YELLOW"]

    if reds:
        overall = "INVALID"
    elif yellows:
        overall = "DEGRADED"
    else:
        overall = "VALID"

    # Regime warning
    regime_warning = None
    if deriv_regime == "holocene":
        co2 = assumptions.get("co2_concentration", {}).get("value")
        if co2 and isinstance(co2, (int, float)) and co2 > 350:
            regime_warning = (
                f"Model derived in Holocene regime (CO₂ < 350 ppm). "
                f"Current CO₂ {co2:.0f} ppm exceeds Holocene range. "
                f"Equations may not apply."
            )

    # Warnings list
    warnings = []
    for k in reds:
        b = REGISTRY.get(k)
        warnings.append(
            f"CRITICAL [{k}]: {b.name if b else k} — RED. "
            f"Value: {subset[k].get('value'):.4g} {subset[k].get('units','')}."
        )
    for k in yellows:
        b = REGISTRY.get(k)
        warnings.append(
            f"CAUTION [{k}]: {b.name if b else k} — YELLOW. "
            f"Value: {subset[k].get('value'):.4g} {subset[k].get('units','')}."
        )
    if regime_warning:
        warnings.append(regime_warning)

    return _json({
        "model_name":            model_name,
        "derivation_regime":     deriv_regime,
        "prediction":            prediction,
        "original_confidence":   base_conf,
        "adjusted_confidence":   adjusted_conf,
        "confidence_multiplier": multiplier,
        "confidence_loss_pct":   round((1.0 - multiplier) * 100, 1),
        "overall_status":        overall,
        "red_assumptions":       reds,
        "yellow_assumptions":    yellows,
        "warnings":              warnings,
        "regime_warning":        regime_warning,
        "cascade_level":         r.get("cascade", {}).get("cascade_level"),
        "timestamp":             r.get("timestamp"),
    })


# ── CASCADE ───────────────────────────────────

@app.route("/v1/cascade", methods=["GET"])
def cascade():
    """
    Current cascade risk and history.

    Query params:
        history=<int>  number of snapshots (default 24)
    """
    n = int(request.args.get("history", 24))
    m = get_monitor()
    r = _current_report()
    return _json({
        "current":   r.get("cascade"),
        "history":   m.cascade_trend(n=n),
        "timestamp": r.get("timestamp"),
    })


# ── TRENDS ───────────────────────────────────

@app.route("/v1/trends", methods=["GET"])
def trends():
    """
    Drift rates and time-to-red for all assumptions.
    Sorted by proximity to RED (soonest first).

    Query params:
        imminent=true  only show assumptions with hours_to_red < 720
        domain=<str>   filter by domain
    """
    m          = get_monitor()
    r          = _current_report()
    trend_data = dict(r.get("trends", {}))
    imminent   = request.args.get("imminent", "false").lower() == "true"
    domain_f   = request.args.get("domain", "").lower()

    if domain_f:
        trend_data = {
            k: v for k, v in trend_data.items()
            if REGISTRY.get(k, None) and
               REGISTRY[k].domain.lower() == domain_f
        }

    if imminent:
        trend_data = {
            k: v for k, v in trend_data.items()
            if isinstance(v.get("hours_to_red"), (int, float))
            and v["hours_to_red"] < 720
        }

    sorted_trends = dict(
        sorted(
            trend_data.items(),
            key = lambda item: item[1].get("hours_to_red") or 1e9,
        )
    )

    return _json({
        "timestamp": r.get("timestamp"),
        "count":     len(sorted_trends),
        "trends":    sorted_trends,
    })


# ── ALERTS ───────────────────────────────────

@app.route("/v1/alerts", methods=["GET"])
def alerts():
    """
    Return and drain pending alerts.
    Once read, alerts are cleared from queue.
    """
    m  = get_monitor()
    al = m.drain_alerts()
    return _json({
        "count":  len(al),
        "alerts": [
            {
                "timestamp":       a.timestamp.isoformat(),
                "assumption_id":   a.assumption_id,
                "assumption_name": a.assumption_name,
                "alert_type":      a.alert_type,
                "previous_status": a.previous_status,
                "current_status":  a.current_status,
                "message":         a.message,
                "hours_to_red":    a.hours_to_red,
                "cascade_level":   a.cascade_level,
            }
            for a in al
        ],
    })


# ── BLIND SPOTS ───────────────────────────────

@app.route("/v1/blind_spots", methods=["GET"])
def blind_spots():
    """
    Assumptions ranked by blind spot size.
    These invalidate the most downstream knowledge when they break.

    Query params:
        top=<int>  number to return (default 10)
    """
    n  = int(request.args.get("top", 10))
    f  = get_forecaster()
    r  = _current_report()
    f.update_from_monitor(r.get("trends", {}))
    bs = f.blind_spots(top_n=n)
    return _json({
        "timestamp":   r.get("timestamp"),
        "blind_spots": bs,
    })


# ── RISK VECTORS ─────────────────────────────

@app.route("/v1/risk_vectors", methods=["GET"])
def risk_vectors():
    """
    3D risk space coordinates for all assumptions.
    x = normalized drift, y = blind spot size, z = urgency.
    Suitable for visualization.
    """
    f = get_forecaster()
    r = _current_report()
    f.update_from_monitor(r.get("trends", {}))
    return _json({
        "timestamp":    r.get("timestamp"),
        "risk_vectors": f.risk_space_3d(),
        "clusters":     f.cluster_risk(),
    })


# ── REASSESSMENT PRIORITIES ──────────────────

@app.route("/v1/reassessment", methods=["GET"])
def reassessment():
    """
    Ordered list of assumptions AI models should reassess first.
    P1 = immediate. P2 = 30 days. P3 = cluster-level.
    """
    f = get_forecaster()
    r = _current_report()
    f.update_from_monitor(r.get("trends", {}))
    return _json({
        "timestamp":    r.get("timestamp"),
        "priorities":   f.reassessment_priorities(),
        "summary":      f.full_report()["summary"],
    })


# ── REGISTRY ─────────────────────────────────

@app.route("/v1/registry", methods=["GET"])
def registry():
    """
    Full assumption registry — boundaries, units, domains, couplings.

    Query params:
        domain=<str>  filter by domain
    """
    domain_f = request.args.get("domain", "").lower()
    out = {}
    for aid, b in REGISTRY.items():
        if domain_f and b.domain.lower() != domain_f:
            continue
        out[aid] = {
            "name":             b.name,
            "parameter":        b.parameter,
            "units":            b.units,
            "domain":           b.domain,
            "green_range":      b.green_range,
            "yellow_range":     b.yellow_range,
            "red_threshold":    b.red_threshold,
            "higher_is_worse":  b.higher_is_worse,
            "layer_key":        b.layer_key,
            "couplings":        b.couplings,
            "rate_of_change":   b.rate_of_change,
            "notes":            b.notes,
        }
    return _json({"count": len(out), "registry": out})


# ── ADAPTER SWAP ─────────────────────────────

@app.route("/v1/adapter", methods=["POST"])
def swap_adapter():
    """
    Hot-swap the data adapter without restarting the service.

    Request body (JSON):
    {
        "adapter": "noaa" | "generic" | "earth_systems",
        "values":  { ... }   (for generic adapter only)
    }
    """
    data         = request.get_json(silent=True)
    adapter_name = (data or {}).get("adapter", "generic")
    values       = (data or {}).get("values", {})

    m = get_monitor()

    try:
        if adapter_name == "noaa":
            from assumption_validator.adapters.noaa import NOAAAdapter
            m.swap_adapter(NOAAAdapter())
        elif adapter_name == "earth_systems":
            from assumption_validator.adapters.earth_systems import EarthSystemsAdapter
            m.swap_adapter(EarthSystemsAdapter())
        elif adapter_name == "generic":
            from assumption_validator.adapters.generic import GenericAdapter
            m.swap_adapter(GenericAdapter(values))
        else:
            return _error(
                f"Unknown adapter '{adapter_name}'. "
                f"Options: noaa, earth_systems, generic"
            )
    except ImportError as exc:
        return _error(f"Adapter import failed: {exc}")

    r = m.poll_once()
    return _json({
        "adapter":   adapter_name,
        "message":   f"Adapter swapped to {adapter_name}",
        "cascade":   r.get("cascade"),
        "summary":   r.get("summary"),
        "timestamp": r.get("timestamp"),
    })


# ── UPDATE VALUES (generic adapter) ──────────

@app.route("/v1/values", methods=["POST"])
def update_values():
    """
    Push new values into a GenericAdapter without swapping adapters.
    Useful for programmatic updates from any source.

    Request body (JSON):
    {
        "co2_ppm":       428.0,
        "amoc_sv":       14.5,
        "grid_inertia_s": 3.2,
        ...
    }
    Keys must match AssumptionBoundary.layer_key values.
    """
    values = request.get_json(silent=True)
    if not values or not isinstance(values, dict):
        return _error("JSON object of {layer_key: value} pairs required")

    m = get_monitor()

    # If adapter supports update, use it; otherwise wrap in GenericAdapter
    if hasattr(m.adapter, "update"):
        m.adapter.update(values)
    else:
        from assumption_validator.adapters.generic import GenericAdapter
        m.swap_adapter(GenericAdapter(values))

    r = m.poll_once()
    return _json({
        "message":   f"Updated {len(values)} values",
        "keys":      list(values.keys()),
        "cascade":   r.get("cascade"),
        "summary":   r.get("summary"),
        "timestamp": r.get("timestamp"),
    })


# ── SSE STREAM ───────────────────────────────

@app.route("/v1/stream", methods=["GET"])
def stream():
    """
    Server-sent events — JSON event each poll cycle.
    Clients receive multiplier, summary, cascade level.

    Usage:
        curl -N http://localhost:5000/v1/stream
    """
    m = get_monitor()

    def generate():
        last_poll = -1
        while True:
            r = m.current_report()
            if r is not None:
                poll = r.get("poll_count", 0)
                if poll != last_poll:
                    last_poll = poll
                    payload = json.dumps({
                        "timestamp":  r.get("timestamp"),
                        "multiplier": r.get("global_confidence_multiplier"),
                        "summary":    r.get("summary"),
                        "cascade":    r.get("cascade"),
                    }, default=str)
                    yield f"data: {payload}\n\n"
            time.sleep(5)

    return Response(
        generate(),
        mimetype = "text/event-stream",
        headers  = {
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

ROUTES = [
    ("GET",  "/health",                "service liveness"),
    ("GET",  "/v1/validity",           "full validity report"),
    ("GET",  "/v1/validity/<id>",      "single assumption"),
    ("GET",  "/v1/domains",            "validity by domain"),
    ("POST", "/v1/adjust",             "adjust prediction confidence"),
    ("GET",  "/v1/cascade",            "cascade risk + history"),
    ("GET",  "/v1/trends",             "drift rates + time-to-red"),
    ("GET",  "/v1/alerts",             "drain alert queue"),
    ("GET",  "/v1/blind_spots",        "ranked blind spot analysis"),
    ("GET",  "/v1/risk_vectors",       "3D risk space coordinates"),
    ("GET",  "/v1/reassessment",       "AI reassessment priorities"),
    ("GET",  "/v1/registry",           "full assumption registry"),
    ("POST", "/v1/adapter",            "hot-swap data adapter"),
    ("POST", "/v1/values",             "push values to generic adapter"),
    ("GET",  "/v1/stream",             "SSE live updates"),
]


def print_routes():
    print("=" * 60)
    print("ASSUMPTION VALIDATOR API")
    print("=" * 60)
    for method, path, desc in ROUTES:
        print(f"  {method:<5} {path:<32} {desc}")
    print("=" * 60)


if __name__ == "__main__":
    print_routes()
    print("Initializing monitor...")
    get_monitor()
    print("Monitor running. API on port 5000.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
