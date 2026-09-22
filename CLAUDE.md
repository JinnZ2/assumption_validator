# CLAUDE.md

Guidance for working in this repository. Public; CC0; nothing here is private.

## What this repository is

A **meta-layer over other systems' assumptions**. It holds no model of
the world. It holds the *stability envelope* of things other models
assumed about the world, watches where the current value sits inside
that envelope, and hands back an adjusted confidence plus the reason.

```text
any model / prediction / decision system
        |
        | (its assumptions, named)
        v
+-------------------------------------------------------------+
|  registry.py      26 AssumptionBoundary records, 8 domains   |
|                   green_range / yellow_range / red_threshold |
|                   COUPLING_GRAPH: who drives whom            |
|                        |                                     |
|   value in ---->  assess() ----> (RiskLevel, penalty, prox)  |
|                        |                                     |
|                        +--> global_confidence_multiplier()   |
|                        +--> detect_cascade_risk()            |
+-------------------------------------------------------------+
        |                        |                        |
        v                        v                        v
  monitors.py              vector_risk.py              api.py
  drift over time          blind spot per assumption   REST + SSE
  d/dt, d2/dt2, ttr        risk velocity + accel       15 routes
  Alert on change          cluster co-failure          any client
        |                        |                        |
        +------------------------+------------------------+
                                 |
                        adapters/ (the only input side)
                        generic / earth_systems / noaa
```

It is **not** a climate model, a prediction engine, or a policy tool.
Nothing here computes what will happen. It computes *how much to trust
whatever else computed it*.

## Hard constraints

```text
python >= 3.9
NOT stdlib-only              numpy, scipy, flask, flask-cors, requests
absolute imports only        `assumption_validator.registry`, so the PARENT
                             dir must be on sys.path; never `import registry`
no tests, no CI              nothing in the tree verifies any of this
license: CC0                 no attribution, nothing proprietary
```

**In this container: numpy, scipy, flask and flask-cors are ABSENT.**
`requests` is present. So `registry.py`, `monitors.py`, `vector_risk.py`
and `api.py` do not import here, and nothing in the repo can be run to
check a change. Treat every edit as unexercised until someone with the
dependencies runs it. Do not report a change as working.

## The four states, and what each refuses

```text
GREEN    equations valid         penalty 0
YELLOW   transition, degrading   penalty scales with position in band
RED      regime changed          equations invalid, not merely imprecise
UNKNOWN  no data                 NOT green. A missing value is not a pass.
```

`UNKNOWN` is the load-bearing one. `assess(None)` returns
`(UNKNOWN, 0.0, 0.0)` -- an absent reading contributes no penalty and
also no reassurance. Anything that maps a missing value onto GREEN, or
onto a default, breaks the only thing this layer is for.

## Cascade: the second refusal

```text
MINIMAL    no convergence
LOW        n_yellow >= 4                       broad degradation
MODERATE   n_coupled >= 3                      coupled pairs degrading
HIGH       n_red >= 1 AND n_coupled >= 2       RED driving coupled YELLOWs
CRITICAL   n_red >= 3, or n_red >= 2 + 2 pairs unknown state
```

Cascade reads `COUPLING_GRAPH`, never the count alone. Three unrelated
REDs and three coupled YELLOWs are different findings and must stay
different. `irreversible_active` is derived from the boundary's own
`notes` text containing `IRREVERSIBLE` -- a string scan, fragile, and
the fragility is the note to keep when editing notes.

## Regime extrapolation (adapters/generic.py)

```text
AssumptionBridge(model, {"derivation_regime": "holocene"})
    |
    +-- REGIMES keyed on CO2 ppm ranges:
    |     holocene / eemian / pliocene / last_glacial_maximum / anthropocene
    +-- no CO2 data              -> status UNKNOWN, never WITHIN_REGIME
    +-- deriv == current         -> WITHIN_REGIME
    +-- exceedance > 0.5         -> CRITICAL_EXTRAPOLATION, extra penalty
```

A model derived in one regime and run in another is the failure this
class exists to name. The check is **CO2-only**; that is a stated
narrowness, not an oversight to widen quietly.

## Vector risk (vector_risk.py)

Seven declared axes per assumption, four derived on init:

```text
declared   drift_rate  drift_acceleration  coupling_strength
           impact_magnitude  knowledge_dependence  time_to_red
           uncertainty_growth
derived    risk_velocity = |drift| * impact
           risk_acceleration = |accel| * impact
           blind_spot_size = 0.30*coupling/15 + 0.40*impact + 0.30*dependence
           urgency
```

The `/15` in `blind_spot_size` is a normalization against a maximum
coupling count of 15. Add a 16th coupling to any assumption and that
term exceeds its share silently. If `COUPLING_GRAPH` grows, this
constant is the thing to revisit.

## Adapters: the only input side

```text
GenericAdapter    flat {layer_key: value}; keys must match
                  AssumptionBoundary.layer_key. Reports available_keys /
                  unknown_keys / missing_keys -- use them; a silently
                  ignored key is an assumption nobody assessed.
EarthSystemsAdapter   LAYER_KEY_MAP from cascade_engine.run_all_layers()
                      output (github.com/JinnZ2/earth-systems-physics).
                      ScenarioAdapter subclasses it.
NOAAAdapter       9 fetchers (IERS, NOAA ESRL, RAPID, NSIDC, GRACE-FO,
                  SWPC, GTN-P, NCEP, ocean). Network.
                  EVERY fetch degrades to physics-based SIMULATION when
                  the feed is down, and labels itself `simulated` in
                  metadata. Preserve that label. A simulated value read
                  as a measured one is this repo's own failure mode,
                  committed inside the repo.
```

## Known README-vs-code drift (verified 2026-09-22, not repaired)

The README is the spec a reader follows; three of its examples do not
run. Repair the code, the README, or record which -- do not leave all
three standing.

```text
README                          tree
adapter.to_layer_states()       exists nowhere; use .fetch() / .full_report()
EarthSystemsMonitor             exists nowhere; the class is UniversalMonitor
GET /v1/layers                  the route is /v1/domains
```

`__init__.py` is a 10-line fragment whose first line reads
`# add to assumption_validator/__init__.py`. It re-exports `vector_risk`
only. `registry`, `monitors` and `api` are reachable only by their full
dotted path. That is a landing artifact, not a design.

## Conventions when editing

- A new assumption is an `AssumptionBoundary` in `REGISTRY` **plus** an
  entry in `COUPLING_GRAPH` if anything depends on it. A boundary with
  no couplings is invisible to cascade detection, which is a real state
  and should be intended rather than forgotten.
- `notes` states *what breaks when this breaks*, not what the parameter
  is. It is read by `detect_cascade_risk` for `IRREVERSIBLE`.
- Never widen a `green_range` because a current reading fell outside
  it. That is the failure the instrument is built to detect, performed
  on the instrument. Update the reading's interpretation, or record the
  boundary as contested; do not retune the envelope to preserve GREEN.
- Penalties, multipliers and the `/15` normalization are declared
  constants with no derivation on record. Changing one changes every
  downstream confidence. Say which and why in the commit message.
- No test exists to catch any of this. Until one does, a change here is
  reviewed by reading, so keep functions small and the refusals loud.

## Commands

```sh
# none of these run in this container (numpy/scipy/flask absent)
python3 -c "from assumption_validator.registry import full_report"   # from PARENT dir
python3 -m assumption_validator.api                                  # flask + SSE on :5000
curl -N http://localhost:5000/v1/stream
curl -X POST http://localhost:5000/v1/adjust -H 'Content-Type: application/json' \
     -d '{"prediction":{"value":15.2},"base_confidence":0.85,
          "derivation_regime":"holocene","assumptions":["amoc_strength"]}'
```

There is no test command. There is no lint command. There is no CI.
