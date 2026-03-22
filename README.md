# assumption_validator
CC0 — No Rights Reserved
github.com/JinnZ2/assumption_validator

A universal framework for detecting when the assumptions underlying
any model, prediction, or decision system are breaking down.

Not a model. Not a prediction tool.
A meta-layer that watches what models assume about the world
and tells you when those assumptions are no longer valid.

---

## The Problem

Every model — climate, grid, financial, medical, logistical — was built
on assumptions about how the world works. Most of those assumptions are
invisible. They are baked into training data, equation forms, parameter
values, and boundary conditions.

When the world changes faster than models update, the assumptions break.
The models keep running. The outputs look confident. The decisions based
on those outputs become systematically wrong.

This framework makes the assumptions visible, monitors them in real time,
and tells any system — AI or human — when it is operating outside the
regime where its equations are valid.

---

## What It Does


Any model or prediction system
│
▼
[ Assumption Validator ]
│
├── Reads current state of assumptions
├── Compares against known stability boundaries
├── Detects drift, acceleration, regime change
├── Computes blind spot size per assumption
├── Detects cascade risk across coupled assumptions
└── Returns: adjusted confidence + warnings + priorities
│
▼
Honest output: here is what I assumed,
here is how much has changed,
here is how much to trust me


---

## Architecture


assumption_validator/
├── registry.py       assumption boundaries — what is stable, what is not
├── monitors.py       live monitoring — drift, acceleration, time-to-red
├── vector_risk.py    blind spot analysis — where knowledge breaks fastest
├── api.py            REST + SSE — any system can query live validity
└── adapters/
├── earth_systems.py   connects to earth-systems-physics engine
├── generic.py         dict-based — works with any data source
└── noaa.py            live NOAA/IERS data feeds


---

## Install

```bash
git clone https://github.com/JinnZ2/assumption_validator
cd assumption_validator
pip install -r requirements.txt

Python 3.9+. Dependencies: numpy, scipy, flask, flask-cors, requests.


Quick Start
Check assumption validity from any data source

from assumption_validator.adapters.generic import GenericAdapter
from assumption_validator.registry import full_report

# Feed in current values from whatever source you have
adapter = GenericAdapter({
    "co2_ppm":          428.0,
    "amoc_sv":          14.5,
    "grid_inertia_s":   3.2,
    "lod_change_ms":    0.8,
    "permafrost_flux":  0.9,
})

states  = adapter.to_layer_states()
report  = full_report(states)

print(report["cascade"]["level"])
print(report["global_confidence_multiplier"])


Wrap any model with assumption awareness

from assumption_validator.adapters.generic import AssumptionBridge

bridge = AssumptionBridge(
    model = your_model,
    model_metadata = {
        "name":              "Your Model",
        "type":              "climate_projection",
        "training_year":     2022,
        "derivation_regime": "holocene",
    }
)

result = bridge.predict(input_data, base_confidence=0.85)

print(result["adjusted_confidence"])
print(result["warnings"])
print(result["overall_status"])


Connect to earth-systems-physics engine



from assumption_validator.adapters.earth_systems import EarthSystemsAdapter
from cascade_engine import run_all_layers, BASELINE

layer_states = run_all_layers(BASELINE)
adapter      = EarthSystemsAdapter(layer_states)
report       = adapter.full_report()


Connect to live NOAA/IERS data


from assumption_validator.adapters.noaa import NOAAAdapter

adapter = NOAAAdapter()
adapter.fetch_all()
report  = adapter.full_report()


Start the API

python -m assumption_validator.api


API Endpoints



GET  /health                         service liveness
GET  /v1/validity                    full assumption validity report
GET  /v1/validity/<id>               single assumption status + trend
GET  /v1/layers                      validity grouped by domain
POST /v1/adjust                      adjust any prediction's confidence
GET  /v1/cascade                     cascade risk level + history
GET  /v1/trends                      drift rates + time-to-red
GET  /v1/alerts                      drain alert queue
GET  /v1/blind_spots                 ranked blind spot analysis
GET  /v1/risk_vectors                3D risk space coordinates
GET  /v1/reassessment                AI reassessment priorities
GET  /v1/registry                    full assumption registry
GET  /v1/stream                      SSE live updates


Adjust a prediction

curl -X POST http://localhost:5000/v1/adjust \
  -H "Content-Type: application/json" \
  -d '{
    "prediction":        {"value": 15.2},
    "base_confidence":   0.85,
    "model_name":        "MyModel",
    "derivation_regime": "holocene",
    "assumptions":       ["co2_concentration", "amoc_strength"]
  }'


Subscribe to live updates

curl -N http://localhost:5000/v1/stream


Assumption Registry
Each assumption has explicit stability boundaries:

GREEN stable
YELLOW transition
RED regime change

Sample assumptions included out of the box:





Adding a new assumption:

from assumption_validator.registry import AssumptionBoundary, REGISTRY

REGISTRY["my_assumption"] = AssumptionBoundary(
    name            = "My Assumption",
    parameter       = "value",
    units           = "units",
    green_range     = (0, 100),
    yellow_range    = (100, 150),
    red_threshold   = 150,
    higher_is_worse = True,
    source_layer    = 0,
    layer_key       = "my_key",
    couplings       = ["other_assumption"],
    notes           = "What breaks when this breaks",
)


Vector Risk
Every assumption is a vector in risk space:

[drift_rate, coupling_strength, impact_magnitude,
 knowledge_dependence, time_to_red, uncertainty_growth]



The framework computes:
	∙	Blind spot size — what fraction of downstream knowledge breaks
	∙	Risk velocity — how fast the blind spot is growing
	∙	Risk acceleration — is it speeding up?
	∙	Time to RED — years at current rate to threshold
	∙	Cluster risk — assumptions that fail together




from assumption_validator.vector_risk import VectorRiskForecaster

f = VectorRiskForecaster()
f.update_from_monitor(monitor.current_report()["trends"])

report = f.full_report()
print(report["blind_spots"])
print(report["reassessment_priorities"])


Adapters
Generic (any data source)

from assumption_validator.adapters.generic import GenericAdapter

adapter = GenericAdapter({
    "co2_ppm": 428.0,
    "amoc_sv": 14.5,
    # any keys that match registry layer_key values
})
report = adapter.full_report()


earth-systems-physics

from assumption_validator.adapters.earth_systems import EarthSystemsAdapter


Reads directly from cascade_engine.run_all_layers() output.


NOAA live feeds

from assumption_validator.adapters.noaa import NOAAAdapter

Pulls from IERS, NOAA ESRL, RAPID array, NSIDC, GRACE-FO.



Cascade Detection
When multiple coupled assumptions degrade simultaneously
the system flags cascade risk:



MINIMAL  → no convergence
LOW      → broad degradation, monitor
MODERATE → coupled pairs degrading
HIGH     → one RED driving coupled YELLOWs
CRITICAL → multiple RED, unknown state


Regime Extrapolation
Models derived in one Earth regime used in another:

from assumption_validator.adapters.generic import AssumptionBridge

bridge = AssumptionBridge(model, {
    "derivation_regime": "holocene"  # where equations were derived
})

result = bridge.predict(x)
# result["regime_extrapolation"]["status"] will flag
# CRITICAL_EXTRAPOLATION if current conditions
# no longer match derivation regime


Regimes tracked: holocene, eemian, pliocene,
last_glacial_maximum, anthropocene.

Continuous Monitoring

from assumption_validator.monitors import EarthSystemsMonitor
from assumption_validator.vector_risk import VectorRiskForecaster

monitor    = EarthSystemsMonitor(poll_interval_s=60)
forecaster = VectorRiskForecaster()
monitor.start()

# Forecaster updates from live monitor data
forecaster.update_from_monitor(
    monitor.current_report()["trends"]
)

alerts = monitor.drain_alerts()


Connects To
	∙	earth-systems-physics — github.com/JinnZ2/earth-systems-physics
Full coupled differential equation stack for Earth physics.
assumption-validator reads its layer outputs directly.



What This Is Not
Not a climate model.
Not a prediction engine.
Not a policy tool.
Not a replacement for domain expertise.


What This Is
Epistemic infrastructure.
A system that knows what other systems assume
and watches those assumptions in real time.
So that when the world changes faster than models update,
something notices.



License
CC0 — No Rights Reserved.
Use it. Build on it. No permission needed.

