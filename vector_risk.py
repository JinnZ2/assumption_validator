# assumption_validator/vector_risk.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Vector-based risk forecasting.
# Each assumption is a vector in n-dimensional risk space.
# Computes blind spot size, drift velocity, acceleration,
# time-to-red, and cascade cluster risk.
# AI systems query this to know where their biggest blind spots are
# and which previous work needs reassessment.

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


# ─────────────────────────────────────────────
# RISK LEVEL
# ─────────────────────────────────────────────

class VectorRisk(Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"
    MINIMAL  = "MINIMAL"


# ─────────────────────────────────────────────
# ASSUMPTION VECTOR
# ─────────────────────────────────────────────

@dataclass
class AssumptionVector:
    """
    Risk vector for one assumption.
    All fields normalized to be comparable across assumptions.

    drift_rate          : rate of change per year in native units
    drift_acceleration  : second derivative — is drift speeding up?
    coupling_strength   : number of other assumptions this one drives
    impact_magnitude    : 0-1 fraction of downstream knowledge invalidated
    knowledge_dependence: 0-1 fraction of deployed models that use this
    time_to_red         : years at current drift rate to RED threshold
    uncertainty_growth  : fractional confidence loss per year of drift
    """
    name                : str
    drift_rate          : float
    drift_acceleration  : float
    coupling_strength   : int
    impact_magnitude    : float
    knowledge_dependence: float
    time_to_red         : float
    uncertainty_growth  : float

    # Derived on init
    risk_velocity       : float = field(init=False)
    risk_acceleration   : float = field(init=False)
    blind_spot_size     : float = field(init=False)
    urgency             : float = field(init=False)

    def __post_init__(self):
        self.risk_velocity    = abs(self.drift_rate) * self.impact_magnitude
        self.risk_acceleration= abs(self.drift_acceleration) * self.impact_magnitude
        self.blind_spot_size  = (
            self.coupling_strength   * 0.30 / 15 +   # normalized to max 15
            self.impact_magnitude    * 0.40 +
            self.knowledge_dependence* 0.30
        )
        self.urgency = 1.0 / max(0.1, self.time_to_red)

    def to_array(self) -> np.ndarray:
        """Normalized numpy array for vector math."""
        return np.array([
            abs(self.drift_rate)      / 10.0,   # rough normalization
            abs(self.drift_acceleration),
            self.coupling_strength    / 15.0,
            self.impact_magnitude,
            self.knowledge_dependence,
            1.0 / max(0.1, self.time_to_red),
            self.uncertainty_growth,
        ])

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def risk_level(self) -> VectorRisk:
        if self.time_to_red < 5  or self.blind_spot_size > 0.80:
            return VectorRisk.CRITICAL
        if self.time_to_red < 15 or self.blind_spot_size > 0.60:
            return VectorRisk.HIGH
        if self.time_to_red < 30 or self.blind_spot_size > 0.40:
            return VectorRisk.MODERATE
        if self.time_to_red < 50:
            return VectorRisk.LOW
        return VectorRisk.MINIMAL


# ─────────────────────────────────────────────
# DEFAULT ASSUMPTION VECTORS
# Values from current Earth system observations.
# Replace with live monitor data via update_from_monitor().
# ─────────────────────────────────────────────

DEFAULT_VECTORS: Dict[str, AssumptionVector] = {

    "rotation_rate": AssumptionVector(
        name                 = "Earth Rotation Rate",
        drift_rate           = 2.3e-9,
        drift_acceleration   = 1.0e-10,
        coupling_strength    = 15,
        impact_magnitude     = 0.95,
        knowledge_dependence = 0.90,
        time_to_red          = 30.0,
        uncertainty_growth   = 0.15,
    ),
    "co2_concentration": AssumptionVector(
        name                 = "Atmospheric CO₂",
        drift_rate           = 2.5,
        drift_acceleration   = 0.10,
        coupling_strength    = 12,
        impact_magnitude     = 0.90,
        knowledge_dependence = 0.85,
        time_to_red          = 9.0,
        uncertainty_growth   = 0.12,
    ),
    "amoc_strength": AssumptionVector(
        name                 = "AMOC Strength",
        drift_rate           = -0.4,
        drift_acceleration   = -0.01,
        coupling_strength    = 8,
        impact_magnitude     = 0.85,
        knowledge_dependence = 0.70,
        time_to_red          = 12.0,
        uncertainty_growth   = 0.20,
    ),
    "greenland_mass": AssumptionVector(
        name                 = "Greenland Ice Mass",
        drift_rate           = -27.0,
        drift_acceleration   = -3.0,
        coupling_strength    = 6,
        impact_magnitude     = 0.80,
        knowledge_dependence = 0.65,
        time_to_red          = 8.0,
        uncertainty_growth   = 0.18,
    ),
    "permafrost_carbon": AssumptionVector(
        name                 = "Permafrost Carbon Release",
        drift_rate           = 0.08,
        drift_acceleration   = 0.02,
        coupling_strength    = 6,
        impact_magnitude     = 0.75,
        knowledge_dependence = 0.60,
        time_to_red          = 6.0,
        uncertainty_growth   = 0.25,
    ),
    "arctic_sea_ice": AssumptionVector(
        name                 = "Arctic Sea Ice Extent",
        drift_rate           = -0.04,
        drift_acceleration   = -0.01,
        coupling_strength    = 7,
        impact_magnitude     = 0.70,
        knowledge_dependence = 0.55,
        time_to_red          = 38.0,
        uncertainty_growth   = 0.10,
    ),
    "jet_stream_shift": AssumptionVector(
        name                 = "Jet Stream Position",
        drift_rate           = 0.05,
        drift_acceleration   = 0.01,
        coupling_strength    = 9,
        impact_magnitude     = 0.85,
        knowledge_dependence = 0.75,
        time_to_red          = 30.0,
        uncertainty_growth   = 0.22,
    ),
    "ocean_ph": AssumptionVector(
        name                 = "Ocean pH",
        drift_rate           = -0.002,
        drift_acceleration   = -0.0003,
        coupling_strength    = 5,
        impact_magnitude     = 0.65,
        knowledge_dependence = 0.50,
        time_to_red          = 35.0,
        uncertainty_growth   = 0.08,
    ),
    "amazon_sink": AssumptionVector(
        name                 = "Amazon Carbon Sink",
        drift_rate           = -0.08,
        drift_acceleration   = -0.02,
        coupling_strength    = 4,
        impact_magnitude     = 0.60,
        knowledge_dependence = 0.45,
        time_to_red          = 6.0,
        uncertainty_growth   = 0.30,
    ),
    "grid_inertia": AssumptionVector(
        name                 = "Grid Inertia",
        drift_rate           = -0.30,
        drift_acceleration   = -0.05,
        coupling_strength    = 5,
        impact_magnitude     = 0.85,
        knowledge_dependence = 0.80,
        time_to_red          = 10.0,
        uncertainty_growth   = 0.15,
    ),
    "sea_level_rate": AssumptionVector(
        name                 = "Sea Level Rise Rate",
        drift_rate           = 0.03,
        drift_acceleration   = 0.005,
        coupling_strength    = 4,
        impact_magnitude     = 0.70,
        knowledge_dependence = 0.60,
        time_to_red          = 50.0,
        uncertainty_growth   = 0.12,
    ),
}


# ─────────────────────────────────────────────
# CLUSTER DEFINITIONS
# Assumptions that couple and fail together
# ─────────────────────────────────────────────

CLUSTERS: Dict[str, List[str]] = {
    "Climate Core":     ["rotation_rate", "co2_concentration", "amoc_strength"],
    "Cryosphere":       ["greenland_mass", "arctic_sea_ice", "sea_level_rate"],
    "Carbon Cycle":     ["co2_concentration", "permafrost_carbon", "amazon_sink", "ocean_ph"],
    "Energy Systems":   ["rotation_rate", "grid_inertia"],
    "Weather Systems":  ["rotation_rate", "jet_stream_shift", "amoc_strength"],
}

AFFECTED_DOMAINS: Dict[str, List[str]] = {
    "rotation_rate":    ["Climate", "Weather", "Navigation", "Grid", "Satellites"],
    "co2_concentration":["Climate", "Carbon Cycle", "Biology", "Ocean Chemistry"],
    "amoc_strength":    ["Climate", "Fisheries", "Agriculture", "Sea Level"],
    "greenland_mass":   ["Sea Level", "Ocean Circulation", "Navigation"],
    "permafrost_carbon":["Climate", "Carbon Cycle", "Infrastructure"],
    "arctic_sea_ice":   ["Climate", "Navigation", "Ecosystems"],
    "jet_stream_shift": ["Weather", "Agriculture", "Aviation"],
    "ocean_ph":         ["Marine Life", "Fisheries", "Carbon Cycle"],
    "amazon_sink":      ["Carbon Cycle", "Climate", "Biodiversity"],
    "grid_inertia":     ["Energy", "Infrastructure", "Economy"],
    "sea_level_rate":   ["Coastal Infrastructure", "Urban Planning", "Insurance"],
}


# ─────────────────────────────────────────────
# VECTOR RISK FORECASTER
# ─────────────────────────────────────────────

class VectorRiskForecaster:
    """
    Computes risk vectors, blind spots, cluster risk,
    time-to-red projections, and AI reassessment priorities.

    Usage
    -----
    forecaster = VectorRiskForecaster()
    forecaster.update_from_monitor(monitor_trends)   # optional live update
    report = forecaster.full_report()
    """

    def __init__(self, vectors: Dict[str, AssumptionVector] = None):
        self.vectors = dict(vectors or DEFAULT_VECTORS)

    # ── LIVE UPDATE ──────────────────────────────────────────────────

    def update_from_monitor(self, trends: Dict[str, Dict]):
        """
        Update vectors with live drift data from EarthSystemsMonitor.all_trends().
        trends : output of monitor.current_report()["trends"]
        """
        for aid, trend in trends.items():
            if aid not in self.vectors:
                continue
            v = self.vectors[aid]

            drift = trend.get("drift_rate_per_hour")
            accel = trend.get("acceleration")
            h2r   = trend.get("hours_to_red")

            if drift is not None:
                self.vectors[aid] = AssumptionVector(
                    name                 = v.name,
                    drift_rate           = drift * 8760,   # convert to per-year
                    drift_acceleration   = (accel * 8760) if accel else v.drift_acceleration,
                    coupling_strength    = v.coupling_strength,
                    impact_magnitude     = v.impact_magnitude,
                    knowledge_dependence = v.knowledge_dependence,
                    time_to_red          = (h2r / 8760) if h2r else v.time_to_red,
                    uncertainty_growth   = v.uncertainty_growth,
                )

    # ── CORE METRICS ─────────────────────────────────────────────────

    def metrics(self) -> Dict[str, Dict]:
        """Full metrics for every assumption."""
        out = {}
        for aid, v in self.vectors.items():
            out[aid] = {
                "name":                aid,
                "label":               v.name,
                "drift_rate":          v.drift_rate,
                "drift_acceleration":  v.drift_acceleration,
                "coupling_strength":   v.coupling_strength,
                "impact_magnitude":    v.impact_magnitude,
                "knowledge_dependence":v.knowledge_dependence,
                "time_to_red":         v.time_to_red,
                "uncertainty_growth":  v.uncertainty_growth,
                "risk_velocity":       v.risk_velocity,
                "risk_acceleration":   v.risk_acceleration,
                "blind_spot_size":     v.blind_spot_size,
                "urgency":             v.urgency,
                "vector_magnitude":    v.magnitude(),
                "risk_level":          v.risk_level().value,
                "domains":             AFFECTED_DOMAINS.get(aid, []),
            }
        return out

    # ── BLIND SPOTS ──────────────────────────────────────────────────

    def blind_spots(self, top_n: int = 5) -> List[Dict]:
        """
        Assumptions ranked by blind spot size.
        These are the ones that invalidate the most downstream knowledge.
        """
        m = self.metrics()
        ranked = sorted(
            m.values(),
            key     = lambda x: x["blind_spot_size"],
            reverse = True,
        )
        return ranked[:top_n]

    # ── CLUSTER RISK ─────────────────────────────────────────────────

    def cluster_risk(self) -> List[Dict]:
        """
        Risk for each coupled cluster.
        Clusters that fail together create compound blind spots.
        """
        m = self.metrics()
        results = []

        for cluster_name, members in CLUSTERS.items():
            present = [k for k in members if k in self.vectors]
            if not present:
                continue

            total_bs  = sum(m[k]["blind_spot_size"] for k in present)
            avg_bs    = min(1.0, total_bs / len(present))
            min_t2r   = min(m[k]["time_to_red"] for k in present)
            max_urgency = max(m[k]["urgency"] for k in present)

            rl = VectorRisk.MINIMAL
            if min_t2r < 5  or avg_bs > 0.80: rl = VectorRisk.CRITICAL
            elif min_t2r < 15 or avg_bs > 0.60: rl = VectorRisk.HIGH
            elif min_t2r < 30 or avg_bs > 0.40: rl = VectorRisk.MODERATE
            elif min_t2r < 50: rl = VectorRisk.LOW

            results.append({
                "cluster":              cluster_name,
                "members":              present,
                "average_blind_spot":   avg_bs,
                "earliest_time_to_red": min_t2r,
                "max_urgency":          max_urgency,
                "risk_level":           rl.value,
            })

        return sorted(results, key=lambda x: x["average_blind_spot"], reverse=True)

    # ── VECTOR MAGNITUDE RANKING ─────────────────────────────────────

    def magnitude_ranking(self) -> List[Dict]:
        """
        All assumptions ranked by total risk vector magnitude.
        Combines all dimensions into a single comparable number.
        """
        m = self.metrics()
        ranked = sorted(
            m.values(),
            key     = lambda x: x["vector_magnitude"],
            reverse = True,
        )
        return ranked

    # ── TIME PROJECTION ──────────────────────────────────────────────

    def project(self, years: int = 20) -> Dict[str, List[Dict]]:
        """
        Project risk evolution for each assumption over N years.
        Uses current drift rate + acceleration.
        """
        out = {}
        for aid, v in self.vectors.items():
            timeline = []
            for yr in range(years + 1):
                # Quadratic drift projection
                drift_at_yr  = v.drift_rate * yr + 0.5 * v.drift_acceleration * yr ** 2
                remaining    = max(0.0, v.time_to_red - yr)
                bs_growth    = min(1.0, v.blind_spot_size * (1 + 0.05 * yr))

                if   remaining < 5  or bs_growth > 0.80: rl = VectorRisk.CRITICAL
                elif remaining < 15 or bs_growth > 0.60: rl = VectorRisk.HIGH
                elif remaining < 30 or bs_growth > 0.40: rl = VectorRisk.MODERATE
                elif remaining < 50: rl = VectorRisk.LOW
                else: rl = VectorRisk.MINIMAL

                timeline.append({
                    "year":             yr,
                    "projected_drift":  drift_at_yr,
                    "remaining_t2r":    remaining,
                    "blind_spot":       bs_growth,
                    "risk_level":       rl.value,
                })
            out[aid] = timeline
        return out

    # ── 3D RISK SPACE ────────────────────────────────────────────────

    def risk_space_3d(self) -> Dict[str, Dict]:
        """
        3D coordinates for each assumption in risk space.
        x = normalized drift rate
        y = blind spot size
        z = urgency (1 / time_to_red)
        Color encodes risk level.
        """
        colors = {
            VectorRisk.CRITICAL: "#FF0000",
            VectorRisk.HIGH:     "#FF6600",
            VectorRisk.MODERATE: "#FFCC00",
            VectorRisk.LOW:      "#33CC33",
            VectorRisk.MINIMAL:  "#00CC66",
        }
        out = {}
        for aid, v in self.vectors.items():
            rl = v.risk_level()
            out[aid] = {
                "label":      v.name,
                "x":          abs(v.drift_rate) / 10.0,
                "y":          v.blind_spot_size,
                "z":          v.urgency,
                "magnitude":  v.magnitude(),
                "risk_level": rl.value,
                "color":      colors[rl],
            }
        return out

    # ── REASSESSMENT PRIORITIES ──────────────────────────────────────

    def reassessment_priorities(self) -> List[Dict]:
        """
        Ordered list of assumptions AI models should reassess first.
        Priority 1 = immediate. Priority 2 = 30 days. Priority 3 = cluster-level.
        """
        bs    = self.blind_spots(top_n=len(self.vectors))
        clust = self.cluster_risk()
        out   = []

        for item in bs:
            rl = item["risk_level"]
            if rl == VectorRisk.CRITICAL.value:
                out.append({
                    "priority":   1,
                    "type":       "ASSUMPTION",
                    "id":         item["name"],
                    "label":      item["label"],
                    "reason":     f"CRITICAL — {item['blind_spot_size']:.0%} blind spot, {item['time_to_red']:.0f} yr to RED",
                    "action":     "IMMEDIATE REASSESSMENT",
                    "domains":    item["domains"],
                })
            elif rl == VectorRisk.HIGH.value:
                out.append({
                    "priority":   2,
                    "type":       "ASSUMPTION",
                    "id":         item["name"],
                    "label":      item["label"],
                    "reason":     f"HIGH — {item['blind_spot_size']:.0%} blind spot, {item['time_to_red']:.0f} yr to RED",
                    "action":     "REASSESS WITHIN 30 DAYS",
                    "domains":    item["domains"],
                })

        for c in clust[:3]:
            if c["risk_level"] in (VectorRisk.CRITICAL.value, VectorRisk.HIGH.value):
                out.append({
                    "priority":   3,
                    "type":       "CLUSTER",
                    "id":         c["cluster"],
                    "label":      f"Cluster: {c['cluster']}",
                    "reason":     f"{c['risk_level']} cluster — {c['average_blind_spot']:.0%} avg blind spot",
                    "action":     "REASSESS CROSS-DOMAIN MODELS",
                    "domains":    c["members"],
                })

        return out

    # ── FULL REPORT ──────────────────────────────────────────────────

    def full_report(self) -> Dict:
        """Single call — full vector risk analysis."""
        m      = self.metrics()
        bs     = self.blind_spots(top_n=5)
        clust  = self.cluster_risk()
        ranked = self.magnitude_ranking()
        prio   = self.reassessment_priorities()
        space  = self.risk_space_3d()

        return {
            "metrics":                  m,
            "blind_spots":              bs,
            "clusters":                 clust,
            "magnitude_ranking":        ranked,
            "reassessment_priorities":  prio,
            "risk_space_3d":            space,
            "summary": {
                "critical_count":   sum(1 for v in m.values() if v["risk_level"] == "CRITICAL"),
                "high_count":       sum(1 for v in m.values() if v["risk_level"] == "HIGH"),
                "top_blind_spot":   bs[0]["label"] if bs else None,
                "fastest_to_red":   min(m.values(), key=lambda x: x["time_to_red"])["label"],
                "highest_magnitude":ranked[0]["label"] if ranked else None,
            },
        }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    f = VectorRiskForecaster()
    r = f.full_report()

    print("=" * 56)
    print("VECTOR RISK FORECASTER")
    print("=" * 56)
    s = r["summary"]
    print(f"  CRITICAL  : {s['critical_count']}")
    print(f"  HIGH      : {s['high_count']}")
    print(f"  Top blind spot     : {s['top_blind_spot']}")
    print(f"  Fastest to RED     : {s['fastest_to_red']}")
    print(f"  Highest magnitude  : {s['highest_magnitude']}")

    print("\nBLIND SPOTS (top 5):")
    for b in r["blind_spots"]:
        print(f"  [{b['risk_level']:<8}]  {b['label']:<35}  "
              f"blind={b['blind_spot_size']:.0%}  t2r={b['time_to_red']:.0f}yr")

    print("\nCLUSTER RISK:")
    for c in r["clusters"]:
        print(f"  [{c['risk_level']:<8}]  {c['cluster']:<20}  "
              f"avg_blind={c['average_blind_spot']:.0%}  "
              f"earliest_t2r={c['earliest_time_to_red']:.0f}yr")

    print("\nREASSESSMENT PRIORITIES:")
    for p in r["reassessment_priorities"]:
        print(f"  [P{p['priority']}] {p['label'][:40]:<40}  {p['action']}")
