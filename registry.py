# assumption_validator/registry.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Universal assumption registry.
# No dependency on earth-systems-physics or any specific data source.
# Define any assumption with stability boundaries.
# Feed in current values from any adapter.
# Get back GREEN/YELLOW/RED with confidence penalties.

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


# ─────────────────────────────────────────────
# RISK LEVEL
# ─────────────────────────────────────────────

class RiskLevel(Enum):
    GREEN   = "GREEN"    # stable — equations valid
    YELLOW  = "YELLOW"   # transition — equations degrading
    RED     = "RED"      # regime changed — equations invalid
    UNKNOWN = "UNKNOWN"  # no data


# ─────────────────────────────────────────────
# ASSUMPTION BOUNDARY
# ─────────────────────────────────────────────

@dataclass
class AssumptionBoundary:
    """
    Defines the stability envelope for one assumption.

    name            : human-readable label
    parameter       : what is being measured
    units           : units of measurement
    green_range     : (min, max) — fully valid
    yellow_range    : (min, max) — transition, degrading
    red_threshold   : value beyond which regime has changed
    higher_is_worse : True if increasing value = increasing risk
    domain          : which field this assumption belongs to
    layer_key       : key used by adapters to look up current value
    couplings       : list of other assumption IDs this one drives
    rate_of_change  : observed rate in units/year (+ = worsening)
    notes           : what breaks when this assumption breaks
    """
    name            : str
    parameter       : str
    units           : str
    green_range     : Tuple[float, float]
    yellow_range    : Tuple[float, float]
    red_threshold   : float
    higher_is_worse : bool
    domain          : str        = "general"
    layer_key       : str        = ""
    couplings       : List[str]  = field(default_factory=list)
    rate_of_change  : float      = 0.0
    notes           : str        = ""

    def assess(self, value: float) -> Tuple[RiskLevel, float, float]:
        """
        Assess a value against stability boundaries.
        Returns (RiskLevel, confidence_penalty 0-1, proximity_to_red 0-1).
        """
        if value is None:
            return (RiskLevel.UNKNOWN, 0.0, 0.0)

        v = float(value)

        if self.higher_is_worse:
            if v <= self.green_range[1]:
                return (RiskLevel.GREEN, 0.0, 0.0)
            elif v <= self.yellow_range[1]:
                span     = self.yellow_range[1] - self.green_range[1] + 1e-30
                progress = (v - self.green_range[1]) / span
                penalty  = 0.3 * progress
                prox_span = self.red_threshold - self.yellow_range[0] + 1e-30
                proximity = (v - self.yellow_range[0]) / prox_span
                return (RiskLevel.YELLOW, penalty, min(1.0, proximity))
            else:
                excess  = (v - self.red_threshold) / (self.red_threshold + 1e-30)
                penalty = min(1.0, 0.8 + 0.2 * excess)
                return (RiskLevel.RED, penalty, 1.0)
        else:
            # lower is worse
            if v >= self.green_range[0]:
                return (RiskLevel.GREEN, 0.0, 0.0)
            elif v >= self.yellow_range[0]:
                span     = self.green_range[0] - self.yellow_range[0] + 1e-30
                progress = (self.green_range[0] - v) / span
                penalty  = 0.3 * progress
                prox_span = self.yellow_range[1] - self.red_threshold + 1e-30
                proximity = (self.yellow_range[1] - v) / prox_span
                return (RiskLevel.YELLOW, penalty, min(1.0, proximity))
            else:
                deficit = (self.red_threshold - v) / (abs(self.red_threshold) + 1e-30)
                penalty = min(1.0, 0.8 + 0.2 * deficit)
                return (RiskLevel.RED, penalty, 1.0)


# ─────────────────────────────────────────────
# UNIVERSAL REGISTRY
# Covers Earth systems, grid, carbon, climate.
# Any domain can add assumptions via register().
# layer_key is what adapters use to look up values.
# ─────────────────────────────────────────────

REGISTRY: Dict[str, AssumptionBoundary] = {

    # ── EARTH ROTATION ───────────────────────────────────────────────
    "rotation_rate": AssumptionBoundary(
        name            = "Earth Rotation Rate",
        parameter       = "fractional_change",
        units           = "dimensionless",
        green_range     = (0.0, 1e-9),
        yellow_range    = (1e-9, 1e-8),
        red_threshold   = 1e-7,
        higher_is_worse = True,
        domain          = "geophysics",
        layer_key       = "omega_change_rads",
        couplings       = ["coriolis", "grid_frequency", "jet_stream_shift", "lod_change"],
        rate_of_change  = 2e-10,
        notes           = "Affects all Coriolis-dependent equations, GPS, grid frequency reference",
    ),

    "lod_change": AssumptionBoundary(
        name            = "Length of Day Change",
        parameter       = "LOD_change_ms",
        units           = "milliseconds",
        green_range     = (-0.5, 0.5),
        yellow_range    = (0.5, 2.0),
        red_threshold   = 5.0,
        higher_is_worse = True,
        domain          = "geophysics",
        layer_key       = "LOD_change_ms",
        couplings       = ["rotation_rate", "coriolis", "grid_frequency"],
        rate_of_change  = 0.02,
        notes           = "Greenland melt already measurable in LOD — GPS precision affected",
    ),

    # ── ATMOSPHERE ───────────────────────────────────────────────────
    "co2_concentration": AssumptionBoundary(
        name            = "Atmospheric CO₂",
        parameter       = "concentration",
        units           = "ppm",
        green_range     = (280.0, 350.0),
        yellow_range    = (350.0, 450.0),
        red_threshold   = 450.0,
        higher_is_worse = True,
        domain          = "atmosphere",
        layer_key       = "co2_ppm",
        couplings       = ["ocean_ph", "permafrost_carbon", "amazon_sink",
                           "ghg_forcing", "marine_productivity"],
        rate_of_change  = 2.5,
        notes           = "All radiative forcing equations derived below 350 ppm",
    ),

    "ghg_forcing": AssumptionBoundary(
        name            = "GHG Radiative Forcing",
        parameter       = "forcing",
        units           = "W/m²",
        green_range     = (0.0, 1.0),
        yellow_range    = (1.0, 4.5),
        red_threshold   = 4.5,
        higher_is_worse = True,
        domain          = "atmosphere",
        layer_key       = "GHG_forcing_Wm2",
        couplings       = ["sst", "permafrost_carbon", "hadley_extent"],
        rate_of_change  = 0.05,
        notes           = "4.5 W/m² = 2xCO2 — nonlinear feedbacks dominate above",
    ),

    "jet_stream_shift": AssumptionBoundary(
        name            = "Jet Stream Poleward Shift",
        parameter       = "shift",
        units           = "degrees latitude",
        green_range     = (0.0, 1.0),
        yellow_range    = (1.0, 3.0),
        red_threshold   = 3.0,
        higher_is_worse = True,
        domain          = "atmosphere",
        layer_key       = "jet_shift_deg",
        couplings       = ["crop_yield", "blocking_frequency", "hadley_extent"],
        rate_of_change  = 0.05,
        notes           = "Weakening meridional gradient — weather persistence, extremes",
    ),

    "hadley_extent": AssumptionBoundary(
        name            = "Hadley Cell Extent",
        parameter       = "poleward_boundary",
        units           = "degrees latitude",
        green_range     = (25.0, 32.0),
        yellow_range    = (32.0, 38.0),
        red_threshold   = 38.0,
        higher_is_worse = True,
        domain          = "atmosphere",
        layer_key       = "hadley_extent_deg",
        couplings       = ["jet_stream_shift", "drought_risk", "crop_yield"],
        rate_of_change  = 0.3,
        notes           = "Subtropical dry belt shifts poleward — desertification cascade",
    ),

    # ── OCEAN ─────────────────────────────────────────────────────────
    "amoc_strength": AssumptionBoundary(
        name            = "AMOC Strength",
        parameter       = "transport",
        units           = "Sv",
        green_range     = (15.0, 20.0),
        yellow_range    = (10.0, 15.0),
        red_threshold   = 10.0,
        higher_is_worse = False,
        domain          = "ocean",
        layer_key       = "amoc_sv",
        couplings       = ["jet_stream_shift", "amazon_sink", "sst",
                           "sea_level_ne_us", "sahel_rainfall"],
        rate_of_change  = -0.4,
        notes           = "IRREVERSIBLE threshold — collapse reorganizes North Atlantic climate",
    ),

    "ocean_ph": AssumptionBoundary(
        name            = "Ocean pH",
        parameter       = "pH",
        units           = "pH units",
        green_range     = (8.05, 8.20),
        yellow_range    = (7.95, 8.05),
        red_threshold   = 7.95,
        higher_is_worse = False,
        domain          = "ocean",
        layer_key       = "ocean_ph",
        couplings       = ["coral_dissolution", "marine_productivity", "carbon_cycle"],
        rate_of_change  = -0.002,
        notes           = "Aragonite saturation threshold for coral dissolution",
    ),

    "marine_productivity": AssumptionBoundary(
        name            = "Marine Productivity Change",
        parameter       = "fractional_change",
        units           = "fraction",
        green_range     = (-0.05, 0.05),
        yellow_range    = (-0.20, -0.05),
        red_threshold   = -0.20,
        higher_is_worse = False,
        domain          = "ocean",
        layer_key       = "marine_productivity_change_frac",
        couplings       = ["carbon_cycle", "food_web", "ocean_ph"],
        rate_of_change  = -0.005,
        notes           = "Ocean produces 50% of Earth oxygen — stratification reducing",
    ),

    "sst_anomaly": AssumptionBoundary(
        name            = "Sea Surface Temperature Anomaly",
        parameter       = "anomaly",
        units           = "K",
        green_range     = (0.0, 0.5),
        yellow_range    = (0.5, 1.5),
        red_threshold   = 2.0,
        higher_is_worse = True,
        domain          = "ocean",
        layer_key       = "sst_anomaly_K",
        couplings       = ["amoc_strength", "marine_productivity", "hurricane_intensity"],
        rate_of_change  = 0.02,
        notes           = "SST drives ENSO, hurricane intensity, marine heatwaves",
    ),

    # ── CRYOSPHERE ────────────────────────────────────────────────────
    "greenland_mass": AssumptionBoundary(
        name            = "Greenland Ice Mass Balance",
        parameter       = "mass_balance",
        units           = "Gt/yr",
        green_range     = (-50.0, 50.0),
        yellow_range    = (-150.0, -50.0),
        red_threshold   = -150.0,
        higher_is_worse = False,
        domain          = "cryosphere",
        layer_key       = "greenland_mass_gt_yr",
        couplings       = ["amoc_strength", "sea_level_rate", "rotation_rate", "lod_change"],
        rate_of_change  = -25.0,
        notes           = "Freshwater pulse into North Atlantic — AMOC freshwater trigger",
    ),

    "antarctica_mass": AssumptionBoundary(
        name            = "Antarctica Ice Mass Balance",
        parameter       = "mass_balance",
        units           = "Gt/yr",
        green_range     = (-50.0, 50.0),
        yellow_range    = (-150.0, -50.0),
        red_threshold   = -150.0,
        higher_is_worse = False,
        domain          = "cryosphere",
        layer_key       = "antarctica_mass_gt_yr",
        couplings       = ["sea_level_rate", "rotation_rate"],
        rate_of_change  = -15.0,
        notes           = "Marine ice sheet instability — potential runaway above threshold",
    ),

    "arctic_sea_ice": AssumptionBoundary(
        name            = "Arctic Sea Ice Extent",
        parameter       = "september_extent",
        units           = "million km²",
        green_range     = (4.0, 8.0),
        yellow_range    = (2.0, 4.0),
        red_threshold   = 2.0,
        higher_is_worse = False,
        domain          = "cryosphere",
        layer_key       = "arctic_ice_mkm2",
        couplings       = ["albedo_feedback", "arctic_amplification", "jet_stream_shift"],
        rate_of_change  = -0.04,
        notes           = "Ice-albedo feedback — loss is self-amplifying",
    ),

    "sea_level_rate": AssumptionBoundary(
        name            = "Sea Level Rise Rate",
        parameter       = "rate",
        units           = "mm/yr",
        green_range     = (0.0, 2.0),
        yellow_range    = (2.0, 6.0),
        red_threshold   = 6.0,
        higher_is_worse = True,
        domain          = "cryosphere",
        layer_key       = "slr_mm_yr",
        couplings       = ["coastal_infrastructure", "fault_stress", "greenland_mass"],
        rate_of_change  = 0.03,
        notes           = "Rate > 6 mm/yr exceeds adaptation capacity for most coastal cities",
    ),

    # ── BIOSPHERE ─────────────────────────────────────────────────────
    "permafrost_carbon": AssumptionBoundary(
        name            = "Permafrost Carbon Flux",
        parameter       = "carbon_release",
        units           = "GtC/yr",
        green_range     = (0.0, 0.5),
        yellow_range    = (0.5, 1.5),
        red_threshold   = 1.5,
        higher_is_worse = True,
        domain          = "biosphere",
        layer_key       = "permafrost_CO2_GtC_yr",
        couplings       = ["co2_concentration", "permafrost_ch4", "arctic_amplification"],
        rate_of_change  = 0.08,
        notes           = "IRREVERSIBLE — self-amplifying, 1.5T tonnes C frozen",
    ),

    "permafrost_ch4": AssumptionBoundary(
        name            = "Permafrost Methane Flux",
        parameter       = "ch4_release",
        units           = "GtC/yr",
        green_range     = (0.0, 0.05),
        yellow_range    = (0.05, 0.20),
        red_threshold   = 0.20,
        higher_is_worse = True,
        domain          = "biosphere",
        layer_key       = "permafrost_CH4_GtC_yr",
        couplings       = ["ghg_forcing", "permafrost_carbon"],
        rate_of_change  = 0.01,
        notes           = "CH4 GWP 28x CO2 — small flux, large forcing",
    ),

    "amazon_sink": AssumptionBoundary(
        name            = "Amazon Carbon Sink",
        parameter       = "net_flux",
        units           = "GtC/yr",
        green_range     = (0.5, 1.0),
        yellow_range    = (0.0, 0.5),
        red_threshold   = 0.0,
        higher_is_worse = False,
        domain          = "biosphere",
        layer_key       = "amazon_sink_GtC_yr",
        couplings       = ["co2_concentration", "drought_risk", "fire_emissions"],
        rate_of_change  = -0.08,
        notes           = "~20% cleared now, threshold ~25-40% — forest-savanna transition",
    ),

    "amazon_tipping": AssumptionBoundary(
        name            = "Amazon Tipping Proximity",
        parameter       = "tipping_proximity",
        units           = "fraction (0=stable, 1=tipped)",
        green_range     = (0.0, 0.4),
        yellow_range    = (0.4, 0.7),
        red_threshold   = 0.8,
        higher_is_worse = True,
        domain          = "biosphere",
        layer_key       = "amazon_tipping_proximity",
        couplings       = ["amazon_sink", "ghg_forcing", "sa_precipitation"],
        rate_of_change  = 0.03,
        notes           = "IRREVERSIBLE above tipping point — 150 GtC release cascade",
    ),

    "nep_carbon_sink": AssumptionBoundary(
        name            = "Terrestrial Net Ecosystem Productivity",
        parameter       = "nep_gC_m2_day",
        units           = "gC/m²/day",
        green_range     = (0.5, 5.0),
        yellow_range    = (0.0, 0.5),
        red_threshold   = 0.0,
        higher_is_worse = False,
        domain          = "biosphere",
        layer_key       = "NEP_gC_m2_day",
        couplings       = ["co2_concentration", "permafrost_carbon", "amazon_tipping"],
        rate_of_change  = -0.02,
        notes           = "Ecosystem flip to carbon source initiates self-amplifying loop",
    ),

    "planetary_boundaries": AssumptionBoundary(
        name            = "Planetary Boundaries Crossed",
        parameter       = "count",
        units           = "of 9",
        green_range     = (0.0, 2.0),
        yellow_range    = (2.0, 5.0),
        red_threshold   = 6.0,
        higher_is_worse = True,
        domain          = "biosphere",
        layer_key       = "planetary_boundaries_crossed",
        couplings       = ["all"],
        rate_of_change  = 0.1,
        notes           = "Interaction effects between crossed boundaries unquantified",
    ),

    # ── LITHOSPHERE ───────────────────────────────────────────────────
    "polar_drift": AssumptionBoundary(
        name            = "Polar Drift Rate",
        parameter       = "drift_rate",
        units           = "degrees/yr",
        green_range     = (0.0, 0.005),
        yellow_range    = (0.005, 0.02),
        red_threshold   = 0.05,
        higher_is_worse = True,
        domain          = "lithosphere",
        layer_key       = "polar_drift_deg_yr",
        couplings       = ["rotation_rate", "crustal_stress"],
        rate_of_change  = 0.001,
        notes           = "GPS already detects polar shift matching ice melt signal",
    ),

    "volcanic_enhancement": AssumptionBoundary(
        name            = "Volcanic Activity Enhancement",
        parameter       = "multiplier",
        units           = "x baseline",
        green_range     = (0.8, 1.2),
        yellow_range    = (1.2, 2.0),
        red_threshold   = 3.0,
        higher_is_worse = True,
        domain          = "lithosphere",
        layer_key       = "volcanic_enhancement",
        couplings       = ["ghg_forcing", "marine_productivity", "aerosol_aod"],
        rate_of_change  = 0.02,
        notes           = "Ice unloading enhances volcanism — Iceland 30-50x post-glacial",
    ),

    # ── IONOSPHERE / ELECTROMAGNETICS ────────────────────────────────
    "gic_risk": AssumptionBoundary(
        name            = "Geomagnetically Induced Current Risk",
        parameter       = "gic_current",
        units           = "A/phase",
        green_range     = (0.0, 10.0),
        yellow_range    = (10.0, 50.0),
        red_threshold   = 50.0,
        higher_is_worse = True,
        domain          = "electromagnetics",
        layer_key       = "gic_current_A",
        couplings       = ["grid_inertia", "transformer_health", "grid_frequency"],
        rate_of_change  = 0.0,   # solar-cycle dependent
        notes           = "Solar max 2025-2026 — transformer saturation, grid cascade",
    ),

    "schumann_shift": AssumptionBoundary(
        name            = "Schumann Resonance Shift",
        parameter       = "frequency_shift",
        units           = "Hz",
        green_range     = (-0.05, 0.05),
        yellow_range    = (-0.20, 0.20),
        red_threshold   = 0.5,
        higher_is_worse = True,
        domain          = "electromagnetics",
        layer_key       = "schumann_f1_shift_hz",
        couplings       = ["ionosphere_height", "lightning_activity"],
        notes           = "Encodes ionosphere height change from thermosphere warming",
    ),

    # ── ENERGY GRID ───────────────────────────────────────────────────
    "grid_inertia": AssumptionBoundary(
        name            = "Grid Inertia",
        parameter       = "inertia",
        units           = "seconds",
        green_range     = (5.0, 10.0),
        yellow_range    = (2.0, 5.0),
        red_threshold   = 2.0,
        higher_is_worse = False,
        domain          = "energy",
        layer_key       = "grid_inertia_s",
        couplings       = ["grid_frequency", "rotation_rate", "gic_risk"],
        rate_of_change  = -0.3,
        notes           = "Declining as rotating plant retires — cascade threshold at 2s",
    ),

    "grid_frequency_deviation": AssumptionBoundary(
        name            = "Grid Frequency Deviation",
        parameter       = "deviation",
        units           = "Hz",
        green_range     = (0.0, 0.05),
        yellow_range    = (0.05, 0.20),
        red_threshold   = 0.50,
        higher_is_worse = True,
        domain          = "energy",
        layer_key       = "grid_freq_deviation_hz",
        couplings       = ["grid_inertia", "rotation_rate", "load_balance"],
        rate_of_change  = 0.01,
        notes           = "Frequency > 0.5 Hz deviation triggers protection systems",
    ),

}


# ─────────────────────────────────────────────
# COUPLING GRAPH
# Which assumption failures propagate to which others
# ─────────────────────────────────────────────

COUPLING_GRAPH: Dict[str, List[str]] = {
    "rotation_rate":        ["jet_stream_shift", "lod_change", "grid_frequency_deviation"],
    "lod_change":           ["rotation_rate", "grid_frequency_deviation"],
    "co2_concentration":    ["ocean_ph", "permafrost_carbon", "ghg_forcing",
                             "marine_productivity", "amazon_tipping"],
    "ghg_forcing":          ["sst_anomaly", "permafrost_carbon", "hadley_extent"],
    "amoc_strength":        ["jet_stream_shift", "amazon_sink", "sst_anomaly"],
    "greenland_mass":       ["amoc_strength", "sea_level_rate", "lod_change"],
    "antarctica_mass":      ["sea_level_rate", "rotation_rate"],
    "arctic_sea_ice":       ["jet_stream_shift", "permafrost_carbon"],
    "permafrost_carbon":    ["co2_concentration", "permafrost_ch4", "ghg_forcing"],
    "amazon_tipping":       ["amazon_sink", "ghg_forcing"],
    "nep_carbon_sink":      ["co2_concentration", "permafrost_carbon", "amazon_tipping"],
    "grid_inertia":         ["grid_frequency_deviation", "gic_risk"],
    "gic_risk":             ["grid_inertia", "grid_frequency_deviation"],
    "volcanic_enhancement": ["ghg_forcing", "marine_productivity"],
}


# ─────────────────────────────────────────────
# REGISTRATION HELPER
# ─────────────────────────────────────────────

def register(assumption_id: str, boundary: AssumptionBoundary):
    """Add or replace an assumption in the registry."""
    REGISTRY[assumption_id] = boundary


# ─────────────────────────────────────────────
# ASSESSMENT FUNCTIONS
# ─────────────────────────────────────────────

def assess_values(values: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Assess a flat dict of {layer_key: value} against the registry.
    This is what adapters call.

    values : dict keyed by layer_key (matches AssumptionBoundary.layer_key)
    returns: dict keyed by assumption_id with full assessment
    """
    # Build reverse lookup: layer_key -> assumption_id
    key_to_id = {b.layer_key: aid for aid, b in REGISTRY.items() if b.layer_key}

    results = {}
    for layer_key, value in values.items():
        aid = key_to_id.get(layer_key)
        if aid is None:
            continue
        boundary = REGISTRY[aid]
        _assess_one(aid, boundary, value, results)

    # Also check any registry entries not covered
    covered = set(results.keys())
    for aid, boundary in REGISTRY.items():
        if aid not in covered:
            results[aid] = {
                "id":      aid,
                "name":    boundary.name,
                "status":  RiskLevel.UNKNOWN.value,
                "value":   None,
                "message": "No data provided",
            }

    return results


def assess_from_layer_states(layer_states: Dict[int, Dict]) -> Dict[str, Dict]:
    """
    Assess from earth-systems-physics layer_states structure.
    Flattens all layer outputs and runs assess_values.
    """
    flat = {}
    for layer_output in layer_states.values():
        if isinstance(layer_output, dict):
            flat.update(layer_output)
    return assess_values(flat)


def _assess_one(
    aid      : str,
    boundary : AssumptionBoundary,
    value    : Any,
    results  : Dict,
):
    """Assess one assumption and write result into results dict."""
    if isinstance(value, bool):
        numeric = 1.0 if value else 0.0
    elif isinstance(value, str):
        results[aid] = {
            "id":     aid,
            "name":   boundary.name,
            "status": RiskLevel.UNKNOWN.value,
            "value":  value,
            "message": value,
        }
        return
    elif value is None:
        results[aid] = {
            "id":     aid,
            "name":   boundary.name,
            "status": RiskLevel.UNKNOWN.value,
            "value":  None,
            "message": "No data",
        }
        return
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            results[aid] = {
                "id":     aid,
                "name":   boundary.name,
                "status": RiskLevel.UNKNOWN.value,
                "value":  value,
                "message": f"Cannot convert {value!r} to float",
            }
            return

    risk, penalty, proximity = boundary.assess(numeric)

    results[aid] = {
        "id":                 aid,
        "name":               boundary.name,
        "domain":             boundary.domain,
        "status":             risk.value,
        "value":              numeric,
        "units":              boundary.units,
        "confidence_penalty": penalty,
        "proximity_to_red":   proximity,
        "green_range":        boundary.green_range,
        "yellow_range":       boundary.yellow_range,
        "red_threshold":      boundary.red_threshold,
        "couplings":          boundary.couplings,
        "layer_key":          boundary.layer_key,
        "notes":              boundary.notes,
    }


def global_confidence_multiplier(assessments: Dict[str, Dict]) -> float:
    """Product of (1 - penalty) across all assessed assumptions."""
    multiplier = 1.0
    for data in assessments.values():
        penalty = data.get("confidence_penalty", 0.0)
        if isinstance(penalty, (int, float)):
            multiplier *= (1.0 - penalty)
    return max(0.0, multiplier)


def detect_cascade_risk(assessments: Dict[str, Dict]) -> Dict:
    """
    Detect convergence of failures across coupled assumptions.
    """
    yellow = [k for k, v in assessments.items() if v.get("status") == "YELLOW"]
    red    = [k for k, v in assessments.items() if v.get("status") == "RED"]

    degraded_set = set(yellow + red)
    coupled_degraded = []
    for src, targets in COUPLING_GRAPH.items():
        if src in degraded_set:
            for tgt in targets:
                if tgt in degraded_set:
                    pair = tuple(sorted((src, tgt)))
                    if pair not in coupled_degraded:
                        coupled_degraded.append(pair)

    n_red    = len(red)
    n_yellow = len(yellow)
    n_coupled= len(coupled_degraded)

    if n_red >= 3 or (n_red >= 2 and n_coupled >= 2):
        level   = "CRITICAL"
        message = "System entering unknown state — multiple coupled assumptions left stable regime"
    elif n_red >= 1 and n_coupled >= 2:
        level   = "HIGH"
        message = "Cascade propagating — RED assumption driving coupled YELLOWs"
    elif n_coupled >= 3:
        level   = "MODERATE"
        message = "Multiple coupled assumption pairs degrading simultaneously"
    elif n_yellow >= 4:
        level   = "LOW"
        message = "Broad degradation — monitor coupling"
    else:
        level   = "MINIMAL"
        message = "No cascade convergence detected"

    irreversible = [
        k for k in red
        if "IRREVERSIBLE" in REGISTRY.get(k, AssumptionBoundary(
            "", "", "", (0,1), (0,1), 0, True, notes=""
        )).notes.upper()
    ]

    return {
        "cascade_level":       level,
        "message":             message,
        "red_assumptions":     red,
        "yellow_assumptions":  yellow,
        "coupled_degraded":    [list(p) for p in coupled_degraded],
        "irreversible_active": irreversible,
        "n_red":               n_red,
        "n_yellow":            n_yellow,
        "n_coupled_pairs":     n_coupled,
    }


def full_report(values: Dict[str, Any]) -> Dict:
    """
    Single call: flat values dict in, full validity report out.
    values keys must match AssumptionBoundary.layer_key values.
    """
    assessments = assess_values(values)
    multiplier  = global_confidence_multiplier(assessments)
    cascade     = detect_cascade_risk(assessments)

    return {
        "assumptions":                  assessments,
        "global_confidence_multiplier": multiplier,
        "cascade":                      cascade,
        "summary": {
            "total":   len(assessments),
            "green":   len([v for v in assessments.values() if v.get("status") == "GREEN"]),
            "yellow":  len([v for v in assessments.values() if v.get("status") == "YELLOW"]),
            "red":     len([v for v in assessments.values() if v.get("status") == "RED"]),
            "unknown": len([v for v in assessments.values() if v.get("status") == "UNKNOWN"]),
        },
    }
