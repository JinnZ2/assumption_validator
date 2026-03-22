# assumption_validator/adapters/generic.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Generic adapter.
# Works with any data source.
# Feed in a flat dict of {layer_key: value} pairs.
# Includes AssumptionBridge for wrapping any model.

from typing import Dict, Any, List, Optional
from datetime import datetime

from assumption_validator.registry import (
    REGISTRY,
    full_report,
    assess_values,
    global_confidence_multiplier,
)


# ─────────────────────────────────────────────
# GENERIC ADAPTER
# ─────────────────────────────────────────────

class GenericAdapter:
    """
    Simplest possible adapter.
    Feed any dict of {layer_key: value} pairs.
    Keys must match AssumptionBoundary.layer_key values.

    Usage
    -----
    adapter = GenericAdapter({
        "co2_ppm":          428.0,
        "amoc_sv":          14.5,
        "grid_inertia_s":   3.2,
        "permafrost_CO2_GtC_yr": 0.9,
    })

    report = adapter.full_report()
    """

    def __init__(self, values: Dict[str, Any] = None):
        self._values = dict(values or {})

    def update(self, values: Dict[str, Any]):
        """Merge new values into current state."""
        self._values.update(values)

    def set(self, key: str, value: Any):
        """Set a single value."""
        self._values[key] = value

    def fetch(self) -> Dict[str, Any]:
        """Return current values. Called by UniversalMonitor."""
        return dict(self._values)

    def full_report(self) -> Dict:
        """Convenience — run full registry assessment on current values."""
        return full_report(self._values)

    def available_keys(self) -> List[str]:
        """Keys in current values that match registry layer_keys."""
        registry_keys = {b.layer_key for b in REGISTRY.values() if b.layer_key}
        return [k for k in self._values if k in registry_keys]

    def unknown_keys(self) -> List[str]:
        """Keys in current values not found in registry."""
        registry_keys = {b.layer_key for b in REGISTRY.values() if b.layer_key}
        return [k for k in self._values if k not in registry_keys]

    def missing_keys(self) -> List[str]:
        """Registry layer_keys not covered by current values."""
        registry_keys = {b.layer_key for b in REGISTRY.values() if b.layer_key}
        return [k for k in registry_keys if k not in self._values]


# ─────────────────────────────────────────────
# ASSUMPTION BRIDGE
# Wraps any model with assumption awareness.
# ─────────────────────────────────────────────

class AssumptionBridge:
    """
    Wraps any model and adds assumption validity awareness.
    The model keeps running exactly as before.
    The bridge intercepts the output and adds:
      - adjusted confidence
      - assumption status
      - warnings
      - regime extrapolation flag

    Usage
    -----
    bridge = AssumptionBridge(
        model = your_model,
        model_metadata = {
            "name":              "My Climate Model",
            "type":              "climate_projection",
            "training_year":     2022,
            "derivation_regime": "holocene",
        },
        adapter = GenericAdapter(current_values),
    )

    result = bridge.predict(input_data, base_confidence=0.85)
    """

    # Which assumption IDs each model type depends on
    MODEL_TYPE_ASSUMPTIONS: Dict[str, List[str]] = {
        "weather_forecast":   ["rotation_rate", "co2_concentration",
                               "jet_stream_shift", "amoc_strength"],
        "climate_projection": ["rotation_rate", "co2_concentration",
                               "amoc_strength", "greenland_mass",
                               "permafrost_carbon", "ghg_forcing"],
        "grid_stability":     ["rotation_rate", "grid_inertia",
                               "grid_frequency_deviation", "gic_risk"],
        "carbon_cycle":       ["co2_concentration", "permafrost_carbon",
                               "amazon_sink", "ocean_ph", "nep_carbon_sink"],
        "agriculture":        ["jet_stream_shift", "co2_concentration",
                               "hadley_extent", "amoc_strength"],
        "coastal_planning":   ["sea_level_rate", "greenland_mass",
                               "antarctica_mass"],
        "ocean_model":        ["amoc_strength", "ocean_ph",
                               "marine_productivity", "sst_anomaly"],
        "energy":             ["grid_inertia", "grid_frequency_deviation",
                               "gic_risk", "rotation_rate"],
        "general":            list(REGISTRY.keys()),
    }

    # Earth regime definitions — CO2 ppm ranges
    REGIMES: Dict[str, Dict] = {
        "last_glacial_maximum": {"co2_range": (170, 200),  "description": "~20,000 years ago"},
        "holocene":             {"co2_range": (260, 280),  "description": "last 11,700 years"},
        "eemian":               {"co2_range": (270, 290),  "description": "~125,000 years ago"},
        "pliocene":             {"co2_range": (350, 450),  "description": "~3 million years ago"},
        "anthropocene":         {"co2_range": (400, 1000), "description": "present-future"},
    }

    def __init__(
        self,
        model,
        model_metadata : Dict[str, Any],
        adapter        : Optional[GenericAdapter] = None,
    ):
        self.model    = model
        self.metadata = model_metadata
        self.adapter  = adapter or GenericAdapter({})

    def predict(
        self,
        input_data     : Any,
        base_confidence: float = 0.5,
    ) -> Dict:
        """
        Run model prediction and return confidence-adjusted output.
        """
        # Get assumption assessment
        values      = self.adapter.fetch()
        report      = full_report(values)
        assumptions = report["assumptions"]

        # Filter to relevant assumptions for this model type
        model_type   = self.metadata.get("type", "general")
        relevant_ids = self.MODEL_TYPE_ASSUMPTIONS.get(
            model_type,
            self.MODEL_TYPE_ASSUMPTIONS["general"],
        )
        subset = {k: v for k, v in assumptions.items() if k in relevant_ids}

        # Confidence multiplier
        multiplier = 1.0
        for v in subset.values():
            p = v.get("confidence_penalty", 0.0)
            if isinstance(p, (int, float)):
                multiplier *= (1.0 - p)

        adjusted_conf = base_confidence * multiplier

        # Overall status
        reds    = [k for k, v in subset.items() if v.get("status") == "RED"]
        yellows = [k for k, v in subset.items() if v.get("status") == "YELLOW"]

        if reds:
            overall = "INVALID"
        elif yellows:
            overall = "DEGRADED"
        else:
            overall = "VALID"

        # Regime extrapolation check
        regime_check = self._check_regime(values)

        # Additional penalty for regime extrapolation
        if regime_check["status"] == "CRITICAL_EXTRAPOLATION":
            adjusted_conf *= 0.1
        elif regime_check["status"] == "HIGH_EXTRAPOLATION":
            adjusted_conf *= 0.3

        # Run the actual model
        try:
            if hasattr(self.model, "predict"):
                prediction = self.model.predict(input_data)
            elif callable(self.model):
                prediction = self.model(input_data)
            else:
                prediction = self.model
        except Exception as exc:
            return {
                "error":          str(exc),
                "overall_status": "ERROR",
                "timestamp":      datetime.utcnow().isoformat(),
            }

        # Build warnings
        warnings = []
        for k in reds:
            b = REGISTRY.get(k)
            warnings.append(
                f"CRITICAL [{k}]: {b.name if b else k} — RED. "
                f"Value: {subset[k].get('value'):.4g} "
                f"{subset[k].get('units','')}."
            )
        for k in yellows:
            b = REGISTRY.get(k)
            warnings.append(
                f"CAUTION [{k}]: {b.name if b else k} — YELLOW. "
                f"Value: {subset[k].get('value'):.4g} "
                f"{subset[k].get('units','')}."
            )
        if regime_check.get("message"):
            warnings.append(regime_check["message"])

        training_year = self.metadata.get("training_year", 2020)
        if training_year < 2023:
            warnings.append(
                f"Model trained in {training_year}. "
                f"Assumptions may not reflect current conditions."
            )

        return {
            "prediction":            prediction,
            "original_confidence":   base_confidence,
            "adjusted_confidence":   round(adjusted_conf, 4),
            "confidence_multiplier": round(multiplier, 4),
            "confidence_loss_pct":   round((1.0 - multiplier) * 100, 1),
            "overall_status":        overall,
            "assumption_status":     subset,
            "regime_extrapolation":  regime_check,
            "red_assumptions":       reds,
            "yellow_assumptions":    yellows,
            "warnings":              warnings,
            "cascade_level":         report["cascade"]["cascade_level"],
            "model_name":            self.metadata.get("name", "unnamed"),
            "model_type":            model_type,
            "timestamp":             datetime.utcnow().isoformat(),
        }

    def _check_regime(self, values: Dict[str, Any]) -> Dict:
        """Check if model is being used outside its derivation regime."""
        deriv   = self.metadata.get("derivation_regime", "holocene")
        co2_val = values.get("co2_ppm")

        if co2_val is None:
            return {
                "status":  "UNKNOWN",
                "message": "Cannot determine current regime — no CO₂ data.",
            }

        # Identify current regime from CO₂
        current_regime = "unknown"
        for regime_name, rdef in self.REGIMES.items():
            lo, hi = rdef["co2_range"]
            if lo <= co2_val <= hi:
                current_regime = regime_name
                break
        if co2_val > 400:
            current_regime = "anthropocene"

        if deriv == current_regime:
            return {
                "status":         "WITHIN_REGIME",
                "derivation":     deriv,
                "current_regime": current_regime,
                "message":        None,
            }

        # Extrapolation risk
        deriv_def   = self.REGIMES.get(deriv, {})
        deriv_co2   = deriv_def.get("co2_range", (280, 280))
        exceedance  = (co2_val - deriv_co2[1]) / (deriv_co2[1] + 1e-9)

        if exceedance > 0.5 or current_regime == "anthropocene":
            status  = "CRITICAL_EXTRAPOLATION"
            message = (
                f"Model derived in {deriv} regime (CO₂ {deriv_co2[0]}–{deriv_co2[1]} ppm). "
                f"Current CO₂ {co2_val:.0f} ppm — {current_regime} regime. "
                f"Equations may not apply to current conditions."
            )
        elif exceedance > 0.2:
            status  = "HIGH_EXTRAPOLATION"
            message = (
                f"Model derived in {deriv} regime. "
                f"Current CO₂ {co2_val:.0f} ppm exceeds derivation range. "
                f"Confidence significantly reduced."
            )
        else:
            status  = "MODERATE_EXTRAPOLATION"
            message = (
                f"Model derived in {deriv} regime. "
                f"Current conditions at edge of derivation range."
            )

        return {
            "status":         status,
            "derivation":     deriv,
            "current_regime": current_regime,
            "co2_current":    co2_val,
            "co2_deriv_max":  deriv_co2[1],
            "message":        message,
        }

    def report(self) -> Dict:
        """Full validity report for this model without running prediction."""
        values = self.adapter.fetch()
        return full_report(values)
