# assumption_validator/adapters/earth_systems.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Adapter connecting assumption-validator to earth-systems-physics.
# Reads layer coupling_state outputs from cascade_engine.run_all_layers()
# and translates them into the flat {layer_key: value} dict
# that the universal registry expects.
#
# earth-systems-physics repo:
# github.com/JinnZ2/earth-systems-physics

from typing import Dict, Any, Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
# LAYER KEY MAP
# Maps earth-systems-physics coupling_state output keys
# to assumption_validator registry layer_key values.
# Left  = key in layer coupling_state output
# Right = layer_key in AssumptionBoundary
# ─────────────────────────────────────────────

LAYER_KEY_MAP: Dict[str, str] = {

    # ── LAYER 0 — ELECTROMAGNETICS ───────────────────────────────────
    "plasma_frequency_hz":       "plasma_frequency_hz",
    "schumann_f1_shift_hz":      "schumann_f1_shift_hz",

    # ── LAYER 1 — MAGNETOSPHERE ──────────────────────────────────────
    "magnetopause_standoff_Re":  "magnetopause_standoff_Re",
    # rotation_coupling is a dict — unpacked below
    # "omega_change_rads" extracted from rotation_coupling dict

    # ── LAYER 2 — IONOSPHERE ─────────────────────────────────────────
    "critical_frequency_hz":     "critical_frequency_hz",
    "joule_heating_Wm3":         "joule_heating_Wm3",

    # ── LAYER 3 — ATMOSPHERE ─────────────────────────────────────────
    "GHG_forcing_Wm2":           "GHG_forcing_Wm2",
    "net_forcing_Wm2":           "net_forcing_Wm2",
    "coriolis_f_rads":           "coriolis_f_rads",
    "jet_shear_proxy":           "jet_shear_proxy",
    "hadley_extent_deg":         "hadley_extent_deg",
    "convection_active":         "convection_active",
    "precipitable_water_mm":     "precipitable_water_mm",

    # ── LAYER 4 — HYDROSPHERE ────────────────────────────────────────
    "AMOC_collapse_risk":        "AMOC_collapse_risk",
    "AMOC_heat_transport_W":     "AMOC_heat_transport_W",
    "arctic_amplification_K":    "arctic_amplification_K",
    "ice_albedo_feedback_Wm2":   "ice_albedo_feedback_Wm2",
    "committed_warming_timescale_yr": "committed_warming_timescale_yr",
    "thermal_SLR_m":             "thermal_SLR_m",

    # ── LAYER 5 — LITHOSPHERE ────────────────────────────────────────
    "LOD_change_ms":             "LOD_change_ms",
    "polar_drift_deg_yr":        "polar_drift_deg_yr",
    "fault_coulomb_change_Pa":   "fault_coulomb_change_Pa",
    "volcanic_enhancement":      "volcanic_enhancement",
    "geological_co2_GtC_yr":     "geological_co2_GtC_yr",

    # ── LAYER 6 — BIOSPHERE ──────────────────────────────────────────
    "NEP_gC_m2_day":             "NEP_gC_m2_day",
    "NEP_carbon_sink":           "NEP_carbon_sink",
    "permafrost_CO2_GtC_yr":     "permafrost_CO2_GtC_yr",
    "permafrost_CH4_GtC_yr":     "permafrost_CH4_GtC_yr",
    "ocean_pH":                  "ocean_ph",
    "coral_dissolution_active":  "coral_dissolution_active",
    "marine_productivity_change_frac": "marine_productivity_change_frac",
    "amazon_tipping_proximity":  "amazon_tipping_proximity",
    "amazon_tipping_imminent":   "amazon_tipping_imminent",
    "atmospheric_CO2_accumulation": "atmospheric_CO2_accumulation",
    "planetary_boundaries_crossed": "planetary_boundaries_crossed",
}

# Keys extracted from nested dicts in layer output
NESTED_EXTRACTIONS: List[Dict] = [
    # rotation_coupling dict from layer 1
    {
        "parent_key": "rotation_coupling",
        "child_key":  "omega_change_rads",
        "output_key": "omega_change_rads",
    },
    # AMOC density gradient from layer 4
    {
        "parent_key": "AMOC_density_gradient",
        "child_key":  None,   # scalar — use value directly
        "output_key": "AMOC_density_gradient",
    },
]

# Synthetic keys computed from layer outputs
# Each entry: output_key, source_keys, transform function
COMPUTED_KEYS: List[Dict] = [
    {
        "output_key": "amoc_sv",
        # Approximate AMOC Sv from heat transport
        # AMOC_heat_transport_W / (rho * cp * delta_T) ~ Sv
        # Using simplified: heat_transport / 4e12 ~ Sv
        "source":     "AMOC_heat_transport_W",
        "transform":  lambda w: max(0.0, w / 4e12) if w else None,
    },
    {
        "output_key": "co2_ppm",
        # Derive CO2 ppm from GHG forcing
        # delta_F = 5.35 * ln(C/C0) => C = C0 * exp(delta_F / 5.35)
        "source":     "GHG_forcing_Wm2",
        "transform":  lambda f: round(280.0 * (2.718281828 ** (f / 5.35)), 1) if f else None,
    },
    {
        "output_key": "slr_mm_yr",
        # Convert thermal SLR from meters to mm/yr
        # thermal_SLR_m is cumulative — use as proxy for rate
        "source":     "thermal_SLR_m",
        "transform":  lambda m: round(m * 1000 / 30.0, 2) if m else None,
        # divided by 30 years of forcing as rough rate estimate
    },
    {
        "output_key": "gic_current_A",
        # Approximate GIC from magnetopause compression
        # Kp ~ 10 when standoff < 6 Re => GIC ~ Kp^2 * 0.5
        "source":     "magnetopause_standoff_Re",
        "transform":  lambda re: round(max(0, (10 - re)) ** 2 * 0.5, 1) if re else 0.0,
    },
    {
        "output_key": "jet_shift_deg",
        # Jet shear proxy to approximate shift
        # Negative shear = westerly, weakening toward zero = poleward shift
        "source":     "jet_shear_proxy",
        "transform":  lambda s: round(max(0, 1 + s * 1000), 2) if s else None,
    },
    {
        "output_key": "grid_inertia_s",
        # Not directly in earth-systems-physics
        # Placeholder — returns None, triggering UNKNOWN status
        "source":     None,
        "transform":  lambda x: None,
    },
    {
        "output_key": "greenland_mass_gt_yr",
        # Not directly output — placeholder
        "source":     None,
        "transform":  lambda x: None,
    },
    {
        "output_key": "sst_anomaly_K",
        # Approximate from arctic amplification
        "source":     "arctic_amplification_K",
        "transform":  lambda k: round(k / 3.0, 2) if k else None,
    },
]


# ─────────────────────────────────────────────
# EARTH SYSTEMS ADAPTER
# ─────────────────────────────────────────────

class EarthSystemsAdapter:
    """
    Adapter that reads earth-systems-physics layer outputs
    and translates them for the universal assumption registry.

    Two usage modes:

    Mode 1 — pass layer_states directly:
        from cascade_engine import run_all_layers, BASELINE
        layer_states = run_all_layers(BASELINE)
        adapter      = EarthSystemsAdapter(layer_states)
        report       = adapter.full_report()

    Mode 2 — lazy fetch on demand:
        adapter = EarthSystemsAdapter()
        adapter.set_params(my_params)
        report  = adapter.full_report()   # runs layers internally
    """

    def __init__(
        self,
        layer_states: Optional[Dict[int, Dict]] = None,
        params      : Optional[Dict]            = None,
    ):
        """
        layer_states : output of cascade_engine.run_all_layers()
                       if None, will run internally on fetch()
        params       : parameter dict for run_all_layers()
                       defaults to BASELINE if not provided
        """
        self._layer_states : Optional[Dict[int, Dict]] = layer_states
        self._params       : Optional[Dict]            = params
        self._last_fetch   : Optional[datetime]        = None
        self._last_values  : Optional[Dict]            = None

    def set_params(self, params: Dict):
        """Update simulation parameters."""
        self._params       = params
        self._layer_states = None   # force re-run on next fetch

    def set_layer_states(self, layer_states: Dict[int, Dict]):
        """Directly inject pre-computed layer states."""
        self._layer_states = layer_states
        self._last_values  = None

    def fetch(self) -> Dict[str, Any]:
        """
        Return flat {layer_key: value} dict for registry assessment.
        Runs earth-systems-physics layers if no states provided.
        """
        # Use cached if layer_states already set and values computed
        if self._last_values is not None and self._layer_states is not None:
            return dict(self._last_values)

        # Run layers if needed
        if self._layer_states is None:
            self._layer_states = self._run_layers()

        values = self._translate(self._layer_states)
        self._last_values = values
        self._last_fetch  = datetime.utcnow()
        return dict(values)

    def full_report(self) -> Dict:
        """Run full registry assessment on current layer states."""
        from assumption_validator.registry import full_report as _full_report
        values = self.fetch()
        report = _full_report(values)
        report["_adapter"]    = "EarthSystemsAdapter"
        report["_timestamp"]  = (
            self._last_fetch.isoformat() if self._last_fetch else None
        )
        return report

    def layer_states(self) -> Optional[Dict[int, Dict]]:
        """Return raw layer states."""
        if self._layer_states is None:
            self._layer_states = self._run_layers()
        return self._layer_states

    # ── INTERNAL ─────────────────────────────────────────────────────

    def _run_layers(self) -> Dict[int, Dict]:
        """Run earth-systems-physics cascade engine."""
        try:
            from cascade_engine import run_all_layers, BASELINE
            params = self._params or BASELINE
            return run_all_layers(params)
        except ImportError as exc:
            raise ImportError(
                "earth-systems-physics not found. "
                "Clone github.com/JinnZ2/earth-systems-physics "
                "into the same parent directory, or pass layer_states directly. "
                f"Original error: {exc}"
            )

    def _translate(self, layer_states: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Translate layer coupling_state outputs to flat registry dict.
        """
        # Flatten all layer outputs
        flat: Dict[str, Any] = {}
        for layer_output in layer_states.values():
            if isinstance(layer_output, dict):
                flat.update(layer_output)

        values: Dict[str, Any] = {}

        # Direct key mappings
        for src_key, dst_key in LAYER_KEY_MAP.items():
            if src_key in flat:
                values[dst_key] = flat[src_key]

        # Nested extractions
        for extraction in NESTED_EXTRACTIONS:
            parent = flat.get(extraction["parent_key"])
            if parent is None:
                continue
            if extraction["child_key"] is None:
                # Scalar value
                values[extraction["output_key"]] = parent
            elif isinstance(parent, dict):
                child = parent.get(extraction["child_key"])
                if child is not None:
                    values[extraction["output_key"]] = child

        # Computed keys
        for comp in COMPUTED_KEYS:
            src_key   = comp.get("source")
            transform = comp["transform"]
            out_key   = comp["output_key"]
            if src_key is None:
                result = transform(None)
            else:
                src_val = flat.get(src_key) or values.get(src_key)
                result  = transform(src_val) if src_val is not None else None
            if result is not None:
                values[out_key] = result

        return values

    def translation_report(self) -> Dict:
        """
        Diagnostic: show which layer keys mapped to which registry keys,
        which were missed, and which registry keys have no data.
        """
        if self._layer_states is None:
            self._layer_states = self._run_layers()

        flat: Dict[str, Any] = {}
        for layer_output in self._layer_states.values():
            if isinstance(layer_output, dict):
                flat.update(layer_output)

        values = self._translate(self._layer_states)

        from assumption_validator.registry import REGISTRY
        registry_keys = {b.layer_key for b in REGISTRY.values() if b.layer_key}

        mapped   = {k: v for k, v in values.items() if k in registry_keys}
        unmapped = {k: v for k, v in values.items() if k not in registry_keys}
        missing  = [k for k in registry_keys if k not in values]

        return {
            "layer_keys_available": len(flat),
            "registry_keys_total":  len(registry_keys),
            "mapped_count":         len(mapped),
            "unmapped_count":       len(unmapped),
            "missing_count":        len(missing),
            "mapped":               mapped,
            "unmapped_keys":        list(unmapped.keys()),
            "missing_keys":         missing,
        }


# ─────────────────────────────────────────────
# SCENARIO RUNNER
# Run earth-systems-physics scenarios and assess validity
# ─────────────────────────────────────────────

class ScenarioAdapter(EarthSystemsAdapter):
    """
    Extends EarthSystemsAdapter with scenario support.
    Run named scenarios from cascade_engine.SCENARIOS
    and get assumption validity reports for each.

    Usage
    -----
    adapter = ScenarioAdapter()
    report  = adapter.run_scenario("amoc_collapse")
    compare = adapter.compare_scenarios(
        ["amoc_collapse", "permafrost_acceleration", "sulfate_geoengineering"]
    )
    """

    def run_scenario(self, scenario_name: str) -> Dict:
        """
        Run a named scenario and return full validity report.
        Does not modify adapter state.
        """
        try:
            from cascade_engine import SCENARIOS, run_cascade, BASELINE
        except ImportError as exc:
            raise ImportError(
                f"earth-systems-physics required for scenario runner. {exc}"
            )

        if scenario_name not in SCENARIOS:
            available = list(SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {available}"
            )

        result = run_cascade(
            SCENARIOS[scenario_name],
            baseline = dict(self._params or BASELINE),
            verbose  = False,
        )

        adapter = EarthSystemsAdapter(result.layer_states)
        report  = adapter.full_report()

        report["_scenario"]          = scenario_name
        report["_forcing"] = {
            "description": SCENARIOS[scenario_name].description,
            "layer":       SCENARIOS[scenario_name].layer,
            "variable":    SCENARIOS[scenario_name].variable,
            "magnitude":   SCENARIOS[scenario_name].magnitude,
            "units":       SCENARIOS[scenario_name].units,
        }
        report["_thresholds_crossed"] = result.threshold_crossings
        report["_amplifying_loops"]   = result.amplifying_loops

        return report

    def compare_scenarios(self, scenario_names: List[str]) -> Dict:
        """
        Run multiple scenarios and compare assumption validity across them.
        Returns side-by-side cascade levels and confidence multipliers.
        """
        comparison = {}
        for name in scenario_names:
            try:
                report = self.run_scenario(name)
                comparison[name] = {
                    "cascade_level":     report["cascade"]["cascade_level"],
                    "confidence":        report["global_confidence_multiplier"],
                    "red_count":         report["summary"]["red"],
                    "yellow_count":      report["summary"]["yellow"],
                    "thresholds_crossed": len(report.get("_thresholds_crossed", [])),
                    "amplifying_loops":  len(report.get("_amplifying_loops", [])),
                }
            except Exception as exc:
                comparison[name] = {"error": str(exc)}

        # Rank by confidence (lowest = most damaging)
        ranked = sorted(
            [(k, v) for k, v in comparison.items() if "confidence" in v],
            key     = lambda x: x[1]["confidence"],
        )

        return {
            "scenarios":     comparison,
            "ranked_by_risk": [r[0] for r in ranked],
            "most_damaging":  ranked[0][0]  if ranked else None,
            "least_damaging": ranked[-1][0] if ranked else None,
        }

    def available_scenarios(self) -> List[str]:
        """List available scenario names."""
        try:
            from cascade_engine import SCENARIOS
            return list(SCENARIOS.keys())
        except ImportError:
            return []


# ─────────────────────────────────────────────
# ADAPTERS __init__.py content
# ─────────────────────────────────────────────

ADAPTERS_INIT = '''
# assumption_validator/adapters/__init__.py
# CC0 — No Rights Reserved

from assumption_validator.adapters.generic import GenericAdapter, AssumptionBridge
from assumption_validator.adapters.noaa import NOAAAdapter
from assumption_validator.adapters.earth_systems import (
    EarthSystemsAdapter,
    ScenarioAdapter,
)

__all__ = [
    "GenericAdapter",
    "AssumptionBridge",
    "NOAAAdapter",
    "EarthSystemsAdapter",
    "ScenarioAdapter",
]
'''


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from assumption_validator.monitors import print_report

    print("EarthSystemsAdapter — attempting to connect to earth-systems-physics\n")

    try:
        adapter = EarthSystemsAdapter()
        report  = adapter.full_report()

        print_report(report, show_green=False)

        print("\nTRANSLATION REPORT:")
        tr = adapter.translation_report()
        print(f"  Layer keys available : {tr['layer_keys_available']}")
        print(f"  Registry keys total  : {tr['registry_keys_total']}")
        print(f"  Mapped               : {tr['mapped_count']}")
        print(f"  Missing              : {tr['missing_count']}")
        if tr["missing_keys"]:
            print(f"  Missing keys: {tr['missing_keys'][:5]}")

        print("\nSCENARIO RUNNER:")
        scenario_adapter = ScenarioAdapter()
        available        = scenario_adapter.available_scenarios()
        if available:
            print(f"  Available scenarios: {available}")
            compare = scenario_adapter.compare_scenarios(available[:3])
            print(f"  Most damaging  : {compare['most_damaging']}")
            print(f"  Least damaging : {compare['least_damaging']}")
        else:
            print("  No scenarios available — earth-systems-physics not found")

    except ImportError as exc:
        print(f"earth-systems-physics not found: {exc}")
        print("\nTo use this adapter:")
        print("  1. Clone github.com/JinnZ2/earth-systems-physics")
        print("  2. Place in same parent directory as assumption-validator")
        print("  3. Or pass layer_states directly:")
        print("     adapter = EarthSystemsAdapter(layer_states=your_states)")
