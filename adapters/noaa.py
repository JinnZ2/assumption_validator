# assumption_validator/adapters/noaa.py
# assumption-validator
# CC0 — No Rights Reserved
#
# NOAA / IERS live data adapter.
# Pulls from public data sources:
#   IERS  — Earth rotation, LOD
#   NOAA ESRL — CO2 (Mauna Loa)
#   RAPID — AMOC strength
#   NSIDC — Arctic sea ice
#   GRACE-FO — Ice mass
#   NOAA SWPC — Space weather / GIC risk
#   GTN-P — Permafrost
#   NCEP — Jet stream, atmosphere
#
# All fetches degrade gracefully to physics-based simulation
# when live feeds are unavailable.
# Simulation values are labeled clearly in metadata.

import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import json
import time


# ─────────────────────────────────────────────
# BASE FETCHER
# ─────────────────────────────────────────────

class DataFetcher:
    """Base class for individual data source fetchers."""

    def __init__(self, timeout: int = 10):
        self.timeout   = timeout
        self.last_fetch: Optional[datetime] = None
        self.last_value: Optional[Any]      = None
        self.source    = "unknown"
        self.simulated = False

    def get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """GET with timeout and error suppression."""
        try:
            return requests.get(url, timeout=self.timeout, **kwargs)
        except Exception:
            return None

    def _years_since(self, year: int, month: int = 1, day: int = 1) -> float:
        ref = datetime(year, month, day)
        return (datetime.utcnow() - ref).days / 365.25

    def _day_of_year(self) -> int:
        return datetime.utcnow().timetuple().tm_yday


# ─────────────────────────────────────────────
# INDIVIDUAL FETCHERS
# ─────────────────────────────────────────────

class IERSFetcher(DataFetcher):
    """
    Earth Orientation Parameters from IERS.
    Provides rotation rate (omega fractional change) and LOD change.
    """

    # IERS rapid data — finals2000A.daily
    URL = "https://datacenter.iers.org/data/latestVersion/finals2000A.data.csv"

    def fetch(self) -> Dict[str, Any]:
        resp = self.get(self.URL)
        if resp and resp.status_code == 200:
            try:
                return self._parse(resp.text)
            except Exception:
                pass
        return self._simulate()

    def _parse(self, text: str) -> Dict[str, Any]:
        """
        Parse IERS finals2000A CSV.
        Column layout varies by version — extract LOD column.
        """
        lines = [l for l in text.strip().splitlines()
                 if not l.startswith("#") and l.strip()]
        if not lines:
            raise ValueError("empty")

        # Last non-empty line = most recent
        last = lines[-1].split()
        # LOD excess (ms) is typically column index 8 in finals2000A
        lod_ms = float(last[8]) if len(last) > 8 else 0.0

        lod_seconds    = 86400.0 + lod_ms / 1000.0
        omega_current  = 2 * np.pi / lod_seconds
        omega_0        = 7.292115e-5
        frac_change    = (omega_current - omega_0) / omega_0

        self.simulated = False
        self.source    = "IERS_finals2000A"
        return {
            "omega_change_rads": abs(frac_change),
            "LOD_change_ms":     lod_ms,
            "_source":           self.source,
            "_simulated":        False,
        }

    def _simulate(self) -> Dict[str, Any]:
        """Physics-based simulation from known trends."""
        yrs      = self._years_since(2020)
        # +1.8 ms/century secular trend from ice melt
        lod_ms   = 1.8e-3 * yrs / 100 + 0.5 * np.sin(2 * np.pi * yrs)
        lod_s    = 86400.0 + lod_ms / 1000.0
        omega    = 2 * np.pi / lod_s
        omega_0  = 7.292115e-5
        frac     = abs((omega - omega_0) / omega_0)

        self.simulated = True
        self.source    = "SIMULATED_IERS_TREND"
        return {
            "omega_change_rads": frac,
            "LOD_change_ms":     lod_ms,
            "_source":           self.source,
            "_simulated":        True,
        }


class NOAAco2Fetcher(DataFetcher):
    """
    CO2 from NOAA ESRL Mauna Loa.
    Falls back to GML JSON endpoint.
    """

    URL_JSON = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.json"
    URL_TXT  = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"

    def fetch(self) -> Dict[str, Any]:
        # Try JSON first
        resp = self.get(self.URL_JSON)
        if resp and resp.status_code == 200:
            try:
                return self._parse_json(resp.json())
            except Exception:
                pass

        # Try text
        resp = self.get(self.URL_TXT)
        if resp and resp.status_code == 200:
            try:
                return self._parse_txt(resp.text)
            except Exception:
                pass

        return self._simulate()

    def _parse_json(self, data: dict) -> Dict[str, Any]:
        obs = data.get("observations", [])
        if not obs:
            raise ValueError("empty")
        latest = obs[-1]
        co2    = float(latest.get("value", 0))
        self.simulated = False
        self.source    = "NOAA_ESRL_MLO"
        return {
            "co2_ppm":    co2,
            "_source":    self.source,
            "_simulated": False,
        }

    def _parse_txt(self, text: str) -> Dict[str, Any]:
        lines = [l for l in text.splitlines()
                 if not l.startswith("#") and l.strip()]
        if not lines:
            raise ValueError("empty")
        last   = lines[-1].split()
        co2    = float(last[3])
        self.simulated = False
        self.source    = "NOAA_ESRL_MLO_TXT"
        return {
            "co2_ppm":    co2,
            "_source":    self.source,
            "_simulated": False,
        }

    def _simulate(self) -> Dict[str, Any]:
        yrs  = self._years_since(2000)
        co2  = 370.0 + 2.5 * yrs
        doy  = self._day_of_year()
        co2 += 3.0 * np.sin(2 * np.pi * (doy - 90) / 365)
        self.simulated = True
        self.source    = "SIMULATED_MAUNA_LOA"
        return {
            "co2_ppm":    round(co2, 2),
            "_source":    self.source,
            "_simulated": True,
        }


class RAPIDFetcher(DataFetcher):
    """
    AMOC strength from RAPID array (NOC Southampton).
    Falls back to trend simulation.
    """

    URL = "https://www.rapid.ac.uk/rapidmoc/data/moc_vertical.nc"

    def fetch(self) -> Dict[str, Any]:
        # RAPID NetCDF — requires scipy/netCDF4 for real parse.
        # Attempt lightweight HEAD check; if available note it.
        # Full parse only if netCDF4 present.
        try:
            import netCDF4
            resp = self.get(self.URL)
            if resp and resp.status_code == 200:
                return self._parse_nc(resp.content)
        except ImportError:
            pass
        return self._simulate()

    def _parse_nc(self, content: bytes) -> Dict[str, Any]:
        """Parse RAPID NetCDF (requires netCDF4)."""
        import netCDF4
        import io
        nc   = netCDF4.Dataset("inmemory", memory=content)
        moc  = nc.variables["moc_mar_hc10"][-1]   # most recent
        amoc = float(np.mean(moc))
        self.simulated = False
        self.source    = "RAPID_ARRAY"
        return {
            "amoc_sv":    round(amoc, 2),
            "_source":    self.source,
            "_simulated": False,
        }

    def _simulate(self) -> Dict[str, Any]:
        yrs  = self._years_since(2004)
        amoc = max(8.0, 18.0 - 0.04 * yrs)
        amoc += 0.5 * np.sin(2 * np.pi * yrs)
        self.simulated = True
        self.source    = "SIMULATED_RAPID_TREND"
        return {
            "amoc_sv":    round(amoc, 2),
            "_source":    self.source,
            "_simulated": True,
        }


class NSIDCFetcher(DataFetcher):
    """
    Arctic sea ice extent from NSIDC.
    """

    URL = "https://noaadata.apps.nsidc.org/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv"

    def fetch(self) -> Dict[str, Any]:
        resp = self.get(self.URL)
        if resp and resp.status_code == 200:
            try:
                return self._parse(resp.text)
            except Exception:
                pass
        return self._simulate()

    def _parse(self, text: str) -> Dict[str, Any]:
        lines = [l for l in text.splitlines()
                 if not l.startswith("#") and "," in l]
        if not lines:
            raise ValueError("empty")
        last   = lines[-1].split(",")
        extent = float(last[4].strip())
        self.simulated = False
        self.source    = "NSIDC_G02135"
        return {
            "arctic_ice_mkm2": round(extent, 2),
            "_source":         self.source,
            "_simulated":      False,
        }

    def _simulate(self) -> Dict[str, Any]:
        yrs    = self._years_since(1979)
        extent = max(2.0, 7.0 - 0.04 * yrs)
        doy    = self._day_of_year()
        # Seasonal: minimum September (~day 260)
        extent += 3.0 * np.cos(2 * np.pi * (doy - 260) / 365)
        self.simulated = True
        self.source    = "SIMULATED_NSIDC_TREND"
        return {
            "arctic_ice_mkm2": round(max(0.5, extent), 2),
            "_source":         self.source,
            "_simulated":      True,
        }


class GRACEFetcher(DataFetcher):
    """
    Ice mass balance from GRACE-FO via NASA PODAAC.
    Returns Greenland and Antarctica mass loss rates.
    """

    URL = "https://podaac-tools.jpl.nasa.gov/drive/files/GeodeticsGravity/tellus/L3/mascon/RL06/JPL/v02/CRI/netcdf/"

    def fetch(self) -> Dict[str, Any]:
        # Full GRACE-FO parse requires large NetCDF files.
        # Simulate from published trend data until lightweight endpoint available.
        return self._simulate()

    def _simulate(self) -> Dict[str, Any]:
        yrs = self._years_since(2002)
        # Greenland: -270 Gt/yr baseline, accelerating ~25 Gt/yr²
        greenland   = -(270.0 + 25.0 * max(0, yrs - 18))
        # Antarctica: -150 Gt/yr baseline, accelerating ~15 Gt/yr²
        antarctica  = -(150.0 + 15.0 * max(0, yrs - 18))
        # Sea level from ice
        slr_ice     = abs(greenland + antarctica) / 360.0  # mm/yr
        slr_thermal = 2.0 + 0.03 * yrs
        slr_total   = slr_ice + slr_thermal

        self.simulated = True
        self.source    = "SIMULATED_GRACE_TREND"
        return {
            "greenland_mass_gt_yr":   round(greenland,  1),
            "antarctica_mass_gt_yr":  round(antarctica, 1),
            "slr_mm_yr":              round(slr_total,  2),
            "_source":                self.source,
            "_simulated":             True,
        }


class SWPCFetcher(DataFetcher):
    """
    Space weather from NOAA Space Weather Prediction Center.
    Kp index proxy for GIC risk.
    """

    URL_KP = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    def fetch(self) -> Dict[str, Any]:
        resp = self.get(self.URL_KP)
        if resp and resp.status_code == 200:
            try:
                return self._parse(resp.json())
            except Exception:
                pass
        return self._simulate()

    def _parse(self, data: list) -> Dict[str, Any]:
        # Data format: [[timestamp, kp], ...]
        # Skip header row
        rows = [r for r in data if isinstance(r[1], (int, float, str))
                and r[0] != "time_tag"]
        if not rows:
            raise ValueError("empty")
        kp       = float(rows[-1][1])
        # Convert Kp to approximate GIC current (A/phase)
        gic_risk = kp ** 2 * 0.5
        self.simulated = False
        self.source    = "NOAA_SWPC"
        return {
            "gic_current_A": round(gic_risk, 1),
            "kp_index":      kp,
            "_source":       self.source,
            "_simulated":    False,
        }

    def _simulate(self) -> Dict[str, Any]:
        # Solar cycle 25 peak ~2025-2026
        yrs_from_peak = abs(self._years_since(2025, 6))
        kp            = max(1.0, 3.5 - yrs_from_peak * 0.5)
        gic_risk      = kp ** 2 * 0.5
        self.simulated = True
        self.source    = "SIMULATED_SOLAR_CYCLE"
        return {
            "gic_current_A": round(gic_risk, 1),
            "kp_index":      round(kp, 1),
            "_source":       self.source,
            "_simulated":    True,
        }


class PermafrostFetcher(DataFetcher):
    """
    Permafrost carbon flux estimates from GTN-P / published synthesis.
    No real-time API available — uses observation-constrained simulation.
    """

    def fetch(self) -> Dict[str, Any]:
        return self._simulate()

    def _simulate(self) -> Dict[str, Any]:
        yrs = self._years_since(2000)
        # Carbon release accelerating from ~0.5 GtC/yr in 2000
        co2_flux = min(2.0, 0.5 + 0.08 * yrs)
        ch4_flux = min(0.4, 0.05 + 0.01 * yrs)
        self.simulated = True
        self.source    = "SIMULATED_GTNP_TREND"
        return {
            "permafrost_CO2_GtC_yr": round(co2_flux, 3),
            "permafrost_CH4_GtC_yr": round(ch4_flux, 3),
            "_source":               self.source,
            "_simulated":            True,
        }


class NCEPFetcher(DataFetcher):
    """
    Atmospheric variables from NCEP/NCAR reanalysis.
    Jet stream position, GHG forcing proxy.
    """

    def fetch(self) -> Dict[str, Any]:
        return self._simulate()

    def _simulate(self) -> Dict[str, Any]:
        yrs = self._years_since(1980)
        # Jet stream: poleward shift ~0.5 deg/decade
        jet_shift  = 0.05 * yrs
        # GHG forcing based on CO2 trend
        co2        = 370.0 + 2.5 * self._years_since(2000)
        ghg        = 5.35 * np.log(co2 / 280.0)
        # Hadley extent
        hadley     = 30.0 + 0.3 * yrs / 10.0
        # Amazon tipping proximity
        amazon_tip = min(0.9, 0.20 + 0.03 * yrs / 10.0)

        self.simulated = True
        self.source    = "SIMULATED_NCEP_TREND"
        return {
            "jet_shift_deg":         round(jet_shift, 2),
            "GHG_forcing_Wm2":       round(ghg, 3),
            "hadley_extent_deg":     round(hadley, 1),
            "amazon_tipping_proximity": round(amazon_tip, 3),
            "_source":               self.source,
            "_simulated":            True,
        }


class OceanFetcher(DataFetcher):
    """
    Ocean chemistry from SOCAT / Argo synthesis.
    pH, SST anomaly, marine productivity change.
    """

    def fetch(self) -> Dict[str, Any]:
        return self._simulate()

    def _simulate(self) -> Dict[str, Any]:
        yrs      = self._years_since(1850)
        # pH declining ~0.002/yr since industrial
        ph       = max(7.80, 8.15 - 0.002 * yrs)
        # SST anomaly
        sst      = min(3.0, 0.01 * yrs)
        # Marine productivity change
        mp       = max(-0.5, -0.005 * yrs)
        # Amazon sink
        amazon   = max(-0.5, 0.8 - 0.08 * (yrs - 150))

        self.simulated = True
        self.source    = "SIMULATED_OCEAN_TREND"
        return {
            "ocean_ph":                       round(ph, 3),
            "sst_anomaly_K":                  round(sst, 2),
            "marine_productivity_change_frac": round(mp, 3),
            "amazon_sink_GtC_yr":             round(amazon, 2),
            "_source":                        self.source,
            "_simulated":                     True,
        }


# ─────────────────────────────────────────────
# NOAA ADAPTER
# ─────────────────────────────────────────────

class NOAAAdapter:
    """
    Live data adapter pulling from NOAA, IERS, NSIDC, RAPID,
    GRACE-FO, SWPC, and auxiliary sources.

    All fetches degrade gracefully to physics-based simulation
    when live feeds are unavailable. Simulated values are
    labeled in _meta output.

    Usage
    -----
    adapter = NOAAAdapter()
    values  = adapter.fetch()        # called by UniversalMonitor
    report  = adapter.full_report()  # convenience

    # Check data provenance
    meta = adapter.meta()
    for source, info in meta.items():
        print(source, "simulated:", info["simulated"])
    """

    def __init__(self, timeout: int = 10):
        self.fetchers = {
            "iers":        IERSFetcher(timeout),
            "co2":         NOAAco2Fetcher(timeout),
            "amoc":        RAPIDFetcher(timeout),
            "sea_ice":     NSIDCFetcher(timeout),
            "ice_mass":    GRACEFetcher(timeout),
            "space_weather": SWPCFetcher(timeout),
            "permafrost":  PermafrostFetcher(timeout),
            "atmosphere":  NCEPFetcher(timeout),
            "ocean":       OceanFetcher(timeout),
        }
        self._last_fetch : Optional[datetime]    = None
        self._last_values: Optional[Dict]        = None
        self._last_meta  : Dict[str, Dict]       = {}
        self._cache_ttl  : int                   = 3600  # seconds

    def fetch(self, force: bool = False) -> Dict[str, Any]:
        """
        Fetch all sources and return flat values dict.
        Cached for cache_ttl seconds unless force=True.
        """
        if (
            not force
            and self._last_values is not None
            and self._last_fetch is not None
            and (datetime.utcnow() - self._last_fetch).seconds < self._cache_ttl
        ):
            return dict(self._last_values)

        values = {}
        meta   = {}

        for name, fetcher in self.fetchers.items():
            try:
                data = fetcher.fetch()
                # Strip internal meta keys before merging
                clean = {k: v for k, v in data.items() if not k.startswith("_")}
                values.update(clean)
                meta[name] = {
                    "simulated": data.get("_simulated", True),
                    "source":    data.get("_source", "unknown"),
                    "keys":      list(clean.keys()),
                }
            except Exception as exc:
                meta[name] = {
                    "simulated": True,
                    "source":    "ERROR",
                    "error":     str(exc),
                    "keys":      [],
                }

        self._last_values = values
        self._last_meta   = meta
        self._last_fetch  = datetime.utcnow()

        return dict(values)

    def meta(self) -> Dict[str, Dict]:
        """Return provenance for last fetch."""
        return dict(self._last_meta)

    def full_report(self) -> Dict:
        """Run full registry assessment on current values."""
        from assumption_validator.registry import full_report as _full_report
        values = self.fetch()
        report = _full_report(values)
        report["_data_meta"] = self.meta()
        return report

    def simulated_count(self) -> int:
        """How many sources are currently simulated."""
        return sum(1 for m in self._last_meta.values() if m.get("simulated"))

    def live_count(self) -> int:
        """How many sources are currently live."""
        return sum(1 for m in self._last_meta.values() if not m.get("simulated"))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from assumption_validator.monitors import print_report

    print("NOAAAdapter — fetching all sources...\n")
    adapter = NOAAAdapter()
    report  = adapter.full_report()

    print_report(report, show_green=False)

    print("\nDATA PROVENANCE:")
    for source, info in adapter.meta().items():
        sim = "SIMULATED" if info.get("simulated") else "LIVE    "
        print(f"  [{sim}]  {source:<15}  {info.get('source','?')}")
        if info.get("error"):
            print(f"             error: {info['error']}")

    print(f"\nLive feeds: {adapter.live_count()}  "
          f"Simulated: {adapter.simulated_count()}")
