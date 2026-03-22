# assumption_validator/monitors.py
# assumption-validator
# CC0 — No Rights Reserved
#
# Universal live monitoring layer.
# No dependency on earth-systems-physics.
# Accepts any adapter as data source.
# Tracks drift, acceleration, time-to-red.
# Fires alerts on status change, acceleration, threshold approach.
# This is the thing that watches the thing.

import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from assumption_validator.registry import (
    REGISTRY,
    RiskLevel,
    full_report,
    detect_cascade_risk,
    global_confidence_multiplier,
    assess_values,
)


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class AssumptionRecord:
    """Single timestamped reading for one assumption."""
    timestamp : datetime
    value     : float
    status    : str
    penalty   : float
    proximity : float


@dataclass
class CascadeSnapshot:
    """Cascade state at one point in time."""
    timestamp  : datetime
    level      : str
    n_red      : int
    n_yellow   : int
    n_coupled  : int
    multiplier : float
    message    : str


@dataclass
class Alert:
    """Alert fired when something changes."""
    timestamp       : datetime
    assumption_id   : str
    assumption_name : str
    alert_type      : str   # STATUS_CHANGE | ACCELERATION | THRESHOLD_IMMINENT | CASCADE | POLL_ERROR
    previous_status : str
    current_status  : str
    message         : str
    hours_to_red    : Optional[float] = None
    cascade_level   : Optional[str]   = None


# ─────────────────────────────────────────────
# ROLLING STATE PER ASSUMPTION
# ─────────────────────────────────────────────

class MonitorState:
    """
    Rolling time series for one assumption.
    Tracks history, drift rate, acceleration, time-to-red.
    """

    def __init__(self, assumption_id: str, maxlen: int = 8760):
        self.assumption_id = assumption_id
        self.maxlen        = maxlen
        self.records: deque = deque(maxlen=maxlen)

        b = REGISTRY.get(assumption_id)
        self.name  = b.name  if b else assumption_id
        self.units = b.units if b else ""

    def push(self, record: AssumptionRecord):
        self.records.append(record)

    def latest(self) -> Optional[AssumptionRecord]:
        return self.records[-1] if self.records else None

    def values(self) -> List[float]:
        return [r.value for r in self.records if r.value is not None]

    def timestamps(self) -> List[datetime]:
        return [r.timestamp for r in self.records]

    def drift_rate(self, window: int = 24) -> Optional[float]:
        """
        Linear slope over last `window` records.
        Units: assumption units per record interval.
        """
        vals = self.values()
        if len(vals) < 2:
            return None
        n      = min(window, len(vals))
        recent = vals[-n:]
        if len(recent) < 2:
            return None
        x = np.arange(len(recent), dtype=float)
        return float(np.polyfit(x, recent, 1)[0])

    def acceleration(self, window: int = 168) -> Optional[float]:
        """
        Second derivative of drift.
        Positive = worsening faster.
        """
        vals = self.values()
        if len(vals) < 4:
            return None
        n      = min(window, len(vals))
        recent = np.array(vals[-n:], dtype=float)
        x      = np.arange(len(recent), dtype=float)
        coeffs = np.polyfit(x, recent, 2)
        return float(2 * coeffs[0])

    def time_to_red(self) -> Optional[float]:
        """
        Extrapolate current drift to RED threshold.
        Returns hours, or None if moving away from red.
        """
        boundary = REGISTRY.get(self.assumption_id)
        if boundary is None:
            return None
        latest = self.latest()
        if latest is None or latest.value is None:
            return None
        rate = self.drift_rate()
        if rate is None or rate == 0:
            return None

        current = latest.value
        red     = boundary.red_threshold

        if boundary.higher_is_worse:
            if current >= red:
                return 0.0
            if rate <= 0:
                return None
            return (red - current) / rate
        else:
            if current <= red:
                return 0.0
            if rate >= 0:
                return None
            return (current - red) / abs(rate)

    def status_history(self, n: int = 24) -> List[str]:
        return [r.status for r in list(self.records)[-n:]]

    def consecutive_degraded(self) -> int:
        """Count of consecutive most-recent records in YELLOW or RED."""
        count = 0
        for r in reversed(list(self.records)):
            if r.status in ("YELLOW", "RED"):
                count += 1
            else:
                break
        return count


# ─────────────────────────────────────────────
# UNIVERSAL MONITOR
# ─────────────────────────────────────────────

class UniversalMonitor:
    """
    Polls any adapter on a schedule.
    Tracks drift and fires alerts.

    The adapter must implement:
        adapter.fetch() -> Dict[str, Any]
        where keys match AssumptionBoundary.layer_key values.

    Usage
    -----
    from assumption_validator.adapters.noaa import NOAAAdapter

    monitor = UniversalMonitor(
        adapter          = NOAAAdapter(),
        poll_interval_s  = 3600,
        alert_callbacks  = [my_alert_handler],
    )
    monitor.start()
    report = monitor.current_report()
    monitor.stop()
    """

    def __init__(
        self,
        adapter,
        poll_interval_s  : float          = 3600.0,
        alert_callbacks  : List[Callable] = None,
    ):
        self.adapter          = adapter
        self.poll_interval    = poll_interval_s
        self.alert_callbacks  = alert_callbacks or []

        # Per-assumption rolling state
        self.states: Dict[str, MonitorState] = {
            aid: MonitorState(aid)
            for aid in REGISTRY
        }

        # Cascade history
        self.cascade_history: deque = deque(maxlen=8760)

        # Alert queue
        self._alerts     : deque        = deque(maxlen=1000)
        self._alert_lock : threading.Lock = threading.Lock()

        # Latest report cache
        self._latest_report : Optional[Dict] = None
        self._latest_values : Optional[Dict] = None
        self._report_lock   : threading.Lock = threading.Lock()

        # Thread control
        self._running    = False
        self._thread     = None
        self._poll_count = 0

    # ── PUBLIC API ───────────────────────────────────────────────────

    def start(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target = self._poll_loop,
            daemon = True,
            name   = "UniversalMonitor",
        )
        self._thread.start()

    def stop(self):
        """Stop background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def poll_once(self) -> Dict:
        """Single synchronous poll. Returns full report."""
        return self._do_poll()

    def current_report(self) -> Optional[Dict]:
        """Latest cached full report."""
        with self._report_lock:
            return self._latest_report

    def current_values(self) -> Optional[Dict]:
        """Latest raw values from adapter."""
        with self._report_lock:
            return self._latest_values

    def drain_alerts(self) -> List[Alert]:
        """Return and clear all pending alerts."""
        with self._alert_lock:
            alerts = list(self._alerts)
            self._alerts.clear()
        return alerts

    def assumption_trend(self, assumption_id: str) -> Dict:
        """Trend analysis for one assumption."""
        ms = self.states.get(assumption_id)
        if ms is None:
            return {"error": f"Unknown assumption: {assumption_id}"}

        latest = ms.latest()
        return {
            "assumption_id":        assumption_id,
            "name":                 ms.name,
            "units":                ms.units,
            "latest_value":         latest.value  if latest else None,
            "latest_status":        latest.status if latest else None,
            "drift_rate_per_poll":  ms.drift_rate(window=24),
            "acceleration":         ms.acceleration(window=168),
            "hours_to_red":         ms.time_to_red(),
            "consecutive_degraded": ms.consecutive_degraded(),
            "record_count":         len(ms.records),
            "status_history_24":    ms.status_history(24),
        }

    def all_trends(self) -> Dict[str, Dict]:
        """Trend analysis for all assumptions."""
        return {aid: self.assumption_trend(aid) for aid in self.states}

    def cascade_trend(self, n: int = 24) -> List[Dict]:
        """Recent cascade history."""
        return [
            {
                "timestamp":  s.timestamp.isoformat(),
                "level":      s.level,
                "n_red":      s.n_red,
                "n_yellow":   s.n_yellow,
                "n_coupled":  s.n_coupled,
                "multiplier": s.multiplier,
                "message":    s.message,
            }
            for s in list(self.cascade_history)[-n:]
        ]

    def swap_adapter(self, adapter):
        """Hot-swap the data adapter without stopping the monitor."""
        self.adapter = adapter

    # ── INTERNAL ─────────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            try:
                self._do_poll()
            except Exception as exc:
                self._raise_alert(Alert(
                    timestamp       = datetime.utcnow(),
                    assumption_id   = "monitor",
                    assumption_name = "Monitor Engine",
                    alert_type      = "POLL_ERROR",
                    previous_status = "UNKNOWN",
                    current_status  = "ERROR",
                    message         = f"Poll error: {exc}",
                ))
            time.sleep(self.poll_interval)

    def _do_poll(self) -> Dict:
        now = datetime.utcnow()

        # Fetch from adapter
        try:
            values = self.adapter.fetch()
        except Exception as exc:
            self._raise_alert(Alert(
                timestamp       = now,
                assumption_id   = "adapter",
                assumption_name = "Data Adapter",
                alert_type      = "POLL_ERROR",
                previous_status = "UNKNOWN",
                current_status  = "ERROR",
                message         = f"Adapter fetch failed: {exc}",
            ))
            values = {}

        # Full registry assessment
        report = full_report(values)

        # Previous statuses for change detection
        prev = {
            aid: (ms.latest().status if ms.latest() else "UNKNOWN")
            for aid, ms in self.states.items()
        }

        # Update rolling states
        for aid, assessment in report["assumptions"].items():
            ms = self.states.get(aid)
            if ms is None:
                continue

            value   = assessment.get("value")
            status  = assessment.get("status", "UNKNOWN")
            penalty = assessment.get("confidence_penalty", 0.0)
            prox    = assessment.get("proximity_to_red", 0.0)

            if isinstance(value, (int, float, np.floating)):
                record = AssumptionRecord(
                    timestamp = now,
                    value     = float(value),
                    status    = status,
                    penalty   = float(penalty) if isinstance(penalty, (int, float)) else 0.0,
                    proximity = float(prox)    if isinstance(prox,   (int, float)) else 0.0,
                )
                ms.push(record)

                # Alert: status change
                prev_status = prev.get(aid, "UNKNOWN")
                if status != prev_status and prev_status != "UNKNOWN":
                    self._raise_alert(Alert(
                        timestamp       = now,
                        assumption_id   = aid,
                        assumption_name = ms.name,
                        alert_type      = "STATUS_CHANGE",
                        previous_status = prev_status,
                        current_status  = status,
                        message         = (
                            f"{ms.name} changed {prev_status} → {status}. "
                            f"Value: {value:.4g} {ms.units}. "
                            f"{assessment.get('notes','')}"
                        ),
                    ))

                # Alert: acceleration
                accel = ms.acceleration()
                if accel is not None and status in ("YELLOW", "RED"):
                    boundary = REGISTRY.get(aid)
                    if boundary:
                        span = abs(boundary.red_threshold - boundary.green_range[0])
                        threshold = span * 0.001
                        if abs(accel) > threshold:
                            self._raise_alert(Alert(
                                timestamp       = now,
                                assumption_id   = aid,
                                assumption_name = ms.name,
                                alert_type      = "ACCELERATION",
                                previous_status = status,
                                current_status  = status,
                                message         = (
                                    f"{ms.name} drift accelerating. "
                                    f"Acceleration: {accel:.3g} {ms.units}/poll². "
                                    f"Current: {value:.4g} {ms.units}."
                                ),
                            ))

                # Alert: threshold imminent
                hours = ms.time_to_red()
                if hours is not None and 0 < hours < 720:
                    boundary = REGISTRY.get(aid)
                    self._raise_alert(Alert(
                        timestamp       = now,
                        assumption_id   = aid,
                        assumption_name = ms.name,
                        alert_type      = "THRESHOLD_IMMINENT",
                        previous_status = status,
                        current_status  = status,
                        message         = (
                            f"{ms.name} RED threshold in {hours:.0f} hours "
                            f"at current rate. "
                            f"Value: {value:.4g}, "
                            f"Threshold: {boundary.red_threshold:.4g} {ms.units}."
                            if boundary else
                            f"{ms.name} RED threshold in {hours:.0f} hours."
                        ),
                        hours_to_red = hours,
                    ))

        # Cascade snapshot
        cascade    = report["cascade"]
        multiplier = report["global_confidence_multiplier"]
        snap = CascadeSnapshot(
            timestamp  = now,
            level      = cascade["cascade_level"],
            n_red      = cascade["n_red"],
            n_yellow   = cascade["n_yellow"],
            n_coupled  = cascade["n_coupled_pairs"],
            multiplier = multiplier,
            message    = cascade["message"],
        )
        self.cascade_history.append(snap)

        # Alert: cascade escalation
        history = list(self.cascade_history)
        if len(history) >= 2:
            prev_snap = history[-2]
            levels    = ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
            if (snap.level in levels and prev_snap.level in levels and
                    levels.index(snap.level) > levels.index(prev_snap.level)):
                self._raise_alert(Alert(
                    timestamp       = now,
                    assumption_id   = "cascade",
                    assumption_name = "Cascade Engine",
                    alert_type      = "CASCADE",
                    previous_status = prev_snap.level,
                    current_status  = snap.level,
                    message         = (
                        f"Cascade escalated: {prev_snap.level} → {snap.level}. "
                        f"RED: {snap.n_red}, YELLOW: {snap.n_yellow}, "
                        f"Coupled pairs: {snap.n_coupled}. "
                        f"Confidence: {multiplier:.0%}. "
                        f"{cascade['message']}"
                    ),
                    cascade_level = snap.level,
                ))

        # Enrich report with trends
        report["trends"]          = self.all_trends()
        report["cascade_history"] = self.cascade_trend(n=24)
        report["poll_count"]      = self._poll_count
        report["timestamp"]       = now.isoformat()

        self._poll_count += 1

        with self._report_lock:
            self._latest_report = report
            self._latest_values = values

        return report

    def _raise_alert(self, alert: Alert):
        with self._alert_lock:
            self._alerts.append(alert)
        for cb in self.alert_callbacks:
            try:
                cb(alert)
            except Exception:
                pass


# ─────────────────────────────────────────────
# CONSOLE OUTPUT
# Formatted for mobile — full shape visible at once
# ─────────────────────────────────────────────

def print_report(report: Dict, show_green: bool = False):
    """Print monitor report. Single copyable block."""
    ts  = report.get("timestamp", "")[:19]
    s   = report.get("summary",  {})
    cas = report.get("cascade",  {})
    mul = report.get("global_confidence_multiplier", 1.0)

    print("=" * 56)
    print(f"ASSUMPTION VALIDATOR  {ts}")
    print("=" * 56)
    print(f"  Confidence : {mul:.0%}")
    print(f"  GREEN      : {s.get('green',  0)}")
    print(f"  YELLOW     : {s.get('yellow', 0)}")
    print(f"  RED        : {s.get('red',    0)}")
    print(f"  CASCADE    : {cas.get('cascade_level','?')}")
    print(f"  {cas.get('message','')}")
    print()

    for aid, data in sorted(report.get("assumptions", {}).items()):
        status = data.get("status", "?")
        if not show_green and status in ("GREEN", "UNKNOWN"):
            continue

        val   = data.get("value")
        units = data.get("units", "")
        val_s = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)

        trend = report.get("trends", {}).get(aid, {})
        drift = trend.get("drift_rate_per_poll")
        h2r   = trend.get("hours_to_red")
        d_s   = f"  drift {drift:+.3g}/poll" if drift is not None else ""
        h_s   = f"  ⚠ RED in {h2r:.0f}h"    if (h2r and h2r < 720) else ""

        marker = {"GREEN":"✓","YELLOW":"~","RED":"✗"}.get(status, "?")
        domain = data.get("domain", "")
        print(
            f"  [{marker}] {status:<7}  "
            f"{domain:<15}  "
            f"{data.get('name','')[:28]:<28}  "
            f"{val_s} {units}{d_s}{h_s}"
        )

    coupled = cas.get("coupled_degraded", [])
    if coupled:
        print()
        print("  ACTIVE COUPLING PATHS:")
        for pair in coupled:
            a, b  = pair[0], pair[1]
            na = REGISTRY.get(a)
            nb = REGISTRY.get(b)
            print(f"    {na.name if na else a}  →  {nb.name if nb else b}")

    irreversible = cas.get("irreversible_active", [])
    if irreversible:
        print()
        print("  IRREVERSIBLE THRESHOLDS ACTIVE:")
        for aid in irreversible:
            b = REGISTRY.get(aid)
            print(f"    ✗ {b.name if b else aid}")

    print("=" * 56)


def print_alert(alert: Alert):
    """Default alert callback."""
    icons = {
        "STATUS_CHANGE":      "⚡",
        "ACCELERATION":       "↑↑",
        "THRESHOLD_IMMINENT": "⏳",
        "CASCADE":            "🔴",
        "POLL_ERROR":         "⚠",
    }
    icon = icons.get(alert.alert_type, "!")
    print(f"\n{icon} ALERT [{alert.alert_type}] {alert.timestamp.isoformat()[:19]}")
    print(f"   {alert.assumption_name}")
    print(f"   {alert.previous_status} → {alert.current_status}")
    print(f"   {alert.message}")
    if alert.hours_to_red is not None:
        print(f"   Hours to RED: {alert.hours_to_red:.0f}")
    if alert.cascade_level:
        print(f"   Cascade level: {alert.cascade_level}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("assumption_validator / monitors.py")
    print("Requires an adapter. Example:\n")
    print("  from assumption_validator.adapters.generic import GenericAdapter")
    print("  from assumption_validator.monitors import UniversalMonitor, print_report")
    print()
    print("  adapter = GenericAdapter({'co2_ppm': 428.0, 'amoc_sv': 14.5})")
    print("  monitor = UniversalMonitor(adapter=adapter, poll_interval_s=60)")
    print("  report  = monitor.poll_once()")
    print("  print_report(report)")
