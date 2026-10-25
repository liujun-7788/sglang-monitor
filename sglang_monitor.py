#!/usr/bin/env python3
"""SGLang engine load monitor — community-style dashboard.

Collectors (background threads, cached snapshots):
  - service metrics: GET <engine>/metrics every 1s (gauges + cumulative counters)
  - host + GPUs:     /proc + nvidia-smi every 3s
  - log fingerprints: tail of the engine log every 5s
  - process info:    every 10s (uptime, restarts)
Frontend: single inline page, vanilla JS, 1s poll of /api/live.
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

PORT = int(os.environ.get("SGLANG_MONITOR_PORT", "7100"))
HISTORY_S = 2 * 3600

# SSRF gate: only this exact allowlisted endpoint may ever be fetched. It is a
# pinned literal-IP URL for the local sglang instance (the monitor's purpose);
# scheme/host/resolved-IP are asserted on every fetch and redirects are off.
SGLANG_HOST = os.environ.get("SGLANG_ENGINE_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("SGLANG_ENGINE_PORT", "7007"))
SGLANG_METRICS_URL = "http://%s:%d/metrics" % (SGLANG_HOST, SGLANG_PORT)
SGLANG_LOAD_URL = "http://%s:%d/get_load" % (SGLANG_HOST, SGLANG_PORT)
ALLOWED_FETCH_URLS = {SGLANG_METRICS_URL, SGLANG_LOAD_URL}


def _recompute_engine_urls():
    """Re-derive the allowlisted fetch endpoints (called after argparse)."""
    global SGLANG_METRICS_URL, SGLANG_LOAD_URL, ALLOWED_FETCH_URLS
    SGLANG_METRICS_URL = "http://%s:%d/metrics" % (SGLANG_HOST, SGLANG_PORT)
    SGLANG_LOAD_URL = "http://%s:%d/get_load" % (SGLANG_HOST, SGLANG_PORT)
    ALLOWED_FETCH_URLS = {SGLANG_METRICS_URL, SGLANG_LOAD_URL}


def _engine_api_key() -> str:
    """The engine's --api-key. Resolution order: SGLANG_API_KEY env (the
    --api-key CLI flag sets it too), then parsed from the launcher script
    (--launcher / SGLANG_LAUNCHER). /get_load sits behind the key; /metrics
    is exempt."""
    global _api_key_cache
    if _api_key_cache is None:
        key = os.environ.get("SGLANG_API_KEY", "")
        if not key and LAUNCHER_PATH:
            try:
                with open(LAUNCHER_PATH, encoding="utf-8") as fh:
                    m = re.search(r"--api-key\s+(\S+)", fh.read())
                # the launcher's EXTRA_ARGS string ends with a closing quote
                # that \S+ would swallow
                key = (m.group(1).rstrip("\"'") if m else "")
            except OSError:
                key = ""
        _api_key_cache = key
    return _api_key_cache
PERSIST_DIR = os.environ.get(
    "SGLANG_MONITOR_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
PERSIST_INTERVAL_S = 60.0


def _fetch_allowlisted(url: str, timeout: float = 5.0,
                       req: "urllib.request.Request" = None):
    import ipaddress
    import socket
    import urllib.error
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    if url not in ALLOWED_FETCH_URLS or parsed.scheme != "http":
        raise ValueError("blocked non-allowlisted URL: %r" % url)
    resolved = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
    if str(resolved) != SGLANG_HOST:
        raise ValueError("blocked rebinding to %s" % resolved)

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        return opener.open(req if req is not None else url, timeout=timeout)
    except urllib.error.HTTPError:
        raise


LOG_CANDIDATES = ["/tmp/sglang.log"]


def _pick_log_path() -> str:
    """Newest engine log next to ENGINE_LOG.

    A launcher may rotate the log file name across restarts (mylog.log,
    mylog_v2.log, ...), so follow the newest matching log instead of a fixed
    whitelist. The live engine keeps its log warm through access-log lines,
    so newest-mtime tracks the running engine.
    """
    import glob
    best = None
    base_dir = os.path.dirname(os.path.abspath(ENGINE_LOG)) or "."
    stem = os.path.basename(ENGINE_LOG)
    if stem.endswith(".log"):
        stem = stem[:-4]
    for p in glob.glob(os.path.join(base_dir, stem + "*.log")):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if best is None or mt > best[0]:
            best = (mt, p)
    if best is None:
        return ENGINE_LOG
    return best[1]

_lock = threading.Lock()
state = {
    "service": {},
    "service_ts": 0,
    "host": {},
    "gpus": [],
    "host_ts": 0,
    "log": {},
    "log_ts": 0,
    "logtail": [],
    "proc": {},
    "proc_ts": 0,
    "cumulative": {},
    "queue_log": [],
}
hist = {k: deque(maxlen=HISTORY_S // 3) for k in (
    "gen_tps", "running", "queued", "token_usage", "cache_hit", "accept_len",
    "cpu", "mem_avail_gb", "gpu_mem_used", "gpu_util")}
_hist_lock = threading.Lock()

# cumulative counters are summed across label variants; gauges take tp_rank 0
SUM_TOTALS = {"prompt_tokens_total", "generation_tokens_total",
              "cached_tokens_total", "evicted_tokens_total",
              "num_requests_total", "hicache_backup_tokens_total",
              "hicache_dropped_tokens_total", "http_requests_total"}
GAUGE_PICK = ["num_running_reqs", "num_queue_reqs", "token_usage",
              "full_token_usage", "cache_hit_rate", "gen_throughput",
              "num_retracted_reqs", "kv_used_tokens", "kv_available_tokens",
              "max_total_num_tokens", "mamba_used_tokens", "new_token_ratio",
              "num_grammar_queue_reqs", "num_prefill_bootstrap_queue_reqs",
              "num_prefill_inflight_queue_reqs",
              "num_decode_prealloc_queue_reqs",
              "num_decode_transfer_queue_reqs",
              # v6 additions
              "hicache_host_used_tokens", "hicache_host_total_tokens",
              "spec_accept_length", "spec_accept_rate", "utilization",
              "http_requests_active", "graph_memory_usage_gb",
              "kv_cache_memory_usage_gb", "weight_memory_usage_gb",
              "startup_available_gpu_memory_gb", "num_paused_reqs",
              "kv_evictable_tokens", "mamba_evictable_tokens",
              "mamba_available_tokens", "pending_prealloc_token_usage"]
line_re = re.compile(r'^(sglang:[a-z_0-9]+)(?:\{[^}]*\})?\s+([-0-9.eE+]+)$')
bucket_re = re.compile(
    r'^sglang:([a-z_0-9]+)_bucket\{[^}]*le="([0-9.eE+]+)"[^}]*\}\s+([-0-9.eE+]+)$')
# user-facing inference requests only (excludes the monitor's own /metrics scrapes)
user_req_re = re.compile(
    r'^sglang:http_requests_total\{[^}]*endpoint="(?:/v1/chat/completions|'
    r'/v1/completions|/generate)"[^}]*\}\s+([-0-9.eE+]+)$')
# per-stage latency buckets (tp_rank 0 only; labels are alphabetical, so the
# whole label block is captured and individual labels extracted order-free)
stage_bucket_re = re.compile(
    r'^sglang:per_stage_req_latency_seconds_bucket\{([^}]*)\}\s+([-0-9.eE+]+)$')
stage_label_res = (
    re.compile(r'stage="([a-z_]+)"'),
    re.compile(r'tp_rank="(\d+)"'),
    re.compile(r'le="([0-9.eE+]+)"'),
)
# user-facing responses by status code (error counting)
resp_re = re.compile(
    r'^sglang:http_responses_total\{[^}]*endpoint="(?:/v1/chat/completions|'
    r'/v1/completions|/generate)"[^}]*status_code="(\d+)"[^}]*\}\s+([-0-9.eE+]+)$')
# scheduler idle counter (tp_rank 0)
idle_re = re.compile(
    r'^sglang:scheduler_idle_seconds_total\{[^}]*tp_rank="0"[^}]*\}\s+([-0-9.eE+]+)$')
# histogram buckets summed across label variants, keyed by upper bound
_buckets = {}
# request-rate tracking (value cached so every snapshot carries it)
_prev_req_total = {"ts": 0.0, "val": 0.0}
_req_rate = {"v": None}


def _percentile(buckets: dict, q: float):
    """q in [0,1]; returns the bucket upper bound crossing q (coarse, standard)."""
    if not buckets:
        return None
    items = sorted(buckets.items())
    total = items[-1][1]
    if total <= 0:
        return None
    target = q * total
    for ub, cum in items:
        if cum >= target:
            return ub
    return items[-1][0]


def _metrics_thread():
    while True:
        snap = {"ok": False}
        try:
            with _fetch_allowlisted(SGLANG_METRICS_URL, timeout=5) as r:
                body = r.read().decode("utf-8", "replace")
            gauges = {}
            totals = {}
            sums = {}
            buckets = {}
            stage_buckets = {}
            user_reqs = 0.0
            resp_total = 0.0
            resp_err = 0.0
            idle_s = None
            for line in body.splitlines():
                if not line.startswith("sglang:"):
                    continue
                mu = user_req_re.match(line)
                if mu:
                    user_reqs += float(mu.group(1))
                    continue
                ms_ = stage_bucket_re.match(line)
                if ms_:
                    labels = ms_.group(1)
                    mst = stage_label_res[0].search(labels)
                    mtp = stage_label_res[1].search(labels)
                    mle = stage_label_res[2].search(labels)
                    if mst and mtp and mle and mtp.group(1) == "0":
                        st = stage_buckets.setdefault(mst.group(1), {})
                        le = float(mle.group(1))
                        st[le] = st.get(le, 0.0) + float(ms_.group(2))
                    continue
                mr_ = resp_re.match(line)
                if mr_:
                    v = float(mr_.group(2))
                    resp_total += v
                    if int(mr_.group(1)) >= 400:
                        resp_err += v
                    continue
                mi_ = idle_re.match(line)
                if mi_:
                    idle_s = float(mi_.group(1))
                    continue
                mb = bucket_re.match(line)
                if mb:
                    base, le, val = mb.group(1), float(mb.group(2)), float(mb.group(3))
                    b = buckets.setdefault(base, {})
                    b[le] = b.get(le, 0.0) + val
                    continue
                m = line_re.match(line)
                if not m:
                    continue
                name, val = m.group(1), float(m.group(2))
                base = name[len("sglang:"):]
                if base in SUM_TOTALS:
                    totals[base] = totals.get(base, 0.0) + val
                elif base in GAUGE_PICK:
                    if base not in gauges:
                        gauges[base] = val
                if base.endswith(("_sum", "_count")):
                    sums[name] = val
            for base, key in (("time_to_first_token", "ttft_mean_ms"),
                              ("inter_token_latency", "itl_mean_ms"),
                              ("queue_time", "queue_wait_mean_s")):
                sname = [n for n in sums if base in n and n.endswith("_sum")]
                cname = [n for n in sums if base in n and n.endswith("_count")]
                if sname and cname and sums[cname[0]] > 0:
                    snap[key] = round(sums[sname[0]] / sums[cname[0]] * 1000.0, 1)
            # percentile latencies from histogram buckets (all-time cumulative)
            for base, key in (("time_to_first_token_seconds", "ttft"),
                              ("inter_token_latency_seconds", "itl"),
                              ("e2e_request_latency_seconds", "e2e"),
                              ("queue_time_seconds", "queue"),
                              ("eviction_duration_seconds", "evict")):
                b = buckets.get(base)
                if b:
                    p50 = _percentile(b, 0.50)
                    p90 = _percentile(b, 0.90)
                    p99 = _percentile(b, 0.99)
                    if p50 is not None:
                        if base == "e2e_request_latency_seconds":
                            snap[key + "_p50_s"] = p50
                            snap[key + "_p90_s"] = p90
                        else:
                            snap[key + "_p50_ms"] = round(p50 * 1000, 1)
                            snap[key + "_p90_ms"] = round(p90 * 1000, 1)
                            snap[key + "_p99_ms"] = round(p99 * 1000, 1)
            # request rate (per minute) from user-facing inference requests only
            cur_total = user_reqs
            now_ts = time.time()
            p = _prev_req_total
            if p["ts"] and now_ts - p["ts"] >= 5:
                _req_rate["v"] = round(
                    (cur_total - p["val"]) / (now_ts - p["ts"]) * 60.0, 1)
                p["ts"], p["val"] = now_ts, cur_total
            elif not p["ts"]:
                p["ts"], p["val"] = now_ts, cur_total
            snap["req_per_min"] = _req_rate["v"]
            # v6b: errors / per-stage latency / scheduler idle
            snap["http_err_count"] = resp_err
            snap["http_resp_total"] = resp_total
            if idle_s is not None:
                snap["sched_idle_s"] = round(idle_s, 1)
            stage_p90 = {}
            for st, b in stage_buckets.items():
                p = _percentile(b, 0.90)
                if p is not None:
                    stage_p90[st] = round(p * 1000, 1)
            if stage_p90:
                snap["stage_p90_ms"] = stage_p90
            snap.update(gauges)
            snap["prompt_tokens"] = totals.get("prompt_tokens_total", 0.0)
            snap["generation_tokens"] = totals.get("generation_tokens_total", 0.0)
            snap["cached_tokens"] = totals.get("cached_tokens_total", 0.0)
            snap["evicted_tokens"] = totals.get("evicted_tokens_total", 0.0)
            snap["requests_total"] = totals.get("num_requests_total", 0.0)
            snap["hicache_backup_tokens_total"] = totals.get(
                "hicache_backup_tokens_total", 0.0)
            snap["hicache_dropped_tokens_total"] = totals.get(
                "hicache_dropped_tokens_total", 0.0)
            try:
                load_req = urllib.request.Request(
                    SGLANG_LOAD_URL,
                    headers={"Authorization": "Bearer " + _engine_api_key()})
                with _fetch_allowlisted(load_req.full_url, timeout=3,
                                        req=load_req) as r:
                    loads = json.load(r)
                if isinstance(loads, list) and loads:
                    snap["waiting_reqs"] = loads[0].get("num_waiting_reqs", 0)
                    snap["pending_tokens"] = loads[0].get("num_pending_tokens", 0)
            except Exception:
                pass
            snap["ok"] = True
        except Exception as exc:
            snap["err"] = str(exc)[:120]
        with _lock:
            state["service"] = snap
            state["service_ts"] = time.time()
            if snap.get("ok"):
                try:
                    _odometer_tick(snap)
                    state["cumulative"] = _cum_public()
                    queue_log.append({"ts": int(time.time()),
                                      "queued": snap.get("num_queue_reqs", 0.0),
                                      "running": snap.get("num_running_reqs", 0.0),
                                      "pending_tokens": snap.get("pending_tokens")})
                except Exception:
                    pass
                with _hist_lock:
                    hist["gen_tps"].append(snap.get("gen_throughput", 0.0))
                    hist["running"].append(snap.get("num_running_reqs", 0.0))
                    hist["queued"].append(snap.get("num_queue_reqs", 0.0))
                    hist["token_usage"].append(snap.get("full_token_usage",
                                                        snap.get("token_usage", 0.0)))
                    hist["cache_hit"].append(snap.get("cache_hit_rate", 0.0))
                state["queue_log"] = list(queue_log)[-10:]
        time.sleep(1.0)


# Cross-restart cumulative odometer: the engine's Prometheus counters are
# process-lifetime and reset on every sglang restart; here each 1s tick banks
# the non-negative delta into _cum so the all-time totals survive restarts
# (at most one 1s sample of throughput can be lost at a crash boundary).
_CUM_SRC = (("gen", "generation_tokens"), ("prompt", "prompt_tokens"),
            ("cached", "cached_tokens"), ("evict", "evicted_tokens"),
            ("requests", "requests_total"))
_cum = {"gen": 0.0, "prompt": 0.0, "cached": 0.0, "evict": 0.0,
        "requests": 0.0, "first_seen": None, "last": None}
queue_log = deque(maxlen=600)


def _odometer_tick(snap: dict) -> None:
    cur = {k: snap.get(src, 0.0) for k, src in _CUM_SRC}
    if _cum["last"] is not None:
        for k, _ in _CUM_SRC:
            delta = cur[k] - _cum["last"][k]
            if delta >= 0:
                _cum[k] = _cum.get(k, 0.0) + delta
    _cum["last"] = cur
    if _cum["first_seen"] is None:
        _cum["first_seen"] = time.strftime("%Y-%m-%d %H:%M")


def _host_thread():
    gpu_cmd = ["nvidia-smi",
               "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
               "clocks.sm,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits", "-i", "0,1,2,3,4"]
    prev_stat = None
    prev_io = {"t": None, "read_bytes": None, "write_bytes": None,
               "rx": None, "tx": None, "rapl": None}
    while True:
        snap = {}
        try:
            with open("/proc/meminfo") as fh:
                mi = {}
                for line in fh:
                    k, _, v = line.partition(":")
                    parts = v.split()
                    mi[k.strip()] = float(parts[0]) / (1024 * 1024) if parts else 0.0
            snap["mem_total_gb"] = round(mi.get("MemTotal", 0), 1)
            snap["mem_used_gb"] = round(mi.get("MemTotal", 0) - mi.get("MemAvailable", 0), 1)
            snap["mem_avail_gb"] = round(mi.get("MemAvailable", 0), 1)
            snap["swap_total_gb"] = round(mi.get("SwapTotal", 0), 1)
            snap["swap_used_gb"] = round(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0), 1)
            snap["shared_gb"] = round(mi.get("Shmem", 0), 1)
            with open("/proc/loadavg") as fh:
                la = fh.read().split()
            snap["load1"] = float(la[0])
            snap["load5"] = float(la[1])
            with open("/proc/stat") as fh:
                first = fh.readline().split()[1:]
            cpu_vals = [float(x) for x in first]
            if prev_stat is not None:
                d = [b - a for a, b in zip(prev_stat, cpu_vals)]
                tot = sum(d)
                snap["cpu_pct"] = round(100.0 * (tot - d[3]) / tot, 1) if tot > 0 else 0.0
            prev_stat = cpu_vals
            snap["cpu_threads"] = os.cpu_count() or 0
            try:
                freqs = []
                for cpu in ("cpu0", "cpu96"):
                    with open("/sys/devices/system/cpu/%s/cpufreq/"
                              "scaling_cur_freq" % cpu) as fh:
                        freqs.append(float(fh.read().strip()) / 1000.0)
                snap["cpu_mhz"] = round(sum(freqs) / len(freqs), 0)
            except OSError:
                snap["cpu_mhz"] = None
            try:
                st = os.statvfs("/data")
                snap["data_free_gb"] = round(st.f_bavail * st.f_frsize / (1024 ** 3), 1)
                snap["data_total_gb"] = round(st.f_blocks * st.f_frsize / (1024 ** 3), 1)
            except OSError:
                pass
        except Exception as exc:
            snap["err"] = str(exc)[:120]
        gpus = []
        try:
            out = subprocess.run(gpu_cmd, capture_output=True, text=True,
                                 timeout=10).stdout
            for row in out.strip().splitlines():
                p = [x.strip() for x in row.split(",")]
                if len(p) >= 8:
                    gpus.append({
                        "index": int(float(p[0])), "name": p[1],
                        "util": float(p[2]), "mem_used": float(p[3]),
                        "mem_total": float(p[4]), "sm_clock": float(p[5]),
                        "power_w": float(p[6]), "temp_c": float(p[7]),
                    })
        except Exception:
            pass
        snap["gpus"] = gpus
        snap_ts = time.time()

        # ---- disk, network, power ----
        try:
            dt = None if prev_io["t"] is None else snap_ts - prev_io["t"]

            # Disk: the single RAID volume (sda) backs /data. Sectors are 512 B.
            rd_bytes = wr_bytes = None
            try:
                with open("/proc/diskstats") as fh:
                    for ln in fh:
                        f = ln.split()
                        if len(f) >= 10 and f[2] == "sda":
                            rd_bytes = float(f[5]) * 512.0
                            wr_bytes = float(f[9]) * 512.0
                            break
            except OSError:
                pass
            disk = _energy["disk"]
            if rd_bytes is not None and prev_io["read_bytes"] is not None and dt:
                disk["read_bps"] = max(0.0, (rd_bytes - prev_io["read_bytes"]) / dt)
                disk["write_bps"] = max(0.0, (wr_bytes - prev_io["write_bytes"]) / dt)
                disk["read_total_gb"] += max(0.0, rd_bytes - prev_io["read_bytes"]) / 2 ** 30
                disk["write_total_gb"] += max(0.0, wr_bytes - prev_io["write_bytes"]) / 2 ** 30
            if rd_bytes is not None:
                prev_io["read_bytes"] = rd_bytes
                prev_io["write_bytes"] = wr_bytes
            # Capacity of /data (same numbers as the existing host snapshot).
            if snap.get("data_total_gb"):
                disk["total_gb"] = snap["data_total_gb"]
                disk["free_gb"] = snap.get("data_free_gb", 0.0)
                disk["used_gb"] = max(0.0, disk["total_gb"] - disk["free_gb"])

            # Network: the primary NIC, plus a count of interfaces that are up.
            rx = tx = None
            try:
                up = 0
                for name in os.listdir("/sys/class/net"):
                    try:
                        if open("/sys/class/net/%s/operstate" % name).read().strip() == "up":
                            up += 1
                    except OSError:
                        pass
                _energy["net"]["iface_total"] = up
                with open("/proc/net/dev") as fh:
                    for ln in fh.readlines()[2:]:
                        if ":" not in ln:
                            continue
                        ifname, rest = ln.split(":", 1)
                        if ifname.strip() == "eno1":
                            cols = rest.split()
                            rx = float(cols[0])
                            tx = float(cols[8])
                            break
            except OSError:
                pass
            net = _energy["net"]
            if rx is not None and prev_io["rx"] is not None and dt:
                net["rx_bps"] = max(0.0, (rx - prev_io["rx"]) / dt)
                net["tx_bps"] = max(0.0, (tx - prev_io["tx"]) / dt)
                net["rx_total_gb"] += max(0.0, rx - prev_io["rx"]) / 2 ** 30
                net["tx_total_gb"] += max(0.0, tx - prev_io["tx"]) / 2 ** 30
            if rx is not None:
                prev_io["rx"] = rx
                prev_io["tx"] = tx
            try:
                net["up"] = (open("/sys/class/net/eno1/operstate").read().strip()
                             == "up")
            except OSError:
                net["up"] = False

            # CPU package energy (RAPL) needs root: the sysfs files are 0400.
            # sudo -n is read-only and already permitted for this user.
            rapl = None
            try:
                out = subprocess.run(
                    ["sudo", "-n", "sh", "-c",
                     "cat /sys/class/powercap/intel-rapl:0/energy_uj; "
                     "cat /sys/class/powercap/intel-rapl:1/energy_uj"],
                    capture_output=True, text=True, timeout=5).stdout.split()
                vals = [float(x) for x in out if x.strip().isdigit()]
                if len(vals) == 2:
                    rapl = sum(vals)
            except Exception:
                rapl = None
            cpu_w = None
            if rapl is not None and prev_io["rapl"] is not None and dt:
                d_uj = rapl - prev_io["rapl"]
                if 0 <= d_uj <= 1e12:      # ignore counter resets
                    cpu_w = d_uj / 1e6 / dt
            if rapl is not None:
                prev_io["rapl"] = rapl
            _energy["cpu_ok"] = cpu_w is not None
            _energy["cpu_w"] = cpu_w

            gpu_w = sum(g.get("power_w", 0.0) for g in gpus) if gpus else 0.0
            _energy["gpu_w"] = gpu_w
            _energy["power_w"] = (gpu_w + (cpu_w or 0.0)
                                  + float(_energy["compensate_w"]))
            _add_energy(_energy["power_w"], snap_ts)
            snap["gpu_w"] = round(gpu_w, 1)
            snap["cpu_w"] = None if cpu_w is None else round(cpu_w, 1)
            snap["power_w"] = round(_energy["power_w"], 1)
            snap["disk"] = dict(disk)
            snap["net"] = dict(net)
        except Exception as exc:
            snap["io_err"] = str(exc)[:120]
        prev_io["t"] = snap_ts

        with _lock:
            state["host"] = snap
            state["gpus"] = gpus
            state["host_ts"] = snap_ts
            if gpus:
                v100 = [g for g in gpus if "V100" in g["name"]]
                with _hist_lock:
                    if v100:
                        hist["gpu_mem_used"].append(
                            sum(g["mem_used"] for g in v100) / len(v100))
                        hist["gpu_util"].append(
                            sum(g["util"] for g in v100) / len(v100))
                    if snap.get("cpu_pct") is not None:
                        hist["cpu"].append(snap["cpu_pct"])
                    hist["mem_avail_gb"].append(snap.get("mem_avail_gb", 0.0))
        time.sleep(3.0)


LINE_RE = re.compile(r"^\[2026-\d\d-\d\d (\d\d:\d\d:\d\d)")
DEC_RE = re.compile(r"#running-req: (\d+),.*accept len: ([0-9.]+), accept rate: ([0-9.]+).*gen throughput \(token/s\): ([0-9.]+)")
# uvicorn access line at completion: time + client address
ACCESS_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*?(\d+\.\d+\.\d+\.\d+):\d+ - "
    r"\"POST /v1/(?:chat/)?completions")
# per-request stats line (needs --enable-request-time-stats-logging):
# ReqTimeStats(rid=..., input_len=N, cached_input_len=M, output_len=K, ...):
# queue_duration=Xms, forward_duration=Yms, entry_time=T[, ttft_ms=..][, decode_ms=..]
STATS_RE = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].*ReqTimeStats\(rid=([^,]+), "
    r"input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), attempts=\d+"
    r"[^)]*\): queue_duration=([0-9.]+)ms, forward_duration=([0-9.]+)ms, "
    r"entry_time=([0-9.]+)(?:, ttft_ms=([0-9.]+))?(?:, decode_ms=([0-9.]+))?")

access_log = deque(maxlen=300)
req_rows = deque(maxlen=100)


def _match_client_ip(log_ts: str) -> str:
    """Pair a completed request with its uvicorn access line (same second-ish)."""
    for i in range(len(access_log) - 1, -1, -1):
        ts, ip, used = access_log[i]
        if used:
            continue
        if abs(_ts_delta_s(ts, log_ts)) <= 3:
            access_log[i] = (ts, ip, True)
            return ip
    return "-"


def _ts_delta_s(a: str, b: str) -> float:
    import datetime
    fa = datetime.datetime.strptime(a, "%Y-%m-%d %H:%M:%S")
    fb = datetime.datetime.strptime(b, "%Y-%m-%d %H:%M:%S")
    return (fa - fb).total_seconds()


# A log truncated while its writer still holds a stale (non-O_APPEND) fd
# offset turns into a sparse file: the gap reads back as megabytes of NUL
# bytes, which would otherwise be kept in memory as one giant "line" and
# blow up /api/live (2026-10-19: 972 MB response -> blank page).
MAX_LOG_READ_BYTES = 8 * 1024 * 1024
MAX_TAIL_CHARS = 2000


def _log_write_cursor(path):
    """Byte offset where the engine will write next, read from /proc fdinfo.

    The engine and all of its workers inherit the launcher's fd, so they share
    one offset; a copytruncate leaves that offset far below the (now sparse)
    apparent end, and new lines land there. Returns -1 when no live writer
    holds the file (fall back to EOF)."""
    import glob
    best = -1
    for d in glob.glob("/proc/[0-9]*/fd/[12]"):
        try:
            if os.readlink(d) != path:
                continue
        except OSError:
            continue
        pid = d.split("/")[2]
        try:
            with open("/proc/%s/fdinfo/%s" % (pid, d.rsplit("/", 1)[-1])) as fi:
                for line in fi:
                    if line.startswith("pos:"):
                        pos = int(line.split(":", 1)[1])
                        if pos > best:
                            best = pos
                        break
        except (OSError, ValueError):
            continue
    return best


def _log_thread():
    counters = {"triton_loads": 0, "ar_failed": 0, "traceback": 0, "oom": 0}
    offset = 0
    latest = {}
    fingerprint = "device-loaded after serving started"
    cur_path = None
    while True:
        try:
            fresh = ""
            path = _pick_log_path()
            if path != cur_path:
                cur_path = path
                offset = 0  # new file: start from its end, skip old history
            with open(path, encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                # Follow the writer's cursor, not EOF: after a truncate under a
                # live writer the apparent size is mostly a NUL hole and new
                # lines land at the (much lower) fd offset.
                cursor = _log_write_cursor(path)
                end = size if cursor < 0 else cursor
                if offset > end:
                    offset = 0  # log file was recreated (engine restart)
                if offset == 0:
                    offset = end  # new file: skip history for the counters
                    if end > 0:
                        # ...but seed the tail panel with the last few KB so a
                        # freshly started monitor is not blank.
                        back = min(end, 65536)
                        fh.seek(end - back)
                        pre = fh.read(back).replace("\x00", "")
                        seed = [ln.strip() for ln in pre.splitlines() if ln.strip()]
                        seed = [ln[:MAX_TAIL_CHARS] for ln in seed][-8:]
                        if seed:
                            with _lock:
                                state["logtail"] = seed
                elif end > offset:
                    if end - offset > MAX_LOG_READ_BYTES:
                        offset = end - MAX_LOG_READ_BYTES
                    fh.seek(offset)
                    raw = fh.read(min(end - offset, MAX_LOG_READ_BYTES))
                    fresh = raw.replace("\x00", "")
                    offset += len(raw.encode("utf-8", "replace"))
            if fresh:
                for line in fresh.splitlines():
                    if fingerprint in line:
                        counters["triton_loads"] += 1
                    elif "Setup Custom allreduce failed" in line:
                        counters["ar_failed"] += 1
                    elif "Traceback" in line:
                        counters["traceback"] += 1
                    elif "out of memory" in line.lower():
                        counters["oom"] += 1
                    m = ACCESS_RE.search(line)
                    if m:
                        ts, ip = m.group(1), m.group(2)
                        # backfill the newest unmatched stats row first (the
                        # access line can land just after its stats line)
                        filled = False
                        for row in reversed(req_rows):
                            if (row.get("ip") == "-"
                                    and abs(_ts_delta_s(ts, row.get("ts_full", ts))) <= 3):
                                row["ip"] = ip
                                filled = True
                                break
                        if not filled:
                            access_log.append((ts, ip, False))
                    m = STATS_RE.search(line)
                    if m:
                        (log_ts, rid, in_len, cached, out_len, q_ms, f_ms,
                         _entry, ttft, dec) = m.groups()
                        in_len, cached, out_len = int(in_len), int(cached), int(out_len)
                        ttft_s = float(ttft) / 1000 if ttft else None
                        dec_s = float(dec) / 1000 if dec else None
                        pre_speed = (round((in_len - cached) / ttft_s, 1)
                                     if ttft_s and ttft_s > 0 else None)
                        out_speed = (round(out_len / dec_s, 1)
                                     if dec_s and dec_s > 0 else None)
                        req_rows.append({
                            "t": log_ts[11:], "ts_full": log_ts, "rid": rid[:18],
                            "ip": _match_client_ip(log_ts),
                            "in": in_len, "cached": cached, "out": out_len,
                            "queue_ms": float(q_ms), "fwd_ms": float(f_ms),
                            "ttft_ms": (round(float(ttft), 1) if ttft else None),
                            "pre_speed": pre_speed, "out_speed": out_speed,
                        })
                    m = DEC_RE.search(line)
                    if m:
                        latest = {"running": int(m.group(1)),
                                  "accept_len": float(m.group(2)),
                                  "accept_rate": float(m.group(3)),
                                  "log_gen_tps": float(m.group(4))}
                        with _hist_lock:
                            hist["accept_len"].append(latest["accept_len"])
                tail = [ln.strip() for ln in fresh.splitlines() if ln.strip()]
                tail = [ln[:MAX_TAIL_CHARS] for ln in tail][-8:]
                if tail:
                    with _lock:
                        st = state.get("logtail", [])
                        state["logtail"] = (st + tail)[-8:]
            with _lock:
                state["log"] = dict(counters, **latest)
                state["log_ts"] = time.time()
        except FileNotFoundError:
            offset = 0  # log rotated (restart)
        except Exception:
            pass
        time.sleep(5.0)


def _is_launcher_pid(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            argv = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return " -m sglang.launch_server" in argv and "python" in argv.split(" ")[0]


def _proc_thread():
    last_pid = None
    restarts = 0
    while True:
        snap = {}
        try:
            out = subprocess.run(["pgrep", "-f", r"python[0-9.]* -m sglang\.launch_server"],
                                 capture_output=True, text=True, timeout=10).stdout
            raw_pids = [int(x) for x in out.split()]
            # -f also matches a diagnostic shell that merely quotes the module
            # name; keep only pids whose argv is the launcher itself.
            pids = [p for p in raw_pids if _is_launcher_pid(p)]
            if pids:
                pid = pids[0]
                if last_pid is not None and pid != last_pid:
                    restarts += 1
                last_pid = pid
                with open("/proc/%d/stat" % pid) as fh:
                    fields = fh.read().split()
                hz = os.sysconf("SC_CLK_TCK")
                uptime_s = 0.0
                try:
                    with open("/proc/uptime") as fh:
                        sys_up = float(fh.read().split()[0])
                    uptime_s = sys_up - float(fields[21]) / hz
                except Exception:
                    pass
                snap["pid"] = pid
                snap["uptime_s"] = int(uptime_s)
                rss_kb = 0
                for p in raw_pids[1:6]:
                    try:
                        with open("/proc/%d/status" % p) as fh:
                            for line in fh:
                                if line.startswith("VmRSS:"):
                                    rss_kb += int(line.split()[1])
                                    break
                    except OSError:
                        pass
                snap["sched_rss_gb"] = round(rss_kb / (1024 * 1024), 1)
            else:
                snap["pid"] = None
        except Exception:
            pass
        snap["restarts"] = restarts
        with _lock:
            state["proc"] = snap
            state["proc_ts"] = time.time()
        time.sleep(10.0)


# Persistence: latest.json (snapshot), snapshots_<date>.jsonl (60s rows),
# daily_stats.json (per-day token deltas vs process-start baseline). Filenames
# are built only from this fixed directory and a strftime date (digits).
_day = {"date": time.strftime("%Y-%m-%d"), "gen0": None, "prompt0": None,
        "cached0": None, "evict0": None, "requests0": None,
        "peak_running": 0.0}


def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _load_daily_baseline() -> None:
    """Restore today's baseline after a monitor restart so daily deltas stay
    correct across restarts (values are process-lifetime counters)."""
    try:
        with open(os.path.join(PERSIST_DIR, "daily_stats.json")) as fh:
            d = json.load(fh)
        rec = d.get(_day["date"])
        if rec:
            for k in ("gen0", "prompt0", "cached0", "evict0", "requests0"):
                if rec.get(k) is not None:
                    _day[k] = rec[k]
            _day["peak_running"] = rec.get("peak_running", 0.0)
    except Exception:
        pass


def _load_pricing() -> dict:
    """Token pricing for the cost estimate (editable at runtime via /api/pricing).

    Defaults are sample CNY-per-1M-token prices shipped as a starting point;
    edit pricing.json (data dir) or the UI fields to match your provider.
    """
    try:
        with open(os.path.join(PERSIST_DIR, "pricing.json")) as fh:
            d = json.load(fh)
        if isinstance(d, dict):
            return {"input": float(d.get("input", 0.8)),
                    "output": float(d.get("output", 2.7)),
                    "cached": float(d.get("cached", 0.086))}
    except Exception:
        pass
    return {"input": 0.8, "output": 2.7, "cached": 0.086}


_pricing = _load_pricing()

# ---- energy accounting ----------------------------------------------------
# Whole-machine power = measured GPUs + measured CPU packages (RAPL) + a
# configurable compensation for hardware that cannot be measured from
# userspace (NVMe, NICs, mainboard, PSU loss). Energy is integrated on the
# host thread cadence and bucketed per day for the 7-day table.
ENERGY_PATH = os.path.join(PERSIST_DIR, "energy.json")
_energy = {
    "price": float(os.environ.get("SGLANG_MONITOR_ELECTRICITY_PRICE", "1.0")),
            # CNY per kWh (sample default; edit in the UI or env)
    "compensate_w": float(os.environ.get("SGLANG_MONITOR_COMPENSATE_W", "45.0")),
    # watts not covered by GPU/CPU sensors
    "total_kwh": 0.0,
    "days": {},               # "YYYY-MM-DD" -> {"kwh": float}
    "first_seen": None,
    "last_ts": None,
    "power_w": 0.0,
    "gpu_w": 0.0,
    "cpu_w": None,            # None when RAPL is unreadable
    "cpu_ok": False,
    "disk": {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0,
             "read_bps": 0.0, "write_bps": 0.0,
             "read_total_gb": 0.0, "write_total_gb": 0.0},
    "net": {"rx_bps": 0.0, "tx_bps": 0.0,
            "rx_total_gb": 0.0, "tx_total_gb": 0.0,
            "iface": "eno1", "up": True, "iface_total": 0},
}


def _load_energy() -> None:
    try:
        with open(ENERGY_PATH) as fh:
            d = json.load(fh)
        for k in ("price", "compensate_w", "total_kwh", "first_seen"):
            if d.get(k) is not None:
                _energy[k] = d[k]
        if isinstance(d.get("days"), dict):
            _energy["days"] = {str(k): {"kwh": float(v.get("kwh", 0.0))}
                               for k, v in d["days"].items()
                               if isinstance(v, dict)}
    except Exception:
        pass
    _energy["first_seen"] = _energy["first_seen"] or time.strftime("%Y-%m-%d")


def _save_energy() -> None:
    _atomic_write(ENERGY_PATH, json.dumps({
        "price": _energy["price"],
        "compensate_w": _energy["compensate_w"],
        "total_kwh": round(_energy["total_kwh"], 4),
        "first_seen": _energy["first_seen"],
        "days": {k: {"kwh": round(v["kwh"], 4)}
                 for k, v in sorted(_energy["days"].items())[-31:]},
    }, ensure_ascii=False, indent=1))


def _add_energy(power_w: float, now: float) -> None:
    last = _energy["last_ts"]
    _energy["last_ts"] = now
    if last is None or power_w <= 0:
        return
    dt = now - last
    # Ignore gaps (host thread restart, suspend): integrating them would
    # invent consumption for time we did not sample.
    if not (0 < dt <= 60):
        return
    kwh = power_w * dt / 3600.0 / 1000.0
    _energy["total_kwh"] += kwh
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    bucket = _energy["days"].setdefault(day, {"kwh": 0.0})
    bucket["kwh"] += kwh


def _energy_public() -> dict:
    price = float(_energy["price"])
    today = time.strftime("%Y-%m-%d")
    days = [{"date": k, "kwh": round(v["kwh"], 3),
             "cost": round(v["kwh"] * price, 2)}
            for k, v in sorted(_energy["days"].items())[-7:]]
    return {
        "power_w": round(_energy["power_w"], 1),
        "gpu_w": round(_energy["gpu_w"], 1),
        "cpu_w": (round(_energy["cpu_w"], 1)
                  if _energy["cpu_w"] is not None else None),
        "cpu_ok": bool(_energy["cpu_ok"]),
        "compensate_w": round(float(_energy["compensate_w"]), 1),
        "total_kwh": round(_energy["total_kwh"], 3),
        "cost": round(_energy["total_kwh"] * price, 2),
        "price": price,
        "first_seen": _energy["first_seen"],
        "today_kwh": round(_energy["days"].get(today, {}).get("kwh", 0.0), 3),
        "today_cost": round(_energy["days"].get(today, {}).get("kwh", 0.0)
                            * price, 2),
        "days": days,
    }


_load_energy()



def _cum_public() -> dict:
    out = {"generation_tokens": round(_cum["gen"]),
           "prompt_tokens": round(_cum["prompt"]),
           "cached_tokens": round(_cum["cached"]),
           "evicted_tokens": round(_cum["evict"]),
           "requests_total": round(_cum["requests"]),
           "first_seen": _cum["first_seen"]}
    daily = state.get("daily") or {}
    for key in ("prompt_tokens", "generation_tokens", "cached_tokens",
                "requests"):
        if daily.get(key) is not None:
            out["today_" + key] = daily[key]
    fresh_in = max(0.0, out["prompt_tokens"] - out["cached_tokens"])
    cost_in = fresh_in / 1e6 * _pricing["input"]
    cost_cached = out["cached_tokens"] / 1e6 * _pricing["cached"]
    cost_out = out["generation_tokens"] / 1e6 * _pricing["output"]
    out["cost_input"] = round(cost_in, 3)
    out["cost_cached"] = round(cost_cached, 3)
    out["cost_output"] = round(cost_out, 3)
    out["cost_total"] = round(cost_in + cost_cached + cost_out, 3)
    out["pricing"] = dict(_pricing)
    return out


def _load_cumulative() -> None:
    """Restore the cross-restart odometer totals written every tick."""
    try:
        with open(os.path.join(PERSIST_DIR, "cumulative.json")) as fh:
            d = json.load(fh)
        _cum["gen"] = float(d.get("generation_tokens") or 0.0)
        _cum["prompt"] = float(d.get("prompt_tokens") or 0.0)
        _cum["cached"] = float(d.get("cached_tokens") or 0.0)
        _cum["evict"] = float(d.get("evicted_tokens") or 0.0)
        _cum["requests"] = float(d.get("requests_total") or 0.0)
        _cum["first_seen"] = d.get("first_seen")
    except Exception:
        pass


def _write_cumulative() -> None:
    _atomic_write(
        os.path.join(PERSIST_DIR, "cumulative.json"),
        json.dumps({
            "first_seen": _cum["first_seen"],
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generation_tokens": round(_cum["gen"]),
            "prompt_tokens": round(_cum["prompt"]),
            "cached_tokens": round(_cum["cached"]),
            "evicted_tokens": round(_cum["evict"]),
            "requests_total": round(_cum["requests"]),
        }, ensure_ascii=False, indent=1))


def _persist_thread():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    _load_daily_baseline()
    _load_cumulative()
    while True:
        try:
            with _lock:
                svc = dict(state["service"])
                host = dict(state["host"])
                gpus = list(state["gpus"])
                log = dict(state["log"])
                proc = dict(state["proc"])
            now = int(time.time())
            today = time.strftime("%Y-%m-%d")
            if today != _day["date"]:
                _day.update({"date": today, "gen0": None, "prompt0": None,
                             "cached0": None, "evict0": None,
                             "requests0": None, "peak_running": 0.0})
            daily = None
            if svc.get("ok"):
                if _day["gen0"] is None:
                    _day["gen0"] = svc.get("generation_tokens", 0.0)
                    _day["prompt0"] = svc.get("prompt_tokens", 0.0)
                    _day["cached0"] = svc.get("cached_tokens", 0.0)
                    _day["evict0"] = svc.get("evicted_tokens", 0.0)
                    _day["requests0"] = svc.get("requests_total", 0.0)
                run = svc.get("num_running_reqs", 0.0)
                _day["peak_running"] = max(_day["peak_running"], run)
                # counter-reset guard: a sglang restart zeroes the process
                # counters; rebase so daily deltas never go negative
                for k, src in (("gen0", "generation_tokens"),
                               ("prompt0", "prompt_tokens"),
                               ("cached0", "cached_tokens"),
                               ("evict0", "evicted_tokens"),
                               ("requests0", "requests_total")):
                    cur = svc.get(src, 0.0)
                    if _day[k] is None or cur < _day[k]:
                        _day[k] = cur
                daily = {
                    "date": today,
                    "gen0": _day["gen0"], "prompt0": _day["prompt0"],
                    "cached0": _day["cached0"], "evict0": _day["evict0"],
                    "requests0": _day["requests0"],
                    "generation_tokens": round(svc.get("generation_tokens", 0.0)
                                               - (_day["gen0"] or 0.0)),
                    "prompt_tokens": round(svc.get("prompt_tokens", 0.0)
                                           - (_day["prompt0"] or 0.0)),
                    "cached_tokens": round(svc.get("cached_tokens", 0.0)
                                           - (_day["cached0"] or 0.0)),
                    "evicted_tokens": round(svc.get("evicted_tokens", 0.0)
                                            - (_day["evict0"] or 0.0)),
                    "requests": round(svc.get("requests_total", 0.0)
                                      - (_day["requests0"] or 0.0)),
                    "peak_running": _day["peak_running"],
                }
                with _lock:
                    state["daily"] = daily
            _save_energy()
            # Write the row even when the engine is down. Skipping it (the old
            # behaviour) left a silent hole in snapshots_*.jsonl: the 2026-10-19
            # outage showed up only as a 2h18m gap with no record of when or why.
            up = bool(svc.get("ok"))
            v100 = [g for g in gpus if "V100" in g.get("name", "")]
            gpu_mem = (round(sum(g.get("mem_used", 0) for g in v100)
                             / len(v100) / 1024, 2) if v100 else None)
            gpu_util = (round(sum(g.get("util", 0) for g in v100)
                              / len(v100), 1) if v100 else None)
            row = {"ts": now,
                   "up": up,
                   "gen_tps": svc.get("gen_throughput", 0.0) if up else 0.0,
                   "running": svc.get("num_running_reqs", 0.0) if up else 0.0,
                   "queued": svc.get("num_queue_reqs", 0.0) if up else 0.0,
                   "token_usage": (svc.get("full_token_usage",
                                           svc.get("token_usage", 0.0))
                                   if up else 0.0),
                   "cache_hit": svc.get("cache_hit_rate", 0.0) if up else 0.0,
                   "accept_len": log.get("accept_len"),
                   "cpu_pct": host.get("cpu_pct"),
                   "mem_avail_gb": host.get("mem_avail_gb"),
                   "gpu_mem_used_gb": gpu_mem,
                   "gpu_util": gpu_util,
                   "pid": proc.get("pid")}
            latest = {"ts": now, "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "up": up, "service": svc, "host": host, "gpus": gpus,
                      "log": log, "proc": proc, "daily": daily,
                      "engine": {"host": SGLANG_HOST, "port": SGLANG_PORT},
                      "auto": {"enabled": _auto["enabled"],
                               "last_result": _auto["last_result"],
                               "tripped": _auto["tripped"]},
                      "cumulative": _cum_public()}
            _atomic_write(os.path.join(PERSIST_DIR, "latest.json"),
                          json.dumps(latest, ensure_ascii=False, indent=1))
            _write_cumulative()
            with open(os.path.join(PERSIST_DIR, "snapshots_"
                      + today + ".jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if daily is not None:
                _atomic_write(os.path.join(PERSIST_DIR, "daily_stats.json"),
                              json.dumps({today: daily}, ensure_ascii=False,
                                         indent=1))
        except Exception:
            pass
        time.sleep(PERSIST_INTERVAL_S)


app = FastAPI(title="sglang monitor")


@app.get("/api/pricing")
def api_pricing_get():
    return {"ok": True, "pricing": dict(_pricing)}


@app.post("/api/pricing")
async def api_pricing_set(request: Request):
    global _pricing
    body = await request.json()
    try:
        p = {"input": float(body.get("input")), "output": float(body.get("output")),
             "cached": float(body.get("cached"))}
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "err": "价格必须是数字"}, status_code=400)
    if not all(0 <= v <= 10000 for v in p.values()):
        return JSONResponse({"ok": False, "err": "价格超出合理范围"}, status_code=400)
    _pricing = p
    os.makedirs(PERSIST_DIR, exist_ok=True)
    _atomic_write(os.path.join(PERSIST_DIR, "pricing.json"),
                  json.dumps(p, ensure_ascii=False, indent=1))
    with _lock:
        state["cumulative"] = _cum_public()
    return {"ok": True, "pricing": p}


@app.get("/api/energy")
def api_energy_get():
    return {"ok": True, "energy": _energy_public()}


@app.post("/api/energy")
async def api_energy_set(request: Request):
    """Update the electricity price and/or the unmeasured-hardware compensation."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "err": "bad json"}, status_code=400)
    try:
        if "price" in body:
            price = float(body["price"])
            if not (0.0 <= price <= 100.0):
                raise ValueError("price out of range")
            _energy["price"] = price
        if "compensate_w" in body:
            comp = float(body["compensate_w"])
            if not (0.0 <= comp <= 5000.0):
                raise ValueError("compensate_w out of range")
            _energy["compensate_w"] = comp
        _save_energy()
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "err": str(exc)[:120]}, status_code=400)
    return {"ok": True, "energy": _energy_public()}

# ---- engine control / launcher editing / log management --------------------
# All paths below come from CLI flags or env; no request input reaches them.
# The only user-supplied value is the launcher CONTENT (validated with bash -n).
# Without --launcher / SGLANG_LAUNCHER the control/restart features stay
# disabled; the dashboard still monitors a running engine read-only.
import glob

LAUNCHER_PATH = os.environ.get("SGLANG_LAUNCHER", "")
LAUNCHER_TMP = LAUNCHER_PATH + ".monitor-new" if LAUNCHER_PATH else ""
LAUNCHER_BAK = LAUNCHER_PATH + ".bak-monitor" if LAUNCHER_PATH else ""
ENGINE_LOG = os.environ.get("SGLANG_ENGINE_LOG", "/tmp/sglang.log")
ENGINE_LOG_ARCH = ENGINE_LOG + ".archived"
LOG_CAP_BYTES = int(os.environ.get("SGLANG_MONITOR_LOG_CAP_MB", "500")) * 1024 * 1024
_ctl_lock = threading.Lock()
_ctl_state = {"last_action": None, "last_result": None, "last_ts": None}

# ---- auto-restart supervisor -------------------------------------------
# The engine dies as a whole when one scheduler rank hits a CUDA OOM
# (scheduler.py sends SIGQUIT to the parent), and nothing else on this box
# supervises it: no systemd unit, no cron, no restart loop in the launcher.
# On 2026-10-19 it stayed down for two hours until someone clicked "启动".
# This thread only covers that gap; it deliberately does not try to fix the
# underlying OOM, so keep it paired with a real fix or it will loop.
AUTO_RESTART = os.environ.get("SGLANG_MONITOR_AUTO_RESTART", "0") == "1"
AUTO_DOWN_TICKS = int(os.environ.get("SGLANG_MONITOR_AUTO_DOWN_TICKS", "3"))
AUTO_MAX_PER_HOUR = int(os.environ.get("SGLANG_MONITOR_AUTO_MAX_PER_HOUR", "3"))
AUTO_MIN_GAP_S = 600.0          # never start more than once per 10 minutes
AUTO_LOAD_GRACE_S = 1500.0      # a starting engine is not "down" for 25 min
_auto = {"enabled": AUTO_RESTART, "down_ticks": 0, "starts": [],
         "last_start_ts": None, "last_result": None, "tripped": None,
         "started_at": None, "manual": False, "manual_note": None}


def _engine_pids():
    try:
        out = subprocess.run(["pgrep", "-f", "sglang.launch_server"],
                             capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def _launcher_running():
    """True while a deliberate stop/start cycle is in flight.

    A deliberate stop/start script (e.g. warm_boot.sh) stops the engine,
    prewarms the page cache and then starts it again — several minutes with
    no sglang.launch_server process. Without this guard the auto-restart
    thread would jump in mid-cycle and race the script.
    """
    try:
        out = subprocess.run(["pgrep", "-f", "warm_boot.sh"],
                             capture_output=True, text=True, timeout=10).stdout
        return bool(out.split())
    except Exception:
        return False


def _do_stop():
    pids = _engine_pids()
    if not pids:
        return "未在运行"
    subprocess.run(["kill", "-TERM", str(pids[0])], timeout=10)
    for _ in range(18):
        time.sleep(10)
        if not _engine_pids():
            return "已停止"
    subprocess.run(["pkill", "-KILL", "-f", "sglang::"], timeout=10)
    return "已强制停止"


def _do_start():
    if _engine_pids():
        return "已在运行"
    if not LAUNCHER_PATH:
        return "未配置启动脚本（--launcher / SGLANG_LAUNCHER）"
    with open(ENGINE_LOG, "ab") as fh:
        subprocess.Popen(["bash", LAUNCHER_PATH],
                         cwd=os.path.dirname(os.path.abspath(LAUNCHER_PATH)),
                         stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return "已启动（模型加载需要几分钟）"


def _auto_tick(now=None):
    """One watchdog decision. Split out so it can be tested without an engine.

    Returns a short reason string describing what this tick decided.
    """
    now = time.time() if now is None else now
    if _engine_pids():
        _auto["down_ticks"] = 0
        _auto["tripped"] = None
        # Remember that this start did produce a running engine. Without it a
        # crash two minutes after a successful start would be mistaken for a
        # still-loading engine and delayed by the whole grace window.
        if _auto["last_start_ts"] is not None and _auto["started_at"] is None:
            _auto["started_at"] = now
        return "up"
    if _launcher_running():
        # A deliberate stop/start cycle owns the engine right now.
        _auto["down_ticks"] = 0
        return "launcher-cycle"
    _auto["down_ticks"] += 1
    if not _auto["enabled"]:
        return "disabled"
    if _auto.get("manual"):
        return "manual-busy"
    never_came_up = (_auto["last_start_ts"] is not None
                     and _auto["started_at"] is None)
    if (never_came_up
            and now - _auto["last_start_ts"] < AUTO_LOAD_GRACE_S):
        return "loading"
    _auto["starts"] = [t for t in _auto["starts"] if now - t < 3600.0]
    if len(_auto["starts"]) >= AUTO_MAX_PER_HOUR:
        _auto["tripped"] = ("每小时自动重启已达上限 %d 次，暂停自动拉起"
                            % AUTO_MAX_PER_HOUR)
        return "tripped"
    if (_auto["last_start_ts"] is not None
            and now - _auto["last_start_ts"] < AUTO_MIN_GAP_S):
        return "min-gap"
    if _auto["down_ticks"] < AUTO_DOWN_TICKS:
        return "debounce"
    res = _do_start()
    _auto["last_start_ts"] = now
    _auto["started_at"] = None
    _auto["starts"].append(now)
    _auto["last_result"] = res
    _auto["down_ticks"] = 0
    with _ctl_lock:
        _ctl_state.update(last_action="auto-start", last_result=res, last_ts=now)
    return "started"


def _auto_thread():
    """Start the engine when it stays down, with debounce and a circuit breaker.

    Guards against the failure mode this is meant for (crash -> hours of
    silence) turning into something worse (crash loop hammering the box):
    three consecutive misses before acting, at most one start per 10 minutes
    and AUTO_MAX_PER_HOUR per hour, and stop trying once the breaker trips.
    """
    while True:
        try:
            _auto_tick()
        except Exception as exc:
            _auto["last_result"] = "自动拉起异常: %s" % exc
        time.sleep(10.0)


@app.get("/api/auto")
def api_auto_get():
    return {"enabled": _auto["enabled"],
            "down_ticks": _auto["down_ticks"],
            "starts_last_hour": len([t for t in _auto["starts"]
                                     if time.time() - t < 3600.0]),
            "max_per_hour": AUTO_MAX_PER_HOUR,
            "last_start_ts": _auto["last_start_ts"],
            "last_result": _auto["last_result"],
            "tripped": _auto["tripped"],
            "manual_note": _auto["manual_note"]}


@app.post("/api/auto/{action}")
def api_auto_set(action: str):
    if action not in ("on", "off"):
        return JSONResponse({"ok": False, "err": "bad action"}, status_code=400)
    _auto["enabled"] = action == "on"
    if action == "on":
        _auto["down_ticks"] = 0
        _auto["tripped"] = None
    return {"ok": True, "enabled": _auto["enabled"]}


@app.post("/api/ctl/{action}")
def api_ctl(action: str):
    if action not in ("start", "stop", "restart"):
        return JSONResponse({"ok": False, "err": "bad action"}, status_code=400)

    def _run():
        with _ctl_lock:
            _ctl_state.update(last_action=action, last_ts=time.time(),
                              last_result="执行中…")
            _auto["manual"] = True
            try:
                if action == "stop":
                    res = _do_stop()
                    # A deliberate stop must stay stopped: otherwise the
                    # watchdog pulls it straight back up and the button
                    # looks broken.
                    _auto["enabled"] = False
                    _auto["manual_note"] = "手动停止，已关闭自动拉起"
                elif action == "start":
                    res = _do_start()
                else:
                    res = _do_stop() + " → " + _do_start()
            except Exception as exc:
                res = "错误: %s" % exc
            finally:
                _auto["manual"] = False
            _ctl_state.update(last_result=res, last_ts=time.time())

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "action": action}


@app.get("/api/params")
def api_params_get():
    if not LAUNCHER_PATH:
        return JSONResponse(
            {"ok": False,
             "err": "未配置启动脚本（--launcher / SGLANG_LAUNCHER）"},
            status_code=400)
    try:
        with open(LAUNCHER_PATH, encoding="utf-8") as fh:
            content = fh.read(300000)
    except OSError as exc:
        return JSONResponse({"ok": False, "err": str(exc)}, status_code=500)
    return {"ok": True, "path": LAUNCHER_PATH, "file": "launcher",
            "content": content}


@app.post("/api/params")
async def api_params_save(request: Request):
    body = await request.json()
    content = body.get("content") or ""
    file = body.get("file") or "launcher"
    if not content.startswith("#!") or "sglang" not in content:
        return JSONResponse({"ok": False, "err": "内容看起来不是启动脚本"},
                            status_code=400)
    if not LAUNCHER_PATH:
        return JSONResponse(
            {"ok": False,
             "err": "未配置启动脚本（--launcher / SGLANG_LAUNCHER）"},
            status_code=400)
    base = LAUNCHER_PATH
    tmp = base + ".monitor-new"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True,
                       timeout=15)
    if r.returncode != 0:
        return JSONResponse({"ok": False,
                             "err": "语法校验失败: " + (r.stderr or "")[:300]},
                            status_code=400)
    subprocess.run(["cp", "-f", base, base + ".bak-monitor"], timeout=15)
    os.replace(tmp, base)
    return {"ok": True, "file": file,
            "note": "已保存（备份 .bak-monitor）；重启后生效"}


def _log_stats():
    files = []
    total = 0
    for p in sorted(glob.glob(ENGINE_LOG + "*")):
        try:
            st = os.stat(p)
        except OSError:
            continue
        files.append({"path": p, "size": st.st_size})
        total += st.st_size
    live = _pick_log_path()
    cur = 0
    try:
        cur = os.path.getsize(live)
    except OSError:
        pass
    return {"files": files, "total": total, "current": cur,
            "current_path": live, "cap": LOG_CAP_BYTES}


@app.get("/api/logs")
def api_logs():
    return _log_stats()


@app.post("/api/logs/clean")
def api_logs_clean():
    freed = 0
    keep = {_pick_log_path(), ENGINE_LOG}
    for p in glob.glob("/tmp/sglang*.log*"):
        if p in keep:
            continue
        try:
            freed += os.path.getsize(p)
            os.remove(p)
        except OSError:
            pass
    return {"ok": True, "freed": freed}


def _log_cap_thread():
    while True:
        try:
            live = _pick_log_path()
            size = os.path.getsize(live)
            if size > LOG_CAP_BYTES:
                st = os.stat(live)
                if st.st_blocks * 512 < size // 2:
                    # Sparse log: the writer holds a stale fd offset, so the
                    # apparent size is mostly a NUL hole. Truncating reclaims
                    # nothing (real usage is st_blocks) and just re-creates the
                    # hole, so leave it and report instead.
                    with _lock:
                        state["log_cap_note"] = (
                            "日志稀疏未清理：表观 %d MB / 实占 %d MB（写入端 fd 偏移陈旧）"
                            % (size // (1024 * 1024),
                               st.st_blocks * 512 // (1024 * 1024)))
                else:
                    with open(live, "rb") as src, \
                            open(live + ".archived", "wb") as dst:
                        src.seek(max(0, size - 20 * 1024 * 1024))
                        dst.write(src.read())
                    with open(live, "wb"):
                        pass  # copytruncate: engine keeps appending (sparse ok)
                    with _lock:
                        state["log_cap_note"] = ("日志超过 %d MB 已自动清理（尾部归档 %s）"
                                                 % (LOG_CAP_BYTES // (1024 * 1024),
                                                    ENGINE_LOG_ARCH))
        except Exception:
            pass
        time.sleep(300)


@app.get("/api/live")
def api_live():
    with _lock:
        cum = dict(state.get("cumulative", {}))
        daily = dict(state.get("daily") or {})
    today_fresh = max(0.0, daily.get("prompt_tokens", 0.0)
                      - daily.get("cached_tokens", 0.0))
    # Today's cost uses the same price table as the cumulative figure, applied
    # to today's token deltas (daily_stats rebases across engine restarts).
    cum["cost_today"] = round(
        today_fresh / 1e6 * _pricing["input"]
        + daily.get("cached_tokens", 0.0) / 1e6 * _pricing["cached"]
        + daily.get("generation_tokens", 0.0) / 1e6 * _pricing["output"], 2)
    cum["today_tokens"] = {
        "prompt_tokens": daily.get("prompt_tokens", 0),
        "cached_tokens": daily.get("cached_tokens", 0),
        "generation_tokens": daily.get("generation_tokens", 0),
    }
    with _lock:
        return JSONResponse({
            "now": time.strftime("%H:%M:%S"),
            "service": dict(state["service"]),
            "host": dict(state["host"]),
            "gpus": list(state["gpus"]),
            "log": dict(state["log"]),
            "logtail": [str(x)[:MAX_TAIL_CHARS] for x in state.get("logtail", [])],
            "proc": dict(state["proc"]),
            "cumulative": cum,
            "energy": _energy_public(),
            "queue_log": list(state.get("queue_log", [])),
            "requests": list(req_rows)[-30:],
            "logs": _log_stats(),
            "ctl": dict(_ctl_state),
            "auto": {"enabled": _auto["enabled"],
                     "down_ticks": _auto["down_ticks"],
                     "last_result": _auto["last_result"],
                     "tripped": _auto["tripped"],
                     "manual_note": _auto["manual_note"],
                     "starts_last_hour": len([t for t in _auto["starts"]
                                              if time.time() - t < 3600.0]),
                     "max_per_hour": AUTO_MAX_PER_HOUR},
            "log_cap_note": state.get("log_cap_note"),
        })


_dayq_cache = {"path": None, "mtime": 0.0, "data": None}
_dayq_lock = threading.Lock()


@app.get("/api/queue_day")
def api_queue_day():
    """Today's queue depth, from the 60 s snapshots (one point per minute).

    The in-memory history deques only span two hours, so the day view reads the
    persisted snapshots_<date>.jsonl instead; the parsed result is cached until
    the file changes.
    """
    today = time.strftime("%Y-%m-%d")
    path = os.path.join(PERSIST_DIR, "snapshots_%s.jsonl" % today)
    try:
        st = os.stat(path)
    except OSError:
        return {"date": today, "points": []}
    with _dayq_lock:
        if _dayq_cache["path"] == path and _dayq_cache["mtime"] == st.st_mtime:
            return _dayq_cache["data"]
    points = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                ts = row.get("ts")
                if not ts:
                    continue
                pending = row.get("pending_tokens")
                points.append({
                    "t": int(ts),
                    "q": round(float(row.get("queued") or 0.0), 2),
                    "r": round(float(row.get("running") or 0.0), 2),
                    "p": None if pending is None else round(float(pending), 0),
                })
    except OSError:
        return {"date": today, "points": []}
    data = {"date": today, "points": points}
    with _dayq_lock:
        _dayq_cache.update(path=path, mtime=st.st_mtime, data=data)
    return data


@app.get("/api/history")
def api_history():
    with _hist_lock:
        return JSONResponse({k: list(v) for k, v in hist.items()})


@app.get("/health")
def health():
    return {"ok": True}


PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<!-- Dual-core Chinese browsers (360/QQ/Sogou) may open this intranet IP in
     their IE core, where the ES2017 dashboard script cannot even parse and
     every card silently stays "-". Ask for the fast core / latest IE mode. -->
<meta name="renderer" content="webkit">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<title>SGLang 7007 负载监控</title>
<style>
*{box-sizing:border-box}
body{background:#f5f6f8;color:#1a1f27;font-family:-apple-system,'Segoe UI','Microsoft YaHei',system-ui,sans-serif;margin:0;padding:18px 22px}
h1{font-size:17px;font-weight:600;margin:0 0 14px;display:flex;align-items:center;gap:10px}
h1 .sub{font-size:12px;color:#7a828e;font-weight:400}
.badge{padding:2px 12px;border-radius:12px;font-size:12px;font-weight:600}
.ok{background:#e2f5e8;color:#1d9d55}.bad{background:#fdeaea;color:#d64545}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:14px}
.grid6{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:14px}
.card{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:14px 18px;min-height:86px}
.card .l{font-size:12px;color:#7a828e;margin-bottom:6px}
.card .v{font-size:30px;font-weight:700;letter-spacing:-0.5px;line-height:1.1}
.card .s{font-size:11.5px;color:#7a828e;margin-top:5px}
.chartcard{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:12px 16px;margin-bottom:14px}
.chartcard .t{font-size:12px;color:#7a828e;margin-bottom:4px;display:flex;justify-content:space-between}
canvas{width:100%;height:150px;display:block}
table{border-collapse:collapse;width:100%;font-size:12.5px}
td,th{padding:6px 10px;text-align:left;border-bottom:1px solid #eef0f3}
th{color:#7a828e;font-weight:500}.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:8px;background:#eef0f3;border-radius:6px;overflow:hidden;margin-top:8px}
.bar>div{height:100%;border-radius:6px}
.gcard{background:#fff;border:1px solid #e6e9ee;border-radius:12px;padding:12px 16px}
.gcard .n{font-size:12.5px;font-weight:600;margin-bottom:2px;display:flex;justify-content:space-between}
.gcard .m{font-size:11px;color:#7a828e;margin-top:6px}
h2{font-size:13px;color:#7a828e;font-weight:500;margin:20px 0 8px}
pre{background:#1a1f27;color:#9aa4b2;border-radius:10px;padding:10px 14px;font-size:11px;overflow-x:auto;margin:0;min-height:60px}
#updated{font-size:11px;color:#7a828e;margin-left:auto}
.btn{background:#fff;border:1px solid #d7dce3;border-radius:8px;padding:5px 14px;font-size:12.5px;color:#1a1f27;cursor:pointer}
.btn:hover{background:#f2f4f7}
.btn:disabled{opacity:.45;cursor:not-allowed}
#pmodal textarea{outline:none}
/* hardware cards: label/value mini grids keep the four cards readable */
.card.hw .v{font-size:24px}
.mini3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:6px 0 4px}
.mini2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:6px 0 4px}
.mini3 .k,.mini2 .k{font-size:11px;color:#9aa4b2;margin-bottom:2px;white-space:nowrap}
.mini3 .vv,.mini2 .vv{font-size:15px;font-weight:700;letter-spacing:-.3px;white-space:nowrap}
.card.hw .bar{height:6px;background:#eef1f5;border-radius:4px;overflow:hidden;margin-top:6px}
.card.hw .bar div{height:100%;border-radius:4px;transition:width .3s}
/* metric cards: one compact line per card (title + value + right-side detail) */
.card.metric{padding:9px 14px;min-height:0}
.card.metric .l{font-size:11.5px;margin-bottom:2px}
.card.metric .v{font-size:24px;letter-spacing:-.4px}
.card.metric .v .unit{font-size:12px;font-weight:400;color:#7a828e;letter-spacing:0}
.vrow{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
.vrow .m{font-size:10.5px;color:#7a828e;line-height:1.3;text-align:right;flex:1 1 auto;min-width:0}
.card.metric .bar{height:5px;background:#eef1f5;border-radius:4px;overflow:hidden;margin-top:5px}
.card.metric .bar div{height:100%;border-radius:4px;transition:width .3s}
.card .v.green{color:#1d9d55}
.card .v.purple{color:#8b5cf6}
.card .v.blue{color:#2563eb}
.card .v.amber{color:#d97706}
.card .v.teal{color:#0d9488}
.card .v.slate{color:#334155}
.card .v.bad{color:#d64545}
</style></head><body>
<div id="oldkernel" style="display:none;background:#fff8e1;border:1px solid #ffe08a;color:#7a5b00;border-radius:10px;padding:12px 16px;margin:0 0 14px;font-size:13px;line-height:1.6">
当前浏览器用的是 IE 兼容内核，本页面的实时指标与图表需要极速内核：请在地址栏切换为「极速模式」，或改用 Chrome / Edge 打开。
</div>
<script>
/* ES5 guard: the main dashboard script is ES2017, so an old kernel fails to
   parse it silently and every card stays at its placeholder. */
window.__m7100ok = false;
setTimeout(function () {
  if (!window.__m7100ok) {
    var el = document.getElementById('oldkernel');
    if (el) { el.style.display = 'block'; }
  }
}, 3000);
</script>
<h1><span id="htitle">SGLang 负载监控</span> <span id="hbadge" class="badge">…</span>
 <span class="sub" id="hsub">引擎 -</span>
 <span id="updated"></span></h1>

<div style="display:flex;gap:8px;align-items:center;margin:0 0 14px;flex-wrap:wrap">
 <button class="btn" id="btnstart" onclick="ctl('start')">启动</button>
 <button class="btn" id="btnstop" onclick="ctl('stop')">停止</button>
 <button class="btn" id="btnrestart" onclick="ctl('restart')">重启</button>
 <button class="btn" onclick="openParams()">参数</button>
 <span id="ctlstatus" style="font-size:12px;color:#7a828e"></span>
 <label style="font-size:12px;color:#4b5563;display:flex;align-items:center;gap:5px;cursor:pointer" title="引擎进程消失时自动重新拉起（10 秒巡检，连续 3 次确认；每小时最多 3 次，超限自动停手）">
  <input type="checkbox" id="autochk" onchange="toggleAuto(this.checked)">自动拉起
 </label>
 <span id="autostatus" style="font-size:11.5px;color:#7a828e"></span>
 <span style="margin-left:auto;font-size:11.5px;color:#7a828e">
  日志 <b id="logsize">-</b> <span id="logcapinfo"></span>
  <button class="btn" style="padding:3px 10px;font-size:11px;margin-left:6px" onclick="cleanLogs()">清理日志</button></span>
</div>

<div id="pmodal" style="display:none;position:fixed;inset:0;background:rgba(15,20,28,.4);z-index:9">
 <div style="background:#fff;max-width:940px;margin:5vh auto;border-radius:12px;padding:16px 20px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
   <span><b style="font-size:14px">启动参数编辑</b>
    <select id="paramfile" onchange="openParams()" style="margin-left:10px;padding:4px 8px;border:1px solid #d7dce3;border-radius:8px;font-size:12.5px">
     <option value="launcher">启动脚本（--launcher / SGLANG_LAUNCHER 指定的脚本）</option>
    </select></span>
   <span><button class="btn" onclick="saveParams()">保存</button>
   <button class="btn" onclick="closeParams()">关闭</button></span></div>
  <textarea id="paramtext" spellcheck="false" style="width:100%;height:62vh;font:12px/1.5 ui-monospace,Consolas,monospace;border:1px solid #e6e9ee;border-radius:8px;padding:10px"></textarea>
  <div id="parammsg" style="font-size:12px;color:#7a828e;margin-top:6px"></div>
 </div>
</div>

<div class="chartcard">
 <div class="t"><span>输出吞吐 · Output tok/s（近 2 小时，3s 粒度）</span><span id="curnow">-</span></div>
 <canvas id="bigchart" height="150"></canvas>
</div>

<div class="grid">
 <div class="card metric"><div class="l">实时输出 · Live Output</div>
  <div class="vrow"><div class="v green" id="gtps">- <span class="unit">tok/s</span></div>
   <div class="m" id="gtpssub">-</div></div></div>
 <div class="card metric"><div class="l">起草接受率 · Draft Acceptance</div>
  <div class="vrow"><div class="v purple" id="acc">-</div>
   <div class="m" id="accsub">-</div></div>
  <div class="bar"><div id="accbar" style="width:0%;background:#8b5cf6"></div></div></div>
 <div class="card metric"><div class="l">并发 · Concurrency</div>
  <div class="vrow"><div class="v blue" id="conc">-</div>
   <div class="m" id="concsub">-</div></div></div>
 <div class="card metric"><div class="l">KV 池占用 · KV Pool</div>
  <div class="vrow"><div class="v amber" id="kv">-</div>
   <div class="m" id="kvsub">-</div></div>
  <div class="bar"><div id="kvbar" style="width:0%;background:#f59e0b"></div></div></div>
</div>

<div class="grid">
 <div class="card metric"><div class="l">首 Token 延迟 · TTFT</div>
  <div class="vrow"><div class="v blue" id="ttftp">-</div>
   <div class="m" id="ttftp_m">全期累计</div></div></div>
 <div class="card metric"><div class="l">端到端 · E2E</div>
  <div class="vrow"><div class="v green" id="e2ep">-</div>
   <div class="m" id="e2ep_m">全期累计</div></div></div>
 <div class="card metric"><div class="l">L2 主机层 · Host Cache</div>
  <div class="vrow"><div class="v teal" id="l2used">-</div>
   <div class="m" id="l2sub">-</div></div>
  <div class="bar"><div id="l2bar" style="width:0%;background:#0d9488"></div></div></div>
 <div class="card metric"><div class="l">投机解码 · Spec Decode</div>
  <div class="vrow"><div class="v purple" id="specacc">-</div>
   <div class="m" id="specsub">-</div></div></div>
</div>

<div class="grid">
 <div class="card metric"><div class="l">显存构成 · VRAM Split</div>
  <div class="vrow"><div class="v slate" id="vsplit">-</div>
   <div class="m" id="vsplitsub">-</div></div></div>
 <div class="card metric"><div class="l">HTTP 错误 · Errors（本进程）</div>
  <div class="vrow"><div class="v green" id="errv">-</div>
   <div class="m" id="errsub">-</div></div></div>
 <div class="card metric"><div class="l">分阶段延迟 P90 · Stage Latency</div>
  <div class="vrow"><div class="v amber" id="stagev">-</div>
   <div class="m" id="stagesub">-</div></div></div>
 <div class="card metric"><div class="l">调度器空闲 · Scheduler Idle</div>
  <div class="vrow"><div class="v green" id="idlev">-</div>
   <div class="m" id="idlesub">-</div></div>
  <div class="bar"><div id="idlebar" style="width:0%;background:#1d9d55"></div></div></div>
</div>

<h2>硬件监视 · Hardware（每 3 秒自动刷新）</h2>
<div class="grid">
 <div class="card hw">
  <div class="l">CPU · 使用率</div>
  <div class="v" id="hwcpu">-</div>
  <div class="bar"><div id="hwcpubar" style="width:0%;background:#3b82f6"></div></div>
  <div class="s" id="hwcpusub">-</div>
 </div>
 <div class="card hw">
  <div class="l">内存 · Memory</div>
  <div class="v" id="hwmem">-</div>
  <div class="bar"><div id="hwmembar" style="width:0%;background:#1d9d55"></div></div>
  <div class="s" id="hwmemsub">-</div>
 </div>
 <div class="card hw">
  <div class="l">硬盘 · Disk（/data）</div>
  <div class="mini3">
   <div><div class="k">总量</div><div class="vv" id="hwdisk_total">-</div></div>
   <div><div class="k">已用</div><div class="vv" id="hwdisk_used">-</div></div>
   <div><div class="k">可用</div><div class="vv" id="hwdisk_free">-</div></div>
  </div>
  <div class="s" id="hwdisksub">-</div>
 </div>
 <div class="card hw">
  <div class="l">网络 · Network（eno1）</div>
  <div class="mini2">
   <div><div class="k">下行 ↓</div><div class="vv" id="hwnet_rx">-</div></div>
   <div><div class="k">上行 ↑</div><div class="vv" id="hwnet_tx">-</div></div>
  </div>
  <div class="s" id="hwnetsub">-</div>
 </div>
</div>

<h2>GPU · 3s</h2>
<div class="grid6" id="gpurow"></div>

<h2>能耗 · Energy（每 3 秒采样）</h2>
<div class="grid">
 <div class="card"><div class="l">整机功耗 · Total Power</div><div class="v" style="font-size:26px;color:#4f46e5" id="powerw">-</div><div class="s" id="powerwsub">-</div></div>
 <div class="card"><div class="l">累计电量 · Energy</div><div class="v" style="font-size:26px;color:#2563eb" id="ekwh">-</div><div class="s" id="efirst">-</div></div>
 <div class="card"><div class="l">估算电费 · Cost</div><div class="v" style="font-size:26px;color:#d97706" id="ecost">-</div><div class="s">电价 <b id="eprice" style="cursor:pointer;color:#1d9d55" onclick="editEnergyPrice()" title="点击修改电价">-</b> 元/kWh</div></div>
 <div class="card"><div class="l">今日电费 · Today Cost</div><div class="v" style="font-size:26px;color:#059669" id="ecosttoday">-</div><div class="s">今日 <b id="ekwhtoday">-</b> kWh · 电价 <b id="eprice2" style="cursor:pointer;color:#1d9d55" onclick="editEnergyPrice()" title="点击修改电价">-</b> 元/kWh</div></div>
</div>

<h2>累计吞吐 · Cumulative（跨重启累计，60s 记账持久化）</h2>
<div class="chartcard" style="padding:20px 24px">
 <div style="display:flex;gap:48px;align-items:flex-start">
  <div>
   <div class="l" style="margin-bottom:8px">总输入 Tokens · Input（跨重启累计）</div>
   <div style="font-size:36px;font-weight:700;line-height:1.15" id="cinbig">-</div>
   <div class="s" style="margin-top:10px;font-size:12.5px">缓存命中 <b id="ccache">-</b></div>
  </div>
  <div>
   <div class="l" style="margin-bottom:8px">总输出 Tokens · Output（跨重启累计）</div>
   <div style="font-size:36px;font-weight:700;line-height:1.15;color:#1d9d55" id="coutbig">-</div>
   <div class="s" style="margin-top:10px;font-size:12.5px">本次启动 <b id="csess">-</b> · 请求 <b id="creqs">-</b></div>
  </div>
  <div id="costline-wrap" style="margin-left:auto;text-align:right;cursor:pointer" title="点击修改单价">
   <div class="l" style="margin-bottom:8px">价值 · Cost（今日 / 累计，点击改单价）</div>
   <div style="font-size:34px;font-weight:700;line-height:1.15;color:#d97706" id="costbig">-</div>
   <div class="s" style="margin-top:10px;font-size:12.5px" id="costsub2">-</div>
  </div>
 </div>
 <div style="border-top:1px solid #eef0f3;margin-top:18px;padding-top:12px">
  <div style="font-size:12px;color:#7a828e;line-height:1.7" id="costsub">-</div>
  <div style="font-size:12px;color:#7a828e;line-height:1.7" id="ctoksub">-</div>
  <div style="display:flex;gap:28px;margin-top:14px;flex-wrap:wrap">
   <div><div class="l" style="font-size:11.5px;color:#7a828e">今日输入 · Input</div><div style="font-size:16px;font-weight:600" id="todayin">-</div></div>
   <div><div class="l" style="font-size:11.5px;color:#7a828e">今日输出 · Output</div><div style="font-size:16px;font-weight:600;color:#1d9d55" id="todayout">-</div></div>
   <div><div class="l" style="font-size:11.5px;color:#7a828e">今日缓存命中 · Cached</div><div style="font-size:16px;font-weight:600;color:#2563eb" id="todaycache">-</div></div>
   <div><div class="l" style="font-size:11.5px;color:#7a828e">今日请求 · Requests</div><div style="font-size:16px;font-weight:600;color:#d97706" id="todayreq">-</div></div>
  </div>
  <div class="bar" style="margin-top:12px"><div id="hitbar" style="width:0%;background:#1d9d55"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#7a828e;margin-top:8px">
   <span>缓存命中率（跨重启） · Cache Hit Rate</span><b id="chit">-</b></div>
 </div>
</div>

<h2>请求队列 · Request Queue</h2>
<div class="grid6">
 <div class="card"><div class="v" style="font-size:22px" id="qwait">-</div><div class="l">排队中 Queued</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="qrun">-</div><div class="l">运行中 Running</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="qpend">-</div><div class="l">队列等待 tokens</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="qwt">-</div><div class="l">入列等待均值</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="qgram">-</div><div class="l">Grammar 队列</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="qinfl">-</div><div class="l">Prefill 在途</div></div>
</div>
<div class="chartcard">
 <div class="t"><span>队列深度 · Queue Depth（今日，60s 采样）</span>
  <span style="display:flex;gap:14px;align-items:center;font-size:11.5px;color:#7a828e">
   <span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#3b82f6;margin-right:5px"></i>运行 Running</span>
   <span><i style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#8b5cf6;margin-right:5px"></i>排队 Queued</span>
   <span id="qpeak">峰值 -</span></span></div>
 <canvas id="qchart" height="150"></canvas>
 <table style="margin-top:8px"><thead><tr><th>时间</th><th class="num">排队</th><th class="num">运行</th><th class="num">等待 tokens</th></tr></thead>
 <tbody id="qbody"><tr><td colspan="4">-</td></tr></tbody></table>
</div>

<h2>请求明细 · Request Details <span style="font-size:11px;color:#9aa4b2">（完成后落表；运行中/排队见上方队列面板）</span></h2>
<div class="chartcard">
 <table><thead><tr><th>日期</th><th>时间</th><th>客户端</th><th class="num">上下文</th><th class="num">缓存命中率</th><th class="num">新算</th><th class="num">输出</th>
 <th class="num">预填充速度</th><th class="num">输出速度</th><th class="num">排队</th><th class="num">总时长</th></tr></thead>
 <tbody id="reqbody"><tr><td colspan="11">-</td></tr></tbody></table>
</div>

<h2>异常计数（进程启动以来） · Anomaly Counters</h2>
<div class="grid6">
 <div class="card"><div class="v" style="font-size:22px" id="ctriton">-</div><div class="l">Triton 运行中加载</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="car">-</div><div class="l">custom AR 失败</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="ctb">-</div><div class="l">Traceback</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="coom">-</div><div class="l">OOM</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="cevict">-</div><div class="l">驱逐 tokens</div></div>
 <div class="card"><div class="v" style="font-size:22px" id="crestart">-</div><div class="l">进程重启</div></div>
</div>

<h2>日志尾部</h2>
<pre id="tail">-</pre>

<script>
window.__m7100ok = true;  /* parsed means a modern engine */
const fmt=(x,d=1)=>x==null||isNaN(x)?'-':Number(x).toFixed(d);
const fmtRate=b=>b==null||isNaN(b)?'-':(b>=1048576?(b/1048576).toFixed(1)+' MB/s':(b/1024).toFixed(1)+' KB/s');
let __dayqAt=0;
function put(id,v){document.getElementById(id).textContent=v;}
function cn(x){if(x>=1e8)return (x/1e8).toFixed(2)+' 亿';if(x>=1e4)return (x/1e4).toFixed(1)+' 万';return Math.round(x).toLocaleString();}
function bignum(el,v){document.getElementById(el).textContent=cn(v);}
async function ctl(action){
 if(action!=='start'&&!confirm((action==='stop'?'停止':'重启')+'引擎？'+(action==='restart'?'（需重新加载约 15 分钟）':'')))return;
 try{await fetch('/api/ctl/'+action,{method:'POST'});put('ctlstatus','已请求: '+action+' …');}catch(e){put('ctlstatus','请求失败');}
}
async function toggleAuto(on){
 try{
  await fetch('/api/auto/'+(on?'on':'off'),{method:'POST'});
  put('autostatus',on?'已开启':'已关闭');
 }catch(e){put('autostatus','设置失败');}
}
async function openParams(){
 try{
  const f=document.getElementById('paramfile').value;
  const d=await(await fetch('/api/params?file='+encodeURIComponent(f))).json();
  if(!d.ok){alert(d.err||'读取失败');return;}
  document.getElementById('paramtext').value=d.content;
  document.getElementById('parammsg').textContent='路径: '+d.path+' · 保存前自动语法校验+备份，重启后生效';
  document.getElementById('pmodal').style.display='block';
 }catch(e){alert('读取失败');}
}
function closeParams(){document.getElementById('pmodal').style.display='none';}
async function saveParams(){
 try{
  const content=document.getElementById('paramtext').value;
  const file=document.getElementById('paramfile').value;
  const label='启动脚本';
  if(!confirm('保存将覆盖【'+label+'】\n（自动备份为 .bak-monitor，重启后生效）\n\n确认保存？'))return;
  const r=await fetch('/api/params',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file:file,content:content})});
  const d=await r.json();
  document.getElementById('parammsg').textContent=d.ok?('已保存 ✓ '+(d.note||'')):('保存失败: '+d.err);
 }catch(e){document.getElementById('parammsg').textContent='保存失败（网络）';}
}
async function cleanLogs(){
 if(!confirm('清理过期日志？（保留当前运行日志，删除历史 sglang 日志与归档）'))return;
 try{
  const d=await(await fetch('/api/logs/clean',{method:'POST'})).json();
  alert('已清理，释放 '+(d.freed/1048576).toFixed(1)+' MB');
 }catch(e){alert('清理失败');}
}
function fmtMB(x){return (x/1048576).toFixed(1)+' MB';}
async function editPricing(){
 const cur=(window.__cum&&window.__cum.pricing)||{input:0.8,output:2.7,cached:0.086};
 const i=prompt('输入单价（元/百万 tokens）：',cur.input);if(i===null)return;
 const o=prompt('输出单价（元/百万 tokens）：',cur.output);if(o===null)return;
 const c=prompt('缓存命中单价（元/百万 tokens）：',cur.cached);if(c===null)return;
 try{
  const r=await fetch('/api/pricing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:i,output:o,cached:c})});
  const d2=await r.json();
  alert(d2.ok?'单价已保存':'保存失败: '+d2.err);
 }catch(e){alert('保存失败（网络）');}
}
async function editEnergyPrice(){
  const cur=(window.__energy&&window.__energy.price!=null)?window.__energy.price:1;
 const v=prompt('电价（元/kWh）：',cur);if(v===null)return;
 try{
  const r=await fetch('/api/energy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({price:v})});
  const d2=await r.json();
  if(!d2.ok)alert('保存失败: '+(d2.err||''));
 }catch(e){alert('保存失败（网络）');}
}
function drawbig(series){ const c=document.getElementById('bigchart');
 const W=c.clientWidth,H=150;c.width=W;c.height=H;
 const g=c.getContext('2d');
 g.clearRect(0,0,W,H);
 g.strokeStyle='#eef0f3';g.lineWidth=1;
 for(let i=1;i<4;i++){g.beginPath();g.moveTo(0,H*i/4);g.lineTo(W,H*i/4);g.stroke();}
 if(!series||series.length<2)return;
 const mx=Math.max(...series,1);
 const px=i=>i/(series.length-1)*(W-70)+8;
 const py=v=>H-18-(v/mx)*(H-34);
 const grad=g.createLinearGradient(0,0,0,H);
 grad.addColorStop(0,'rgba(59,130,246,0.22)');grad.addColorStop(1,'rgba(59,130,246,0)');
 g.beginPath();g.moveTo(px(0),py(series[0]));
 series.forEach((v,i)=>g.lineTo(px(i),py(v)));
 g.lineTo(px(series.length-1),H-14);g.lineTo(px(0),H-14);g.closePath();
 g.fillStyle=grad;g.fill();
 g.beginPath();g.strokeStyle='#3b82f6';g.lineWidth=1.8;
 series.forEach((v,i)=>{i?g.lineTo(px(i),py(v)):g.moveTo(px(i),py(v));});
 g.stroke();
 const li=series.length-1;
 g.fillStyle='#3b82f6';g.beginPath();g.arc(px(li),py(series[li]),3.5,0,7);g.fill();
 g.fillStyle='#1a1f27';g.font='12px monospace';
 g.fillText('max '+mx.toFixed(1),8,16);
 g.fillText('now '+series[li].toFixed(1),px(li)-64,py(series[li])-8);
}
function niceMax(v){
 const base=Math.pow(10,Math.floor(Math.log10(Math.max(v,1))));
 for(const s of [1,2,5,10]){ if(v<=s*base) return s*base; }
 return 10*base;
}
function drawqueue(points){
 const c=document.getElementById('qchart');
 const W=c.clientWidth||900, H=150;
 const dpr=window.devicePixelRatio||1;
 c.width=W*dpr; c.height=H*dpr;
 c.style.width=W+'px'; c.style.height=H+'px';
 const g=c.getContext('2d');
 g.setTransform(dpr,0,0,dpr,0,0);
 g.clearRect(0,0,W,H);
 const L=34,R=14,T=14,B=22, pw=W-L-R, ph=H-T-B;
 const d0=new Date(); d0.setHours(0,0,0,0);
 const t0=d0.getTime()/1000, now=Date.now()/1000;
 const px=t=>L+Math.min(1,Math.max(0,(t-t0)/86400))*pw;
 const ymax=niceMax((points||[]).reduce((a,p)=>Math.max(a,p.q||0,p.r||0),0));
 const py=v=>T+ph-(Math.min(v,ymax)/ymax)*ph;
 // future part of the day: subtle shade + dashed "now" line
 const xn=px(now);
 if(xn<L+pw){g.fillStyle='#f7f8fb';g.fillRect(xn,T,L+pw-xn,ph);}
 // y grid and labels
 g.font='10px ui-monospace,Consolas,monospace';
 g.textAlign='right'; g.textBaseline='middle';
 for(let i=0;i<=3;i++){
  const v=ymax*i/3, y=py(v);
  g.strokeStyle=i?'#eef1f5':'#e2e6ec'; g.lineWidth=1;
  g.beginPath(); g.moveTo(L,y); g.lineTo(L+pw,y); g.stroke();
  g.fillStyle='#9aa4b2'; g.fillText(Math.round(v),L-6,y);
 }
 // x ticks every 6h
 g.textAlign='center'; g.textBaseline='top';
 for(let hh=0;hh<=24;hh+=6){
  const x=L+(hh/24)*pw;
  g.strokeStyle='#eef1f5'; g.beginPath(); g.moveTo(x,T); g.lineTo(x,T+ph); g.stroke();
  g.fillStyle='#9aa4b2'; g.fillText((hh<10?'0':'')+hh+':00',x,H-16);
 }
 if(!points||points.length<2){
  g.fillStyle='#9aa4b2'; g.font='12px system-ui,sans-serif'; g.textAlign='center';
  g.fillText('今日暂无队列数据',L+pw/2,T+ph/2);
  put('qpeak','峰值 -'); return;
 }
 // area + line per series (running behind queued)
 const drawSeries=(key,color)=>{
  g.beginPath();
  points.forEach((p,i)=>{const x=px(p.t),y=py(p[key]||0); i?g.lineTo(x,y):g.moveTo(x,y);});
  const lg=g.createLinearGradient(0,T,0,T+ph);
  lg.addColorStop(0,color+'50'); lg.addColorStop(1,color+'06');
  g.lineTo(px(points[points.length-1].t),T+ph);
  g.lineTo(px(points[0].t),T+ph);
  g.closePath(); g.fillStyle=lg; g.fill();
  g.beginPath();
  points.forEach((p,i)=>{const x=px(p.t),y=py(p[key]||0); i?g.lineTo(x,y):g.moveTo(x,y);});
  g.strokeStyle=color; g.lineWidth=1.8; g.lineJoin='round'; g.lineCap='round'; g.stroke();
 };
 drawSeries('r','#3b82f6');
 drawSeries('q','#8b5cf6');
 // now marker
 g.setLineDash([4,3]); g.strokeStyle='#cbd5e1'; g.lineWidth=1;
 g.beginPath(); g.moveTo(xn,T); g.lineTo(xn,T+ph); g.stroke(); g.setLineDash([]);
 // last-point dots
 const last=points[points.length-1];
 [['r','#3b82f6'],['q','#8b5cf6']].forEach(([k,col])=>{
  const x=px(last.t), y=py(last[k]||0);
  g.beginPath(); g.arc(x,y,3.6,0,7); g.fillStyle='#fff'; g.fill();
  g.strokeStyle=col; g.lineWidth=2; g.stroke();
 });
 const peakR=points.reduce((a,p)=>Math.max(a,p.r||0),0);
 const peakQ=points.reduce((a,p)=>Math.max(a,p.q||0),0);
 put('qpeak','峰值 运行 '+Math.round(peakR)+' · 排队 '+Math.round(peakQ));
}
async function tick(){
 try{
  const d=await(await fetch('/api/live')).json();
  const s=d.service||{},h=d.host||{},p=d.proc||{},L=d.log||{};
  if(d.engine){const ht=document.getElementById('htitle');
   if(ht)ht.textContent='SGLang '+d.engine.port+' 负载监控';
   const hs=document.getElementById('hsub');
   if(hs)hs.textContent='引擎 http://'+d.engine.host+':'+d.engine.port;}
  const b=document.getElementById('hbadge');
  const hasPid=p.pid!=null&&p.pid!='null';
  if(s.ok&&hasPid){b.textContent='运行中 Running';b.className='badge ok';}
  else{b.textContent='不可达 / 已停止';b.className='badge bad';}
  const liveTps=(L.log_gen_tps!=null&&L.log_gen_tps>1)?L.log_gen_tps:s.gen_throughput;
  put('gtps',fmt(liveTps,2));
  put('gtpssub','服务端均值 '+fmt(s.gen_throughput,1)+' tok/s');
  if(L.accept_rate!=null){put('acc',(100*L.accept_rate).toFixed(1)+'%');
   document.getElementById('accbar').style.width=Math.min(100,100*L.accept_rate)+'%';
   put('accsub','Mean Len '+fmt(L.accept_len,2));}
  else{put('acc','-');document.getElementById('accbar').style.width='0%';put('accsub','无解码样本');}
  put('conc',(s.num_running_reqs??'-')+' / '+(s.num_queue_reqs??'-'));
  put('concsub','运行 / 排队 · 活跃 KV '+(s.kv_used_tokens!=null?Math.round(s.kv_used_tokens).toLocaleString():'-'));
  // 硬件监视 · Hardware
  put('hwcpu',h.cpu_pct!=null?fmt(h.cpu_pct)+'%':'-');
  const cbar=document.getElementById('hwcpubar');
  if(cbar)cbar.style.width=Math.min(100,Math.max(0,h.cpu_pct||0))+'%';
  put('hwcpusub',(h.cpu_mhz!=null?fmt(h.cpu_mhz,0)+' MHz':'')+
   (h.cpu_threads?' · '+h.cpu_threads+' 线程':'')+' · load '+fmt(h.load1,2)+' / '+fmt(h.load5,2));
  put('hwmem',fmt(h.mem_used_gb)+' / '+fmt(h.mem_total_gb)+' GB');
  const mbar=document.getElementById('hwmembar');
  if(mbar&&h.mem_total_gb)mbar.style.width=Math.min(100,100*h.mem_used_gb/h.mem_total_gb)+'%';
  put('hwmemsub','可用 '+fmt(h.mem_avail_gb)+' GB · swap '+fmt(h.swap_used_gb,1)+' GB');
  const hd=h.disk||{};
  if(hd.total_gb){
   put('hwdisk_total',fmt(hd.total_gb/1024,2)+' TB');
   put('hwdisk_used',fmt(hd.used_gb/1024,2)+' TB');
   put('hwdisk_free',fmt(hd.free_gb/1024,2)+' TB');
   put('hwdisksub','读 '+fmtRate(hd.read_bps)+' · 写 '+fmtRate(hd.write_bps)+
    ' · 累计读 '+fmt(hd.read_total_gb,2)+' GB / 写 '+fmt(hd.write_total_gb,2)+' GB');}
  else{put('hwdisk_total','-');put('hwdisk_used','-');put('hwdisk_free','-');put('hwdisksub','-');}
  const nw=h.net||{};
  put('hwnet_rx',fmtRate(nw.rx_bps));
  put('hwnet_tx',fmtRate(nw.tx_bps));
  put('hwnetsub',(nw.up?'✅ 正常':'⚠ 断开')+' · '+(nw.iface_total||0)+' 个网卡 up · 累计 '+
   fmt(nw.rx_total_gb,2)+' / '+fmt(nw.tx_total_gb,2)+' GB');
  // 能耗 · Energy
  const en=d.energy||{};
  window.__energy=en;
  put('powerw',en.power_w!=null?fmt(en.power_w,1)+' W':'-');
  put('powerwsub','GPU '+fmt(en.gpu_w,1)+' W · CPU '+
   (en.cpu_w==null?'不可读':fmt(en.cpu_w,1)+' W')+' · 补偿 '+fmt(en.compensate_w,0)+' W');
  put('ekwh',en.total_kwh!=null?fmt(en.total_kwh,2)+' kWh':'-');
  put('ecost',en.cost!=null?'¥'+fmt(en.cost,2):'-');
  put('eprice',en.price!=null?fmt(en.price,2):'-');
  put('ecosttoday',en.today_cost!=null?'¥'+fmt(en.today_cost,2):'-');
  put('ekwhtoday',en.today_kwh!=null?fmt(en.today_kwh,2):'-');
  put('eprice2',en.price!=null?fmt(en.price,2):'-');
  put('efirst','自 '+(en.first_seen||'-')+' 起累计 · 今日 '+fmt(en.today_kwh,2)+' kWh / ¥'+fmt(en.today_cost,2));
  const used=s.kv_used_tokens,tot=s.max_total_num_tokens;
  if(used!=null&&tot){const kp=100*used/tot;
   put('kv',kp.toFixed(1)+'%');
   document.getElementById('kvbar').style.width=Math.min(100,kp)+'%';
   put('kvsub',Math.round(used).toLocaleString()+' / '+Math.round(tot).toLocaleString()+
    ' tokens'+(used>0?'':' · 空闲占用为 0'));}
  else{put('kv','-');document.getElementById('kvbar').style.width='0%';put('kvsub','-');}
  const cum=d.cumulative||{};
  window.__cum=cum;
  document.getElementById('costline-wrap').onclick=editPricing;
  bignum('cinbig',cum.prompt_tokens);
  bignum('coutbig',cum.generation_tokens);
  put('ccache',cn(cum.cached_tokens));
  put('csess',cn(s.generation_tokens));
  put('creqs',cn(cum.requests_total));
  const pr=cum.pricing||{input:0.8,output:2.7,cached:0.086};
  put('costbig','¥'+fmt(cum.cost_today,2)+' / ¥'+fmt(cum.cost_total,2));
  put('costsub2','今日 ¥'+fmt(cum.cost_today,2)+' · 累计 输入 ¥'+fmt(cum.cost_input,2)+
   ' · 缓存 ¥'+fmt(cum.cost_cached,2)+' · 输出 ¥'+fmt(cum.cost_output,2));
  put('costsub','单价 入'+pr.input+' / 出'+pr.output+' / 缓存'+pr.cached+
   ' 元每百万（示例价 · 界面可改）');
  put('ctoksub','自 '+(cum.first_seen||'-')+' 开始记账 · 驱逐 '+cn(cum.evicted_tokens)+' tokens');
  const td=cum;
  put('todayin',td.today_prompt_tokens!=null?cn(td.today_prompt_tokens):'-');
  put('todayout',td.today_generation_tokens!=null?cn(td.today_generation_tokens):'-');
  put('todaycache',td.today_cached_tokens!=null?cn(td.today_cached_tokens):'-');
  put('todayreq',td.today_requests!=null?cn(td.today_requests):'-');
  const chit=((cum.prompt_tokens||0)+(cum.cached_tokens||0)>0)?(100*cum.cached_tokens/(cum.prompt_tokens+cum.cached_tokens)):0;
  put('chit',chit.toFixed(1)+'%');
  document.getElementById('hitbar').style.width=Math.min(chit,100)+'%';
  put('ctriton',L.triton_loads??'-');put('car',L.ar_failed??'-');
  put('ctb',L.traceback??'-');put('coom',L.oom??'-');
  put('cevict',s.evicted_tokens!=null?cn(s.evicted_tokens):'-');
  put('crestart',p.restarts??'-');
  const gr=document.getElementById('gpurow');gr.innerHTML='';
  (d.gpus||[]).forEach(g=>{
   const pct=(100*g.mem_used/g.mem_total).toFixed(0);
   const col='#3b82f6';
   const div=document.createElement('div');div.className='gcard';
   div.innerHTML='<div class="n"><span>GPU '+g.index+' · '+g.name.replace('NVIDIA ','').replace('Tesla ','')+
    '</span><span style="color:#7a828e;font-weight:400">'+g.temp_c+'°C</span></div>'+
    '<div style="font-size:10.5px;color:#9aa4b2;margin-top:6px">算力负载 '+g.util+'%</div>'+
    '<div class="bar"><div style="width:'+g.util+'%;background:#f59e0b"></div></div>'+
    '<div style="font-size:10.5px;color:#9aa4b2;margin-top:6px">显存占用 '+pct+'%</div>'+
    '<div class="bar"><div style="width:'+pct+'%;background:'+col+'"></div></div>'+
    '<div class="m">'+(g.mem_used/1024).toFixed(1)+' / '+(g.mem_total/1024).toFixed(0)+' GiB · '+
    g.sm_clock+' MHz · '+g.power_w+' W</div>';
   gr.appendChild(div);});
  document.getElementById('tail').textContent=(d.logtail||[]).join('\n')||'-';
  document.getElementById('curnow').textContent='当前 '+fmt(liveTps,2)+' tok/s · 更新于 '+d.now;
  put('qwait',s.num_queue_reqs??'-');
  put('qrun',s.num_running_reqs??'-');
  put('qpend',s.pending_tokens!=null?cn(s.pending_tokens):'-');
  put('qwt',s.queue_wait_mean_s!=null?fmt(s.queue_wait_mean_s,0)+' ms':'-');
  put('qgram',s.num_grammar_queue_reqs??'-');
  put('qinfl',s.num_prefill_inflight_queue_reqs??'-');
  const qt=document.getElementById('qbody');
  // Single live row, rewritten on every tick.
  qt.innerHTML='<tr><td>'+d.now+'</td>'+
   '<td class="num">'+Math.round(s.num_queue_reqs??0)+'</td>'+
   '<td class="num">'+Math.round(s.num_running_reqs??0)+'</td>'+
   '<td class="num">'+(s.pending_tokens!=null?Math.round(s.pending_tokens).toLocaleString():'-')+'</td></tr>';
  const rq=(d.requests||[]).slice().reverse();
  const rb=document.getElementById('reqbody');
  if(rq.length){rb.innerHTML=rq.map(x=>
   '<tr><td>'+((x.ts_full||'').slice(0,10)||'-')+'</td><td>'+x.t+'</td><td>'+x.ip+'</td>'+
   '<td class="num">'+Math.round(x.in).toLocaleString()+'</td>'+
   '<td class="num">'+(x.in>0?(100*x.cached/x.in).toFixed(1)+'%':'-')+'</td>'+
   '<td class="num">'+Math.round(x.in-x.cached).toLocaleString()+'</td>'+
   '<td class="num">'+Math.round(x.out)+'</td>'+
   '<td class="num">'+(x.pre_speed!=null?fmt(x.pre_speed,0)+' tok/s':'-')+'</td>'+
   '<td class="num">'+(x.out_speed!=null?fmt(x.out_speed,1)+' tok/s':'-')+'</td>'+
   '<td class="num">'+fmt(x.queue_ms,0)+' ms</td>'+
   '<td class="num">'+fmt(x.fwd_ms,0)+' ms</td></tr>').join('');}
  else{rb.innerHTML='<tr><td colspan="11">暂无已完成请求记录（明细在请求完成后记录）</td></tr>';}
  const lg=d.logs||{};
  if(lg.cap){put('logsize',fmtMB(lg.current||0)+' / '+fmtMB(lg.cap));
   put('logcapinfo',(lg.total?' · 全部日志 '+fmtMB(lg.total):''));}
  if(d.log_cap_note)put('logcapinfo',d.log_cap_note);
  const c=d.ctl||{};
  if(c.last_action){put('ctlstatus',c.last_action+' → '+(c.last_result||''));}
  const au=d.auto||{};
  const chk=document.getElementById('autochk');
  if(chk&&chk.checked!==!!au.enabled)chk.checked=!!au.enabled;
  if(au.tripped){put('autostatus','⚠ '+au.tripped);}
  else if(au.enabled){put('autostatus','开（近1小时 '+fmt(au.starts_last_hour,0)+'/'+fmt(au.max_per_hour,0)+' 次）');}
  else{put('autostatus',au.manual_note?('关 · '+au.manual_note):'关');}
  // latency: mean on the card, percentiles as one compact right-hand line
  put('ttftp',s.ttft_mean_ms!=null?fmt(s.ttft_mean_ms,0)+' ms':'-');
  put('ttftp_m',s.ttft_p50_ms!=null?
   ('P50 '+fmt(s.ttft_p50_ms,0)+' · P90 '+(s.ttft_p90_ms!=null?fmt(s.ttft_p90_ms,0):'-')+
    ' · P99 '+(s.ttft_p99_ms!=null?fmt(s.ttft_p99_ms,0):'-')+' ms'):'全期累计 · ms');
  put('e2ep',s.e2e_p50_s!=null?fmt(s.e2e_p50_s,1)+' s':'-');
  put('e2ep_m',s.e2e_p50_s!=null?
   ('P50 '+fmt(s.e2e_p50_s,1)+' · P90 '+(s.e2e_p90_s!=null?fmt(s.e2e_p90_s,1):'-')+' s'):'全期累计 · s');
  // L2 host cache
  const l2u=s.hicache_host_used_tokens,l2t=s.hicache_host_total_tokens;
  if(l2u!=null&&l2t){
   const l2p=100*l2u/l2t;
   put('l2used',l2p.toFixed(1)+'%');
   document.getElementById('l2bar').style.width=Math.min(100,l2p)+'%';
   const drop=s.hicache_dropped_tokens_total;
   put('l2sub',cn(l2u).replace(/ /g,'')+' / '+cn(l2t).replace(/ /g,'')+
    ' · 备份 '+(s.hicache_backup_tokens_total!=null?cn(s.hicache_backup_tokens_total).replace(/ /g,''):'-')+
    (drop>0?' · 丢弃 '+cn(drop).replace(/ /g,''):''));
  }else{
   put('l2used','-');document.getElementById('l2bar').style.width='0%';
   put('l2sub','hicache 未启用');
  }
  // spec decode
  put('specacc',s.spec_accept_length!=null?fmt(s.spec_accept_length,2):'-');
  put('specsub','接受率 '+(s.spec_accept_rate!=null?(100*s.spec_accept_rate).toFixed(1)+'%':'-'));
  // VRAM split
  const w=s.weight_memory_usage_gb,k=s.kv_cache_memory_usage_gb,gm=s.graph_memory_usage_gb;
  if(w!=null||k!=null){
   put('vsplit',fmt(w,1)+' + '+fmt(k,1)+' GB');
   put('vsplitsub','图 '+fmt(gm,1)+' GB · 启动可用 '+fmt(s.startup_available_gpu_memory_gb,2)+' GB');
  }else{put('vsplit','-');put('vsplitsub','-');}
  // errors
  const eN=s.http_err_count,rN=s.http_resp_total;
  const errEl=document.getElementById('errv');
  if(eN!=null&&rN!=null){
   put('errv',Math.round(eN));
   errEl.className='v '+(eN>0?'bad':'green');
   put('errsub','总响应 '+Math.round(rN).toLocaleString()+
    ' · 错误率 '+(rN>0?(100*eN/rN).toFixed(2):'0.00')+'%');
  }else{
   put('errv','-');errEl.className='v green';put('errsub','-');
  }
  // stage latency
  const st=s.stage_p90_ms||{};
  if(Object.keys(st).length){
   const sv=st.prefill_forward!=null?st.prefill_forward:(st.request_process!=null?st.request_process:null);
   put('stagev',sv!=null?fmt(sv,0)+' ms':'—');
   put('stagesub','请求 '+(st.request_process!=null?fmt(st.request_process,0)+' ms':'-')+
    ' · 预填 '+(st.prefill_forward!=null?fmt(st.prefill_forward,0)+' ms':'-'));
  }else{put('stagev','-');put('stagesub','-');}
  // scheduler idle
  const idleS=s.sched_idle_s,upS=(d.proc||{}).uptime_s;
  if(idleS!=null&&upS){
   const ip=100*idleS/upS;
   put('idlev',ip.toFixed(1)+'%');
   document.getElementById('idlebar').style.width=Math.min(100,ip)+'%';
   put('idlesub','空闲 '+fmt(idleS/3600,1)+' h / 运行 '+fmt(upS/3600,1)+' h');
  }else{put('idlev','-');document.getElementById('idlebar').style.width='0%';put('idlesub','-');}
   fetch('/api/history').then(r=>r.json()).then(dh=>{
    drawbig(dh.gen_tps||[]);}).catch(()=>{});
   // Day-scale queue chart: 60 s snapshots, refreshed every 30 s.
   if(Date.now()-__dayqAt>30000){__dayqAt=Date.now();
    fetch('/api/queue_day').then(r=>r.json()).then(dq=>{
     drawqueue(dq.points||[]);}).catch(()=>{});}
 }catch(e){console.log(e);}
}
tick();setInterval(tick,1000);
</script></body></html>"""


def _retitle_page() -> None:
    """Swap the pinned default engine port out of the page <title>."""
    global PAGE
    PAGE = PAGE.replace("SGLang 7007 负载监控</title>",
                        "SGLang %d 负载监控</title>" % SGLANG_PORT)


@app.get("/")
def index():
    return HTMLResponse(PAGE)


@app.on_event("startup")
def start_collectors():
    for fn in (_metrics_thread, _host_thread, _log_thread, _proc_thread,
               _persist_thread, _log_cap_thread, _auto_thread):
        threading.Thread(target=fn, daemon=True).start()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="SGLang engine load monitor — community-style dashboard")
    ap.add_argument("--port", type=int, default=PORT,
                    help="dashboard port (default 7100)")
    ap.add_argument("--engine-host", default=SGLANG_HOST,
                    help="engine host (default 127.0.0.1)")
    ap.add_argument("--engine-port", type=int, default=SGLANG_PORT,
                    help="engine port (default 7007)")
    ap.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY", ""),
                    help="engine --api-key (else parsed from --launcher)")
    ap.add_argument("--launcher", default=LAUNCHER_PATH,
                    help="engine launcher script; enables start/stop/restart")
    ap.add_argument("--engine-log", default=ENGINE_LOG,
                    help="engine log file (default /tmp/sglang.log)")
    ap.add_argument("--data-dir", default=None,
                    help="override the persistent data directory")
    ap.add_argument("--electricity-price", type=float, default=None,
                    help="CNY per kWh for the energy card")
    args, _ = ap.parse_known_args()
    PORT = args.port
    SGLANG_HOST = args.engine_host
    SGLANG_PORT = args.engine_port
    _recompute_engine_urls()
    if args.api_key:
        os.environ["SGLANG_API_KEY"] = args.api_key
    LAUNCHER_PATH = args.launcher
    LAUNCHER_TMP = LAUNCHER_PATH + ".monitor-new" if LAUNCHER_PATH else ""
    LAUNCHER_BAK = LAUNCHER_PATH + ".bak-monitor" if LAUNCHER_PATH else ""
    ENGINE_LOG = args.engine_log
    ENGINE_LOG_ARCH = ENGINE_LOG + ".archived"
    if args.data_dir:
        PERSIST_DIR = args.data_dir
    if args.electricity_price is not None:
        _energy["price"] = args.electricity_price
    _retitle_page()
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
