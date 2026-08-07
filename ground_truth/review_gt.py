#!/usr/bin/env python3
"""Phase 10d: local human review + correction UI for the Tesseract seed. 🔴 PHI-DISPLAYING.

>>> ARNAV RUNS THIS. Claude never opens it, never screenshots it, never reads a render or a
>>> review record. A human looks at the images; that is the whole premise (CLAUDE.md §0).

    .venv/bin/python ground_truth/review_gt.py \
        --set renders/v2:ground_truth/seed_v2 \
        --groups ground_truth/seed_groups_v2.csv

then open http://127.0.0.1:8765/ in a browser. `--summary` prints the PHI-free stats instead
of serving (it needs the same --set/--groups so it can diff against the seed).

CONSUMED INPUT SCHEMA (pinned by 10b/10c; this file reads it, never re-derives it)
---------------------------------------------------------------------------------
  <renders_dir>/render_manifest.csv   PHI-FREE. Columns, exactly and in order
                                      (render.py:86,104): image_id, frame_idx, w, h,
                                      sha256, fallback_used.
  <renders_dir>/<image_id>.png        PHI. image_id = 8 hex of sha256(f"{sop_uid}|{frame_idx}")
                                      (render.py:576-585) — STABLE across re-renders, so the
                                      same image can legitimately appear in two render dirs.
  <seed_dir>/<image_id>.json          PHI. Keys (seed_tesseract.py:568-574): image_id, w, h,
                                      provenance, tokens[]. Each token
                                      (seed_tesseract.py:251-258): text (str, RAW), bbox
                                      (list[float] [x0,y0,x1,y1], top-left origin, pixels),
                                      confidence (float), label (str, always "PHI").
  --groups CSV                        PHI-FREE: image_id,group (scripts/build_seed_groups.py:64).
                                      The ONLY stratum interface. This file never reads the
                                      back-map, manifest.csv or gt_sample_*.csv.

WRITTEN OUTPUT — ground_truth/review/<image_id>.json, one per image, atomic
---------------------------------------------------------------------------------
  {image_id, state, round, timestamp, gate:{conf,len,alnum},
   tokens:[{text, box:[x0,y0,x1,y1], label, confidence, seed_index}],
   defer_reason?, note?}

10e MUST GLOB `*.json`, NOT `*`: a save writes `<image_id>.json.<pid>.<tid>.part` first, and a
killed process leaves that temp file behind. `.part` orphans are inert, but a loose glob would
parse one as a review record.

`box` is `GTToken.bbox` exactly (harness/contract.py:96-118) so 10e maps 1:1 with zero
re-interpretation. FOR WHOEVER WRITES 10e: `round`, `timestamp`, `gate`, `confidence` and
`seed_index` are ANNOTATION PROVENANCE ONLY. None of them may become a gt.csv column — a
timestamp in gt.csv makes its hash non-reproducible and breaks the frozen-artifact
guarantee (10a's column spec already forbids it). `note` is PHI by assumption and 10e must
never read it. `timestamp` must never be used for ordering or dedup (D-1.1).

PHI RULES BAKED IN HERE
  - Binds 127.0.0.1 only, on a constant port. No auth, BECAUSE it cannot be reached off-box.
  - Request logging is silenced; nothing PHI ever reaches stdout.
  - Token text travels only in request/response BODIES — never a URL, query string or header.
  - Handler exceptions return a fixed string; no traceback, no body echo (carry-over #1).
  - No outbound request of any kind, from the server or the page.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socketserver
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- Constants (D-10d record item 1: NOT configurable, deliberately) ------------------

# The server needs no authentication *because* it cannot be reached from off-box. Making
# the bind configurable would silently remove the only thing standing in for auth, so there
# is no --host, no --port, no HOST/PORT env read. 8765 avoids the usual dev-server ports.
# If it is ever busy, edit this line.
HOST = "127.0.0.1"
PORT = 8765

UI_HTML = Path(__file__).with_name("review_ui.html")

# Round 1 writes here. A later self-agreement pass (D-10.7) writes round 2 to
# ground_truth/review_r2/ so round 1 is never overwritten — that flow is NOT built now, but
# the directory is a constant rather than an inline string so adding it is a one-line edit.
# ANCHORED TO THE REPO, not to the CWD: a CWD-relative path would silently write PHI records
# outside the repo when the server is started from ~ or /tmp, where neither .gitignore nor the
# Read-deny rules that name `ground_truth/review/` apply (CLAUDE.md §6.6).
REVIEW_DIR = Path(__file__).resolve().parent.parent / "ground_truth" / "review"

# Copied from render.py:86,104 rather than imported: this server is stdlib-only and must not
# pull pydicom/numpy/PIL into the PHI-displaying process. The header is validated against
# this tuple on every load, so drift in 10b fails loudly here instead of silently.
MANIFEST_NAME = "render_manifest.csv"
MANIFEST_COLUMNS = ("image_id", "frame_idx", "w", "h", "sha256", "fallback_used")

STATES = ("accepted", "edited", "deferred")
# PHI-free by construction, so --summary can report it as counts. `possible_non_blank_control`
# is the plan §2.3 signal ("a blank control frame found to carry text must leave the control
# set") which would otherwise live only in Arnav's memory. No free-form values in this field.
DEFER_REASONS = ("unreadable", "ambiguous_token", "possible_non_blank_control", "needs_cal")

# D-10d.A = A3. Per-stratum view-time confidence gate. Hidden means EXCLUDED from the record,
# not "accepted" — the token stays on disk in the frozen seed, so an over-filtered stratum is
# fixed by moving the slider instead of re-seeding. These live in ONE dict because the
# UNCOUNTED rows get recalibrated once human word counts exist, and that must be a one-line
# edit. There is NO length key at any stratum, ever: one ct_secondary_capture frame's entire
# real content is `R` and `L`, and a misread single character that gets redacted IS a false
# redaction — the headline metric.
DEFAULT_GATE_CONF = 0.0  # unlisted/unknown stratum: show everything. More boxes costs clicks;
# fewer boxes silently deletes real text, which is the expensive error.
STRATUM_GATES: dict[str, float] = {
    "us_ge": 10.0,  # ~55 real by eye, c10 keeps 52
    "us_philips": 30.0,  # 34 real, c30 keeps 33
    "us_siemens": 30.0,  # 17 real, c30 keeps 17
    "us_toshiba_canon": 10.0,  # UNCOUNTED - loose on purpose
    "us_samsung": 10.0,  # UNCOUNTED - loose on purpose
    "us_sonosite": 10.0,  # UNCOUNTED - loose on purpose
    "us_other": 10.0,  # UNCOUNTED - loose on purpose
    "ct_secondary_capture": 0.0,  # no confidence floor discriminates
    "mg_2d": 60.0,  # ~1.5 real vs 363.7 ungated
    "mg_tomo": 60.0,  # blank stratum
    "ct_scout": 60.0,  # blank stratum
    "ct_axial": 60.0,  # blank control
}


def visible(conf: float | None, gate: float) -> bool:
    """Is this token shown to the reviewer (and therefore kept in the record)?

    Two tokens are NEVER hidden, at any gate value:
      * `conf is None` — drawn by hand. A human-drawn box is not a Tesseract detection.
      * `conf < 0` — seed_tesseract.py:134 uses CONF_FALLBACK = -1.0 when Tesseract reports
        no confidence at all. "No signal" is not "low confidence"; gating it away would
        silently delete real text, which is the expensive error (D-10d.A).
    There is no length term here, deliberately — see STRATUM_GATES.
    """
    return conf is None or conf < 0 or conf >= gate


class SetupError(Exception):
    """A startup problem. Raised, never swallowed — a silently dropped image is D-10c.4."""


# --- Loading -------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != MANIFEST_COLUMNS:
            raise SetupError(f"{path}: header is not {list(MANIFEST_COLUMNS)}")
        return list(reader)


def load_groups(path: Path) -> dict[str, str]:
    """image_id -> stratum. The groups CSV is the ENTIRE stratum interface (see docstring)."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or {"image_id", "group"} - set(reader.fieldnames):
            raise SetupError(f"{path}: expected columns image_id,group")
        return {r["image_id"]: r["group"] for r in reader}


def parse_set(spec: str) -> tuple[Path, Path]:
    renders, sep, seed = spec.partition(":")
    if not sep or not renders or not seed:
        raise SetupError(f"--set must be <renders_dir>:<seed_dir>, got {spec!r}")
    return Path(renders), Path(seed)


def load_pairs(specs: list[str], groups: dict[str, str]) -> tuple[dict[str, dict], dict[str, int]]:
    """Union of every (renders_dir, seed_dir) pair. Every failure is loud.

    A missing seed JSON is a STARTUP FAILURE, not a skipped image: an image dropped here
    never reaches gt.csv and nobody ever finds out, which is exactly what D-10c.4 exists to
    prevent.
    """
    images: dict[str, dict] = {}
    stats = {"dup_identical": 0, "no_stratum": 0}
    for spec in specs:
        renders_dir, seed_dir = parse_set(spec)
        manifest = renders_dir / MANIFEST_NAME
        if not manifest.is_file():
            raise SetupError(f"{manifest} not found — is {renders_dir} a 10b renders dir?")
        for row in _read_manifest(manifest):
            image_id = row["image_id"]
            png, seed = renders_dir / f"{image_id}.png", seed_dir / f"{image_id}.json"
            if not png.is_file():
                raise SetupError(f"{manifest} lists {image_id} but {png} is missing")
            if not seed.is_file():
                raise SetupError(
                    f"{manifest} lists {image_id} but its seed {seed} is missing. "
                    "Seed that batch before reviewing — a silently skipped image never "
                    "reaches gt.csv (D-10c.4)."
                )
            prev = images.get(image_id)
            if prev is not None:
                # image_id = sha256(sop_uid|frame_idx) is STABLE across re-renders, so the
                # same image legitimately appears in several batch dirs (renders/v2 is a
                # superset of renders/gt). Identical manifest sha256 proves identical pixels
                # -> dedupe, first --set wins, counted and reported. A DIFFERING sha256 means
                # two different images share an id: merging their ground truth is
                # unrecoverable, so that is a hard error naming both paths (same argument as
                # render.py:_check_ids). Decided with Arnav 2026-08-07.
                if prev["sha256"] != row["sha256"]:
                    raise SetupError(
                        f"duplicate image_id {image_id} with DIFFERENT pixels:\n"
                        f"  {prev['manifest']}  sha256={prev['sha256']}\n"
                        f"  {manifest}  sha256={row['sha256']}\n"
                        "Refusing to merge two images' ground truth."
                    )
                # Identical pixels are not enough: the two SEEDS must also come from the same
                # Tesseract build and traineddata. Silently keeping the first pair's seed would
                # mix seeder versions inside one gt.csv, which rule #9 says must be surfaced
                # loudly, not resolved by argument order.
                if read_json(prev["seed"])["provenance"] != read_json(seed)["provenance"]:
                    raise SetupError(
                        f"duplicate image_id {image_id}: same pixels, but the two seeds have "
                        f"DIFFERENT provenance (rule #9 — seeder versions are not comparable):\n"
                        f"  {prev['seed']}\n  {seed}\nRe-seed one of them, or pass only one pair."
                    )
                stats["dup_identical"] += 1
                continue
            stratum = groups.get(image_id)
            if stratum is None:
                stats["no_stratum"] += 1
            images[image_id] = {
                "image_id": image_id,
                "png": png,
                "seed": seed,
                "manifest": manifest,
                "sha256": row["sha256"],
                "w": int(row["w"]),
                "h": int(row["h"]),
                "stratum": stratum or "(unknown)",
                "gate_conf": STRATUM_GATES.get(stratum or "", DEFAULT_GATE_CONF),
            }
    if not images:
        raise SetupError("no images — check --set")
    return images, stats


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def record_path(image_id: str) -> Path:
    return REVIEW_DIR / f"{image_id}.json"


def write_record(image_id: str, payload: dict) -> dict:
    """Validate and atomically write one review record. Returns what was written.

    Atomic (.part + os.replace, carry-over #7) because a review JSON is the only record of
    human annotation work and cannot be re-derived: a kill mid-write must leave the previous
    valid file, never a truncated one.
    """
    state = payload.get("state")
    if state not in STATES:
        raise ValueError("bad state")
    reason = payload.get("defer_reason")
    if state == "deferred":
        if reason not in DEFER_REASONS:
            raise ValueError("deferred requires a defer_reason from the enum")
    elif reason is not None:
        raise ValueError("defer_reason is only valid on a deferred record")
    # A deferred image is UNDECIDED, so it records NO tokens — enforced here rather than
    # trusted from the client. Otherwise a defer would freeze the unvetted seed strings into a
    # record, and 10e would be skipping them on `state` alone with real-looking rows sitting
    # underneath. Re-opening a deferred image re-reads the seed, which is what "come back to
    # this" should mean.
    raw_tokens = [] if state == "deferred" else (payload.get("tokens") or [])
    tokens = []
    seen_idx: set[int] = set()
    for tok in raw_tokens:
        box = [float(v) for v in tok["box"]]
        if len(box) != 4:
            raise ValueError("box must be [x0,y0,x1,y1]")
        if tok["label"] not in ("PHI", "KEEP"):
            raise ValueError("label must be PHI or KEEP")  # binary; there is no OTHER
        if not str(tok["text"]).strip():
            # An empty box cannot become a gt.csv row: it has no token_text to score against.
            raise ValueError("every token needs text — fill the box or delete it")
        idx = tok.get("seed_index")
        if idx is not None:
            # Unvalidated indexes silently corrupt the seed-quality diff: a duplicate makes
            # collect() count one seed token twice, and a negative one compares against the
            # WRONG seed token (Python's negative indexing) while passing an `idx < len` guard.
            idx = int(idx)
            if idx < 0 or idx in seen_idx:
                raise ValueError("seed_index must be unique and non-negative")
            seen_idx.add(idx)
        conf = tok.get("confidence")
        tokens.append({
            "text": str(tok["text"]),
            "box": box,
            "label": tok["label"],
            # None = drawn by hand. A human-drawn box is not a Tesseract detection, so no
            # confidence gate may ever hide it.
            "confidence": None if conf is None else float(conf),
            "seed_index": idx,
        })
    record = {
        "image_id": image_id,
        "state": state,
        # D-10.7: which review round this record belongs to. Round 2 is not built yet.
        "round": int(payload.get("round") or 1),
        # Provenance, never an index: not used for ordering or dedup anywhere (D-1.1), and
        # never a gt.csv column (it would break the frozen hash).
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        # The effective view-time gate at decision time, in the same {conf,len,alnum} shape
        # the metric gate (D-10d.B) uses so both read as the same kind of object. THIS IS
        # ANNOTATION PROVENANCE ONLY: it says what the human was shown, never what should be
        # scored. `len` is pinned at 1 and `alnum` at True server-side — the UI has no length
        # control and this file refuses to record one.
        "gate": {"conf": float((payload.get("gate") or {}).get("conf", DEFAULT_GATE_CONF)),
                 "len": 1, "alnum": True},
        "tokens": tokens,
        # Seed indexes the human DELETED as spurious. Needed because `tokens` holds only what
        # was VISIBLE at the gate, so "absent from tokens" is ambiguous: it means either
        # "hidden by the gate" (must come back when the slider drops) or "deleted" (must stay
        # gone). Without this the two are indistinguishable on reopen and a deleted box
        # resurrects — putting invented text into gt.csv, the dangerous axis. Provenance for
        # the UI; 10e reads `tokens` and ignores this.
        "deleted_seed_indexes": sorted(
            {int(i) for i in (payload.get("deleted_seed_indexes") or []) if int(i) >= 0}
        ),
    }
    if state == "deferred":
        record["defer_reason"] = reason
    note = payload.get("note")
    if note:
        # PHI BY ASSUMPTION — Arnav may well type a token in here. Never printed to stdout,
        # never in --summary, never in a URL/header/log/error, never read by 10e.
        record["note"] = str(note)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dest = record_path(image_id)
    # Per-thread temp name: ThreadingHTTPServer can run two saves of the same image_id
    # concurrently, and a shared .part name would let one thread's os.replace consume the
    # other's file. os.replace itself is atomic, so `dest` is never partial either way.
    tmp = dest.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.part")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())  # so the rename cannot publish a zero-length file
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return record


# --- Server ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "review_gt"
    sys_version = ""

    # http.server logs every request path to stderr by default. A path here is only ever a
    # hash, but silencing the log is what makes "nothing PHI to stdout" structural instead of
    # incidental. log_error/log_request route through log_message, so this covers them.
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _fail(self, code: int = 500) -> None:
        # A FIXED string: no exception text, no traceback, no echo of the request body. The
        # browser is a display surface; the terminal and the response body are not.
        self._send(code, b"request failed", "text/plain")

    def _entry(self, path: str, prefix: str) -> dict | None:
        # Only ids present in the loaded set resolve, so no path traversal is possible.
        return self.server.images.get(path[len(prefix):])

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        try:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/index":
                self._json([
                    {"image_id": e["image_id"], "stratum": e["stratum"],
                     "gate_conf": e["gate_conf"],
                     "state": (read_json(record_path(i))["state"]
                               if record_path(i).is_file() else "unreviewed")}
                    for i, e in self.server.images.items()
                ])
            elif path.startswith("/api/png/"):
                entry = self._entry(path, "/api/png/")
                if not entry:
                    return self._fail(404)
                self._send(200, entry["png"].read_bytes(), "image/png")
            elif path.startswith("/api/image/"):
                entry = self._entry(path, "/api/image/")
                if not entry:
                    return self._fail(404)
                seed = read_json(entry["seed"])
                saved = record_path(entry["image_id"])
                self._json({
                    "image_id": entry["image_id"], "w": entry["w"], "h": entry["h"],
                    "stratum": entry["stratum"], "default_gate": entry["gate_conf"],
                    "seed_tokens": seed.get("tokens", []),
                    "saved": read_json(saved) if saved.is_file() else None,
                })
            else:
                self._fail(404)
        except Exception:
            self._fail()

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        try:
            path = self.path.split("?", 1)[0]
            entry = self._entry(path, "/api/save/") if path.startswith("/api/save/") else None
            if not entry:
                return self._fail(404)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            record = write_record(entry["image_id"], json.loads(body))
            self._json({"state": record["state"], "n_tokens": len(record["tokens"])})
        except (ValueError, KeyError, TypeError):
            self._fail(400)  # the payload was bad
        except Exception:
            self._fail(500)  # the SERVER failed (disk full, permissions) — do not blame the
            # client, or the page reports "invalid edit" for a save that must be retried


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, images: dict[str, dict]):
        self.images = images
        super().__init__(addr, Handler)

    def handle_error(self, request, client_address) -> None:
        # socketserver prints a traceback to stderr by default; a traceback can carry a
        # request body, and a request body carries token text.
        pass

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn(), which can issue a reverse-DNS
        # lookup — a network request. It carries no PHI, but this page's guarantee is that it
        # is INCAPABLE of egress, so bind through TCPServer and set the names by hand.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def serve(images: dict[str, dict], port: int = PORT) -> None:
    with Server((HOST, port), images) as httpd:
        print(f"review UI on http://{HOST}:{httpd.server_address[1]}/  (ctrl-c to stop)")
        httpd.serve_forever()


# --- PHI-free summary -----------------------------------------------------------------


def _pct(vals: list[float], q: float, nd: int = 1) -> float:
    """Nearest-rank percentile: ceil(q*n)-1, NOT int(q*n).

    `int(q*n)` returns the MAXIMUM for every p90 with n <= 10, and for p50 with n = 2 — so a
    small stratum's geometry line would overstate box height. This number is the one that
    settles whether iou_thr = 0.5 is right, so it has to be the percentile it claims to be.
    """
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[max(0, math.ceil(q * len(s)) - 1)], nd)


def _pc(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "-"


def _blank_stats() -> dict:
    return {"total": 0, "accepted": 0, "edited": 0, "deferred": 0, "unreviewed": 0,
            "unchanged": 0, "text_fixed": 0, "box_fixed": 0, "deleted": 0, "added": 0,
            "hidden": 0, "non_default_gate": 0, "w": [], "h": [], "hrel": [],
            "defers": dict.fromkeys(DEFER_REASONS, 0)}


def collect(images: dict[str, dict]) -> dict[str, dict]:
    """Per-stratum PHI-free counters. Never a token string, never a note, never a box."""
    out: dict[str, dict] = {"ALL": _blank_stats()}
    for image_id, entry in images.items():
        for key in ("ALL", entry["stratum"]):
            st = out.setdefault(key, _blank_stats())
            st["total"] += 1
        rec_path = record_path(image_id)
        if not rec_path.is_file():
            for key in ("ALL", entry["stratum"]):
                out[key]["unreviewed"] += 1
            continue
        rec = read_json(rec_path)
        if rec["state"] == "deferred":
            # A deferred image is UNDECIDED, not "every seed box deleted". Counting its
            # empty token list as deletions would make the seed look far worse than it is,
            # and its geometry does not exist yet. Only the state and the reason count.
            for key in ("ALL", entry["stratum"]):
                out[key]["deferred"] += 1
                out[key]["defers"][rec["defer_reason"]] += 1
            continue
        seed_tokens = read_json(entry["seed"]).get("tokens", [])
        gate = float(rec.get("gate", {}).get("conf", 0.0))
        shown = {i for i, t in enumerate(seed_tokens) if visible(t.get("confidence"), gate)}
        kept = {t["seed_index"] for t in rec["tokens"] if t.get("seed_index") is not None}
        for key in ("ALL", entry["stratum"]):
            st = out[key]
            st[rec["state"]] += 1
            st["hidden"] += len(seed_tokens) - len(shown)
            st["non_default_gate"] += int(gate != entry["gate_conf"])
            st["deleted"] += len(shown - kept)
            st["added"] += sum(1 for t in rec["tokens"] if t.get("seed_index") is None)
            for tok in rec["tokens"]:
                w, h = tok["box"][2] - tok["box"][0], tok["box"][3] - tok["box"][1]
                st["w"].append(w)
                st["h"].append(h)
                st["hrel"].append(h / entry["h"] if entry["h"] else 0.0)
                idx = tok.get("seed_index")
                if idx is None or idx >= len(seed_tokens):
                    continue
                seed_tok = seed_tokens[idx]
                if tok["text"] != seed_tok["text"]:
                    st["text_fixed"] += 1
                elif [float(v) for v in seed_tok["bbox"]] != tok["box"]:
                    st["box_fixed"] += 1
                else:
                    st["unchanged"] += 1
    return out


def print_summary(images: dict[str, dict]) -> None:
    stats = collect(images)
    rounds: dict[int, int] = {}
    stamps: list[str] = []
    for image_id in images:
        p = record_path(image_id)
        if p.is_file():
            rec = read_json(p)
            rounds[rec.get("round", 1)] = rounds.get(rec.get("round", 1), 0) + 1
            stamps.append(rec["timestamp"][:10])
    # Every block is reported per stratum as well as pooled. Pooling twelve strata is the
    # defect that made the ct_scout sweep meaningless and then hid the ultrasound gate
    # failure for two days (CLAUDE.md §8) — a pooled-only summary would repeat it.
    for key in ["ALL"] + sorted(k for k in stats if k != "ALL"):
        s = stats[key]
        done = s["accepted"] + s["edited"]
        print(f"\n=== {key}  ({s['total']} images) ===")
        print(f"  progress   accepted {s['accepted']}  edited {s['edited']}  "
              f"deferred {s['deferred']}  unreviewed {s['unreviewed']}")
        tot = s["unchanged"] + s["text_fixed"] + s["box_fixed"] + s["deleted"]
        if tot:
            print(f"  seed       kept {s['unchanged']} ({_pc(s['unchanged'], tot)})  "
                  f"text-fixed {s['text_fixed']} ({_pc(s['text_fixed'], tot)})  "
                  f"box-fixed {s['box_fixed']} ({_pc(s['box_fixed'], tot)})  "
                  f"deleted {s['deleted']} ({_pc(s['deleted'], tot)})  added {s['added']}")
        if s["w"]:
            # The measurement that settles whether iou_thr = 0.5 is right (EasyOCR's boxes
            # measured ~1.9x taller than the text they bound). Produce the number here;
            # iou_thr and harness/matching.py are NOT touched in this phase.
            print(f"  geometry   w p10/p50/p90 {_pct(s['w'], .1)}/{_pct(s['w'], .5)}/"
                  f"{_pct(s['w'], .9)}px  h {_pct(s['h'], .1)}/{_pct(s['h'], .5)}/"
                  f"{_pct(s['h'], .9)}px  h/img p10/p50/p90 {_pct(s['hrel'], .1, 4)}/"
                  f"{_pct(s['hrel'], .5, 4)}/{_pct(s['hrel'], .9, 4)}")
        if s["deferred"]:
            print("  defers     " + "  ".join(f"{k} {v}" for k, v in s["defers"].items() if v))
        if done or s["deferred"]:
            print(f"  gate       hid {s['hidden']} seed tokens  "
                  f"non-default gate on {s['non_default_gate']} images")
    print(f"\nrounds: {rounds or '-'}   date range: "
          f"{min(stamps) if stamps else '-'} .. {max(stamps) if stamps else '-'}")


# --- CLI ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 10d review UI (localhost only)")
    ap.add_argument("--set", dest="sets", action="append", required=True, metavar="RENDERS:SEED",
                    help="repeatable <renders_dir>:<seed_dir> pair; the review set is the union")
    ap.add_argument("--groups", type=Path, required=True,
                    help="PHI-free image_id,group CSV from scripts/build_seed_groups.py")
    ap.add_argument("--summary", action="store_true", help="print PHI-free stats and exit")
    args = ap.parse_args(argv)

    try:
        groups = load_groups(args.groups)
        images, stats = load_pairs(args.sets, groups)
    except SetupError as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # TYPE NAME ONLY for anything this file did not author (a malformed seed JSON, a
        # permission error). Same rule as render.py: an uncaught traceback would render the
        # failing input into the terminal, and a seed JSON's content is token text.
        print(f"startup failed: {type(exc).__name__} while loading inputs", file=sys.stderr)
        return 2

    if args.summary:
        print_summary(images)
        return 0

    print(f"{len(images)} images from {len(args.sets)} pair(s)")
    if stats["dup_identical"]:
        print(f"{stats['dup_identical']} duplicate image_id(s) across pairs, all sha256-identical "
              "-> deduped, first --set wins")
    if stats["no_stratum"]:
        print(f"{stats['no_stratum']} image_id(s) not in the groups CSV -> default gate c"
              f"{DEFAULT_GATE_CONF:g} (show everything)")
    serve(images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
