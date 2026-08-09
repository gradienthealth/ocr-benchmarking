"""Synthetic-only tests for `ground_truth/review_gt.py` + `review_ui.html` (Phase 10d). 🟢 PHI-FREE.

Every fixture here is fabricated in memory by `tests/synthetic.py` (`CMFN`/`GRDN`-style fake
tokens, blank frames) and saved into `tmp_path`. Zero real renders, zero `.dcm`, zero
`gt.csv`, zero review records from the real `ground_truth/review/`. The HTTP server is bound
on an EPHEMERAL loopback port (`Server((HOST, 0), images)`) so a test run can never collide
with, or be reached by, the real review session on port 8765.

RULE FOR EVERY TEST IN THIS FILE: **assert on the artifact, not on the label.** A saved
review is verified by reading the written JSON back off disk and comparing token text and
boxes — never by asserting that a `state` field exists or that a file was created. The
Phase-10b lesson: 43 tests passed while the renderer always decoded frame 0, because they
asserted on a manifest column that merely *said* "middle frame".
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ground_truth import review_gt
from tests.synthetic import FAKE_TOKENS, make_blank_image, make_synthetic_image

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_HTML = REPO_ROOT / "ground_truth" / "review_ui.html"
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "block_phi_read.py"

SIZE = (320, 240)  # every synthetic render is this size, so h/img ratios are hand-computable
# An obviously-fake note in the shape Arnav might really type (a token + prose). It is PHI by
# assumption, so it must reproduce through the reopen endpoint and appear NOWHERE else.
FAKE_NOTE = "CMFN-00421 overlaps the ruler"


# ---------------------------------------------------------------------------
# fixture builders — synthetic renders dir + seed dir + groups CSV
# ---------------------------------------------------------------------------


def _seed_tok(text: str, box: tuple[float, float, float, float], conf: float) -> dict:
    """One seed token in seed_tesseract.py's shape: text / bbox / confidence / label."""
    return {"text": text, "bbox": [float(v) for v in box], "confidence": float(conf),
            "label": "PHI"}


def _build_set(
    root: Path,
    name: str,
    seeds: dict[str, list[dict]],
    *,
    sha_override: dict[str, str] | None = None,
    skip_seed: tuple[str, ...] = (),
    blank: tuple[str, ...] = (),
    provenance: dict | None = None,
) -> tuple[Path, Path]:
    """Write one (renders_dir, seed_dir) pair: a real synthetic PNG + manifest + seed JSON."""
    renders = root / name / "renders"
    seed_dir = root / name / "seed"
    renders.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_id, tokens in seeds.items():
        png = renders / f"{image_id}.png"
        if image_id in blank:
            img, _ = make_blank_image(SIZE)
        else:
            img, _ = make_synthetic_image(list(FAKE_TOKENS[:2]), size=SIZE)
        img.save(png, format="PNG")
        sha = (sha_override or {}).get(image_id) or hashlib.sha256(png.read_bytes()).hexdigest()
        rows.append((image_id, 0, SIZE[0], SIZE[1], sha, 0))
        if image_id in skip_seed:
            continue
        (seed_dir / f"{image_id}.json").write_text(
            json.dumps({"image_id": image_id, "w": SIZE[0], "h": SIZE[1],
                        "provenance": provenance or {"synthetic": True}, "tokens": tokens}),
            encoding="utf-8",
        )
    lines = [",".join(review_gt.MANIFEST_COLUMNS)]
    lines += [",".join(str(v) for v in row) for row in rows]
    (renders / review_gt.MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return renders, seed_dir


def _groups_csv(path: Path, mapping: dict[str, str]) -> Path:
    lines = ["image_id,group"] + [f"{k},{v}" for k, v in mapping.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _review_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module's REVIEW_DIR at tmp_path (it is resolved at call time)."""
    target = tmp_path / "review"
    monkeypatch.setattr(review_gt, "REVIEW_DIR", target)
    return target


# ---------------------------------------------------------------------------
# a faithful stand-in for the browser client
# ---------------------------------------------------------------------------


def _ui_visible(tok: dict, gate: float) -> bool:
    """Hand-transcribed mirror of review_ui.html's `visible()`:

        return t.confidence == null || t.confidence < 0 || t.confidence >= g;

    (`== null` matches null AND undefined, i.e. a missing confidence key — which is what
    Python's `.get()` returning None models here.)

    Transcribed rather than delegated to `review_gt.visible()` on purpose — a simulation that
    called the server's own function could never notice the page and the server disagreeing.
    `test_page_and_server_gate_agree` pins the two implementations against each other.
    """
    conf = tok.get("confidence")
    return conf is None or conf < 0 or conf >= gate


#: Mirrors review_ui.html's DEFAULT_LABEL. The page ignores the seed's own label — it is a
#: hardcoded "PHI" placeholder (D-10c.2), not information — and starts every token here.
UI_DEFAULT_LABEL = "KEEP"


def _ui_tokens_from_seed(seed_tokens: list[dict]) -> list[dict]:
    """What review_ui.html builds from `seed_tokens` on first open (load(): bbox -> box)."""
    return [{"text": t["text"], "box": list(t["bbox"]), "label": UI_DEFAULT_LABEL,
             "confidence": t["confidence"], "seed_index": k}
            for k, t in enumerate(seed_tokens)]


def _ui_reload_tokens(seed_tokens: list[dict], saved: dict | None) -> list[dict]:
    """Mirror of review_ui.html `load()` + `restoreSaved()`: the working set after a reopen.

    Start from the seed, overlay every saved token onto its seed_index, keep hand-added ones,
    and drop the seed indexes the record lists as DELETED. Anything merely hidden by the gate
    comes back. Deletion is read from `deleted_seed_indexes`, not inferred from "absent from
    tokens at the saved gate" — the inference resurrected a deleted box whenever the gate was
    raised above its confidence before saving.
    """
    toks = _ui_tokens_from_seed(seed_tokens)
    if not saved:
        return toks
    by_index: dict[int, dict] = {}
    added: list[dict] = []
    for tok in saved["tokens"]:
        if tok.get("seed_index") is None:
            added.append(dict(tok))
        else:
            by_index[tok["seed_index"]] = dict(tok)
    dropped = set(saved.get("deleted_seed_indexes") or [])
    kept = [dict(by_index.get(t["seed_index"], t)) for t in toks
            if t["seed_index"] not in dropped]
    return kept + added


def _ui_save_body(tokens: list[dict], gate: float, state: str = "accepted",
                  n_seed: int | None = None, **extra) -> dict:
    """What review_ui.html POSTs: only tokens passing the view gate are in the record.

    `deleted_seed_indexes` mirrors the page's `deletedIndexes()` — every seed index no longer
    in the working set, independent of the gate (the working set keeps hidden tokens and drops
    only deleted ones). `n_seed` is the seed's token count; omit it when nothing was deleted.
    """
    live = {t["seed_index"] for t in tokens if t.get("seed_index") is not None}
    body = {"state": state, "gate": {"conf": gate},
            "tokens": [t for t in tokens if _ui_visible(t, gate)],
            "deleted_seed_indexes": [k for k in range(n_seed or 0) if k not in live]}
    body.update(extra)
    return body


class _Client:
    """urllib client that records every URL it requests (so a test can prove no PHI in one)."""

    def __init__(self, srv) -> None:
        self.srv = srv
        self.base = f"http://{review_gt.HOST}:{srv.server_address[1]}"
        self.urls: list[str] = []

    def _open(self, req: urllib.request.Request, url: str) -> tuple[int, bytes]:
        self.urls.append(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def get(self, path: str) -> tuple[int, bytes]:
        url = self.base + path
        return self._open(urllib.request.Request(url), url)

    def get_json(self, path: str) -> dict | list:
        code, body = self.get(path)
        assert code == 200, (path, code, body)
        return json.loads(body)

    def post(self, path: str, payload) -> tuple[int, bytes]:
        url = self.base + path
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        return self._open(req, url)

    def save(self, image_id: str, payload) -> tuple[int, bytes]:
        return self.post(f"/api/save/{image_id}", payload)


@contextlib.contextmanager
def _running(images: dict[str, dict]):
    """Run the real Server on an ephemeral loopback port in a daemon thread."""
    srv = review_gt.Server((review_gt.HOST, 0), images)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(srv)
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# 1. zero egress in the page itself
# ---------------------------------------------------------------------------


def test_ui_html_is_incapable_of_egress() -> None:
    """The page's own bytes: no absolute URL, no external asset, only relative fetch targets."""
    ui = UI_HTML.read_text(encoding="utf-8")
    low = ui.lower()

    for token in ("http://", "https://", "//", "@import", "url(", "<link", "<iframe",
                  "xmlhttprequest", "websocket", "sendbeacon", "eventsource", "importscripts",
                  "navigator.sendbeacon", "document.write", "<form"):
        assert token not in low, f"{token!r} in review_ui.html — egress surface"

    # `//` is absent above, which also covers protocol-relative refs; belt-and-braces on tags
    # that can pull a remote asset even without a scheme.
    assert not re.search(r"<(script|img|iframe|source|embed|object|audio|video)[^>]*\bsrc\s*=", low)
    # spellcheck/autofill are egress paths that are not fetches, so they get their own check
    assert 'spellcheck="false"' in ui and 'autocomplete="off"' in ui
    assert "inp.spellcheck = false" in ui and "inp.autocomplete = 'off'" in ui

    # Every network call the page can make, and every asset URL it assigns, must be a
    # same-origin RELATIVE path: leading "/", no scheme, no authority.
    calls = re.findall(r"\bfetch\(\s*['\"]([^'\"]*)['\"]", ui)
    assert len(calls) == len(re.findall(r"\bfetch\(", ui)) == 3, calls
    srcs = re.findall(r"\.src\s*=\s*['\"]([^'\"]*)['\"]", ui)
    assert srcs, "expected the canvas image source assignment"
    for target in calls + srcs:
        assert target.startswith("/"), target
        assert not target.startswith("//"), target
        assert ":" not in target, target
    assert set(calls) == {"/api/index", "/api/image/", "/api/save/"}
    assert set(srcs) == {"/api/png/"}


# ---------------------------------------------------------------------------
# 2. loopback-only bind, not configurable
# ---------------------------------------------------------------------------


def test_server_binds_loopback_and_bind_is_not_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    assert review_gt.HOST == "127.0.0.1"
    assert isinstance(review_gt.PORT, int) and review_gt.PORT == 8765

    _review_dir(tmp_path, monkeypatch)
    renders, seed = _build_set(tmp_path, "a", {"aaaa0001": [_seed_tok("CMFN", (1, 1, 9, 9), 90)]})
    images, _ = review_gt.load_pairs([f"{renders}:{seed}"], {})

    # The artifact: what the socket is actually bound to (0.0.0.0 would fail here).
    with _running(images) as client:
        assert client.srv.socket.getsockname()[0] == "127.0.0.1"
        assert client.srv.server_address[0] == "127.0.0.1"
        assert client.get("/api/index")[0] == 200

    src = inspect.getsource(review_gt)
    # HOST/PORT are each bound exactly once, to a literal — nothing reassigns them.
    assert re.findall(r"^HOST\s*=.*$", src, re.M) == ['HOST = "127.0.0.1"']
    assert re.findall(r"^PORT\s*=.*$", src, re.M) == ["PORT = 8765"]
    assert not re.findall(r"^\s+(HOST|PORT)\s*=", src, re.M)
    assert "Server((HOST, port)" in src  # serve() cannot bind anything but HOST
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    # HTTPServer.server_bind would call socket.getfqdn() — a reverse-DNS lookup, i.e. a
    # network request from a server whose guarantee is that it is incapable of egress
    assert "getfqdn" not in code
    assert "socketserver.TCPServer.server_bind(self)" in inspect.getsource(review_gt.Server)

    # No environment can move the bind.
    assert not re.search(r"os\.environ|environb|getenv|EnvironmentVariable", code)

    # The parser's real option strings: nothing to set a host or a port.
    declared = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))
    assert declared == {"--set", "--groups", "--summary"}
    with pytest.raises(SystemExit):
        review_gt.main(["--help"])
    help_text = capsys.readouterr().out
    assert set(re.findall(r"--[a-zA-Z][\w-]*", help_text)) == {
        "--set", "--groups", "--summary", "--help"}
    for banned in ("--host", "--port", "--bind", "--listen"):
        assert banned not in help_text


# ---------------------------------------------------------------------------
# 3. request log silenced
# ---------------------------------------------------------------------------


def test_no_endpoint_writes_anything_to_stdout_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    _review_dir(tmp_path, monkeypatch)
    seed = [_seed_tok("CMFN", (1, 1, 9, 9), 90)]
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], {})

    capfd.readouterr()  # drop anything emitted while building fixtures
    paths: list[str] = []
    with _running(images) as client:
        for path in ("/", "/index.html", "/api/index", "/api/png/aaaa0001",
                     "/api/image/aaaa0001", "/api/png/nosuchid", "/api/image/nosuchid",
                     "/nope", "/../etc/passwd"):
            client.get(path)
            paths.append(path)
        client.save("aaaa0001", _ui_save_body(_ui_tokens_from_seed(seed), 0.0))
        client.save("nosuchid", {"state": "accepted", "tokens": []})
        client.post("/api/save/aaaa0001", b"{not json")
        client.post("/nope", {"state": "accepted"})
        paths += ["/api/save/aaaa0001", "/api/save/nosuchid", "/nope"]

    captured = capfd.readouterr()
    assert captured.out == "", captured.out
    assert captured.err == "", captured.err
    for path in paths:  # no access-log line for any request path, valid or not
        assert path not in captured.out and path not in captured.err
    assert "aaaa0001" not in captured.out + captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# 4. no PHI anywhere Claude or a log could see it
# ---------------------------------------------------------------------------


def test_token_text_never_reaches_stdout_a_url_or_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    _review_dir(tmp_path, monkeypatch)
    seed = [_seed_tok(FAKE_TOKENS[1], (10, 10, 50, 34), 90),
            _seed_tok(FAKE_TOKENS[2], (10, 50, 60, 74), 90)]
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    groups = _groups_csv(tmp_path / "groups.csv", {"aaaa0001": "mg_2d"})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], review_gt.load_groups(groups))

    capfd.readouterr()
    with _running(images) as client:
        client.get_json("/api/image/aaaa0001")
        toks = _ui_tokens_from_seed(seed)
        toks[0]["text"] = FAKE_TOKENS[3]  # a correction: new token text in the request BODY
        code, _ = client.save("aaaa0001", _ui_save_body(toks, 0.0, state="edited",
                                                        note=FAKE_NOTE))
        assert code == 200
        urls = list(client.urls)

    # The written record really does hold the token text (so the assertions below mean
    # something): the artifact contains PHI, the channels do not.
    record = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
    assert record["tokens"][0]["text"] == FAKE_TOKENS[3]
    assert record["note"] == FAKE_NOTE

    rc = review_gt.main(["--set", f"{renders}:{seed_dir}", "--groups", str(groups), "--summary"])
    assert rc == 0
    captured = capfd.readouterr()
    haystack = captured.out + captured.err + "\n".join(urls)
    for token in (*FAKE_TOKENS, FAKE_NOTE, "CMFN-00421"):
        assert token not in haystack, f"{token!r} leaked into stdout/stderr/URL"
    for url in urls:  # every path segment is the hashed image_id, never text
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/[a-z/._]*(aaaa0001|nosuchid)?", url), url


# ---------------------------------------------------------------------------
# 5. round-trip: the record on disk, and what reopen returns
# ---------------------------------------------------------------------------


def test_saved_record_round_trips_tokens_and_boxes_not_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_dir = _review_dir(tmp_path, monkeypatch)
    seeds = {
        "aaaa0001": [_seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 90)],
        "aaaa0002": [_seed_tok(FAKE_TOKENS[1], (10, 10, 50, 34), 90)],
        "aaaa0003": [_seed_tok(FAKE_TOKENS[2], (10, 10, 60, 34), 90)],
        "aaaa0004": [_seed_tok(FAKE_TOKENS[3], (10, 10, 70, 34), 90)],
    }
    renders, seed_dir = _build_set(tmp_path, "a", seeds)
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], {})

    with _running(images) as client:
        # accepted: unchanged text + box
        client.save("aaaa0001", _ui_save_body(_ui_tokens_from_seed(seeds["aaaa0001"]), 0.0))
        # edited: text corrected AND box moved, plus a hand-drawn box
        edited = _ui_tokens_from_seed(seeds["aaaa0002"])
        edited[0]["text"] = "GRDN9999"
        edited[0]["box"] = [11.0, 12.0, 55.0, 40.0]
        # explicit, not inherited: the page defaults every token to KEEP, so PHI here is the
        # reviewer's decision — and it keeps both labels in this round-trip assertion.
        edited[0]["label"] = "PHI"
        edited.append({"text": "ACC-0099", "box": [100.0, 100.0, 160.0, 130.0],
                       "label": "KEEP", "confidence": None, "seed_index": None})
        client.save("aaaa0003", _ui_save_body(edited, 0.0, state="edited"))
        # deferred
        client.save("aaaa0002", _ui_save_body([], 0.0, state="deferred",
                                              defer_reason="unreadable"))

        # (a) accepted round-trips text + box byte-for-byte
        disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
        assert disk["state"] == "accepted"
        assert [(t["text"], t["box"]) for t in disk["tokens"]] == [(FAKE_TOKENS[0],
                                                                   [10.0, 10.0, 30.0, 34.0])]
        reopened = client.get_json("/api/image/aaaa0001")
        assert reopened["saved"]["tokens"] == disk["tokens"]

        # (b) edited: reopen serves the SAVED tokens, which differ from the seed
        disk = json.loads(review_gt.record_path("aaaa0003").read_text(encoding="utf-8"))
        assert [(t["text"], t["box"], t["label"], t["seed_index"]) for t in disk["tokens"]] == [
            ("GRDN9999", [11.0, 12.0, 55.0, 40.0], "PHI", 0),
            ("ACC-0099", [100.0, 100.0, 160.0, 130.0], "KEEP", None),
        ]
        reopened = client.get_json("/api/image/aaaa0003")
        assert reopened["saved"]["tokens"] == disk["tokens"]
        assert reopened["saved"]["tokens"][0]["text"] != seeds["aaaa0003"][0]["text"]
        # the untouched seed is still served alongside it
        assert reopened["seed_tokens"] == seeds["aaaa0003"]

        # (c) deferred
        disk = json.loads(review_gt.record_path("aaaa0002").read_text(encoding="utf-8"))
        assert disk["state"] == "deferred" and disk["tokens"] == []
        assert client.get_json("/api/image/aaaa0002")["saved"] == disk

        # (d) an unreviewed image has NO file at all
        assert not review_gt.record_path("aaaa0004").exists()
        assert client.get_json("/api/image/aaaa0004")["saved"] is None
        index = {e["image_id"]: e["state"] for e in client.get_json("/api/index")}
        assert index == {"aaaa0001": "accepted", "aaaa0002": "deferred",
                         "aaaa0003": "edited", "aaaa0004": "unreviewed"}

    assert sorted(p.name for p in review_dir.iterdir()) == [
        "aaaa0001.json", "aaaa0002.json", "aaaa0003.json"]


# ---------------------------------------------------------------------------
# 6. atomicity
# ---------------------------------------------------------------------------


def test_failed_save_leaves_the_previous_record_intact_and_no_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_dir = _review_dir(tmp_path, monkeypatch)
    good = {"state": "accepted", "gate": {"conf": 0},
            "tokens": [{"text": FAKE_TOKENS[0], "box": [1, 2, 3, 4], "label": "PHI",
                        "confidence": 90.0, "seed_index": 0}]}
    review_gt.write_record("aaaa0001", good)
    dest = review_gt.record_path("aaaa0001")
    before = dest.read_text(encoding="utf-8")
    assert json.loads(before)["tokens"][0]["text"] == FAKE_TOKENS[0]
    assert list(review_dir.glob("*.part")) == []  # a clean save leaves no temp behind

    replacement = {"state": "edited", "gate": {"conf": 0},
                   "tokens": [{"text": "GRDN9999", "box": [9, 9, 9, 9], "label": "PHI",
                               "confidence": 90.0, "seed_index": 0}]}

    def _boom(*_args, **_kwargs):
        raise OSError("simulated kill mid-save")

    # (a) killed in the window between writing the temp file and renaming it
    with monkeypatch.context() as ctx:
        ctx.setattr(review_gt.os, "replace", _boom)
        with pytest.raises(OSError, match="simulated kill mid-save"):
            review_gt.write_record("aaaa0001", replacement)
    assert dest.read_text(encoding="utf-8") == before
    assert json.loads(dest.read_text(encoding="utf-8"))["tokens"][0]["text"] == FAKE_TOKENS[0]
    assert list(review_dir.glob("*.part")) == []

    # (b) killed before a single byte is serialised (unserialisable seed_index)
    doomed = json.loads(json.dumps(replacement))
    doomed["tokens"][0]["seed_index"] = {1, 2}
    with pytest.raises(TypeError):
        review_gt.write_record("aaaa0001", doomed)
    assert dest.read_text(encoding="utf-8") == before
    assert list(review_dir.glob("*.part")) == []
    assert sorted(p.name for p in review_dir.iterdir()) == ["aaaa0001.json"]

    # (c) a temp file belonging to ANOTHER writer (different pid/thread — the temp name is
    # per-writer so two concurrent saves cannot consume each other's file) is never adopted:
    # the next good save neither renames it into place nor deletes it.
    foreign = review_dir / "aaaa0001.json.999999.1.part"
    foreign.write_text('{"truncat', encoding="utf-8")
    review_gt.write_record("aaaa0001", replacement)
    after = json.loads(dest.read_text(encoding="utf-8"))
    assert after["state"] == "edited"
    assert after["tokens"][0]["text"] == "GRDN9999"
    assert foreign.read_text(encoding="utf-8") == '{"truncat'
    assert list(review_dir.glob(f"*.{os.getpid()}.*.part")) == []
    assert sorted(p.name for p in review_dir.iterdir()) == [
        "aaaa0001.json", "aaaa0001.json.999999.1.part"]


# ---------------------------------------------------------------------------
# 7. --summary arithmetic
# ---------------------------------------------------------------------------


def _section(out: str, key: str) -> str:
    blocks = out.split("\n=== ")
    for block in blocks[1:]:
        if block.split("  (", 1)[0] == key:
            return block
    raise AssertionError(f"no section for {key!r} in:\n{out}")


def test_summary_counts_and_percentiles_match_hand_computed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _review_dir(tmp_path, monkeypatch)
    seeds = {
        # A: 6 seed tokens; only the conf-10 one is hidden at the c30 the human reviewed at
        # (the CONF_FALLBACK -1 one is never hidden, so `hidden` must come out as 1, not 2).
        "aaaa0001": [
            _seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 90),    # kept unchanged   w20 h24
            _seed_tok(FAKE_TOKENS[1], (10, 50, 50, 74), 90),    # text-fixed       w40 h24
            _seed_tok(FAKE_TOKENS[2], (10, 90, 70, 114), 90),   # box-fixed        w60 h30
            _seed_tok(FAKE_TOKENS[3], (10, 130, 90, 154), 90),  # deleted
            _seed_tok(FAKE_TOKENS[4], (10, 170, 110, 194), 10),  # hidden by the gate
            _seed_tok(FAKE_TOKENS[5], (10, 200, 40, 224), -1.0),  # kept, PHI-labelled w30 h24
        ],
        # B: both kept unchanged, incl. a single-character token.  w6 h14 / w40 h24
        "aaaa0002": [_seed_tok("L", (5, 5, 11, 19), 80),
                     _seed_tok(FAKE_TOKENS[0], (5, 30, 45, 54), 80)],
        "aaaa0003": [_seed_tok(FAKE_TOKENS[1], (5, 5, 45, 29), 80)],  # deferred
        "aaaa0004": [_seed_tok(FAKE_TOKENS[2], (5, 5, 45, 29), 80)],  # never reviewed
    }
    renders, seed_dir = _build_set(tmp_path, "a", seeds)
    groups = _groups_csv(tmp_path / "groups.csv", {
        "aaaa0001": "mg_2d", "aaaa0002": "ct_secondary_capture",
        "aaaa0003": "mg_2d", "aaaa0004": "ct_axial"})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], review_gt.load_groups(groups))

    a_toks = _ui_tokens_from_seed(seeds["aaaa0001"])
    a_toks[1]["text"] = "GRDN9999"                      # text-fixed
    a_toks[2]["box"] = [10.0, 90.0, 70.0, 120.0]        # box-fixed (h 24 -> 30)
    # The page defaults every token to KEEP (DEFAULT_LABEL), so a PHI label is the reviewer's
    # affirmative decision. It must NOT disturb the kept/text-fixed/box-fixed split — the
    # seed's own label is a hardcoded "PHI" placeholder (D-10c.2) and carries no information,
    # so this token still counts as `unchanged`; only the label DISTRIBUTION records it.
    a_toks[5]["label"] = "PHI"
    del a_toks[3]                                       # deleted by the human
    a_toks.append({"text": "ACC-0099", "box": [200.0, 10.0, 260.0, 58.0], "label": "PHI",
                   "confidence": None, "seed_index": None})  # added by hand  w60 h48
    with _running(images) as client:
        assert client.save("aaaa0001", _ui_save_body(a_toks, 30.0, state="edited"))[0] == 200
        assert client.save(
            "aaaa0002", _ui_save_body(_ui_tokens_from_seed(seeds["aaaa0002"]), 0.0))[0] == 200
        assert client.save("aaaa0003", _ui_save_body([], 0.0, state="deferred",
                                                     defer_reason="needs_cal"))[0] == 200

    stats = review_gt.collect(images)
    every = stats["ALL"]
    assert (every["total"], every["accepted"], every["edited"], every["deferred"],
            every["unreviewed"]) == (4, 1, 1, 1, 1)
    # seed quality: A gives 2 kept + 1 text-fixed + 1 box-fixed + 1 deleted + 1 added and
    # hides 1; B gives 2 kept; the deferred image contributes nothing but its state.
    assert (every["unchanged"], every["text_fixed"], every["box_fixed"], every["deleted"],
            every["added"], every["hidden"], every["non_default_gate"]) == (4, 1, 1, 1, 1, 1, 1)
    # label distribution over the RECORDED tokens: A records 5 (the relabelled seed token and
    # the hand-drawn one are PHI, the other 3 keep the KEEP default), B records 2 KEEP. A
    # seed-vs-record label diff is deliberately NOT computed — the seed's label is a hardcoded
    # placeholder, so such a diff would read ~100% on every stratum and mean nothing.
    assert (every["keep"], every["phi"]) == (5, 2)
    assert every["defers"] == {"unreadable": 0, "ambiguous_token": 0,
                               "possible_non_blank_control": 0, "needs_cal": 1}
    mg = stats["mg_2d"]
    assert (mg["total"], mg["edited"], mg["deferred"], mg["unchanged"], mg["text_fixed"],
            mg["box_fixed"], mg["deleted"], mg["added"]) == (2, 1, 1, 2, 1, 1, 1, 1)
    assert (mg["keep"], mg["phi"]) == (3, 2)
    sc = stats["ct_secondary_capture"]
    assert (sc["total"], sc["accepted"], sc["unchanged"], sc["hidden"], sc["deleted"],
            sc["added"], sc["non_default_gate"]) == (1, 1, 2, 0, 0, 0, 0)
    assert (stats["ct_axial"]["total"], stats["ct_axial"]["unreviewed"]) == (1, 1)
    # geometry inputs, hand-listed: A's five recorded boxes then B's two
    assert every["w"] == [20.0, 40.0, 60.0, 30.0, 60.0, 6.0, 40.0]
    assert every["h"] == [24.0, 24.0, 30.0, 24.0, 48.0, 14.0, 24.0]

    review_gt.print_summary(images)
    out = capsys.readouterr().out
    all_block = _section(out, "ALL")
    assert "(4 images)" in out
    assert "progress   accepted 1  edited 1  deferred 1  unreviewed 1" in all_block
    assert ("seed       kept 4 (57.1%)  text-fixed 1 (14.3%)  box-fixed 1 (14.3%)  "
            "deleted 1 (14.3%)  added 1") in all_block
    assert "labels     KEEP 5  PHI 2" in all_block
    # sorted w = [6,20,30,40,40,60,60] -> idx 0/3/6; h = [14,24,24,24,24,30,48] -> idx 0/3/6;
    # h/img = h/240 -> 0.0583/0.1/0.2
    assert ("geometry   w p10/p50/p90 6.0/40.0/60.0px  h 14.0/24.0/48.0px  "
            "h/img p10/p50/p90 0.0583/0.1/0.2") in all_block
    assert "gate       hid 1 seed tokens  non-default gate on 1 images" in all_block
    assert "defers     needs_cal 1" in all_block

    mg_block = _section(out, "mg_2d")
    assert ("seed       kept 2 (40.0%)  text-fixed 1 (20.0%)  box-fixed 1 (20.0%)  "
            "deleted 1 (20.0%)  added 1") in mg_block
    # sorted w = [20,30,40,60,60] -> idx 0/2/4; h = [24,24,24,30,48] -> idx 0/2/4
    assert ("geometry   w p10/p50/p90 20.0/40.0/60.0px  h 24.0/24.0/48.0px  "
            "h/img p10/p50/p90 0.1/0.1/0.2") in mg_block

    sc_block = _section(out, "ct_secondary_capture")
    assert ("seed       kept 2 (100.0%)  text-fixed 0 (0.0%)  box-fixed 0 (0.0%)  "
            "deleted 0 (0.0%)  added 0") in sc_block
    # NEAREST-RANK percentile, idx = ceil(q*n)-1. n=2: p10 and p50 both take idx 0, p90 idx 1.
    # sorted w = [6,40]; h = [14,24]; h/img = h/240 = [0.0583, 0.1].
    assert ("geometry   w p10/p50/p90 6.0/6.0/40.0px  h 14.0/14.0/24.0px  "
            "h/img p10/p50/p90 0.0583/0.0583/0.1") in sc_block

    axial_block = _section(out, "ct_axial")
    assert "progress   accepted 0  edited 0  deferred 0  unreviewed 1" in axial_block
    assert "seed  " not in axial_block and "geometry" not in axial_block
    assert "rounds: {1: 3}" in out


# ---------------------------------------------------------------------------
# 8. the PHI hook actually denies a review record
# ---------------------------------------------------------------------------


def _load_hook():
    spec = importlib.util.spec_from_file_location("block_phi_read_under_test", HOOK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phi_hook_denies_a_review_record_but_not_the_source(
    capsys: pytest.CaptureFixture,
) -> None:
    hook = _load_hook()
    for denied in ("ground_truth/review/abc123.json",
                   "ground_truth/review_r2/abc123.json",
                   str(REPO_ROOT / "ground_truth" / "review" / "abc123.json"),
                   "ground_truth/seed/abc123.json"):
        capsys.readouterr()
        with pytest.raises(SystemExit) as exc:
            hook.check({"tool_name": "Read", "tool_input": {"file_path": denied}})
        assert exc.value.code == 0
        emitted = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
        assert emitted["hookEventName"] == "PreToolUse"
        assert emitted["permissionDecision"] == "deny"
        assert denied in emitted["permissionDecisionReason"]

    # the source files must stay readable, or the phase cannot be worked on at all
    for allowed in ("ground_truth/review_gt.py", "ground_truth/review_ui.html",
                    str(REPO_ROOT / "ground_truth" / "review_gt.py"),
                    "tests/test_review_gt.py", "ground_truth/seed_summary.json"):
        capsys.readouterr()
        assert hook.check({"tool_name": "Read", "tool_input": {"file_path": allowed}}) is None
        assert capsys.readouterr().out == "", allowed


# ---------------------------------------------------------------------------
# 9. gate semantics — the filter that protects the headline metric
# ---------------------------------------------------------------------------


def test_view_gate_excludes_only_by_confidence_and_is_reversible_from_the_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _review_dir(tmp_path, monkeypatch)
    seed = [
        _seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 5),    # below c30
        _seed_tok(FAKE_TOKENS[1], (10, 50, 50, 74), 45),   # above c30
        _seed_tok(FAKE_TOKENS[2], (10, 90, 70, 114), 95),  # above c30
    ]
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], {})
    path = review_gt.record_path("aaaa0001")

    def _written() -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    with _running(images) as client:
        # (a) first open, save at c30: the conf-5 token is EXCLUDED from the record
        served = client.get_json("/api/image/aaaa0001")
        toks = _ui_reload_tokens(served["seed_tokens"], served["saved"])
        assert client.save("aaaa0001", _ui_save_body(toks, 30.0))[0] == 200
        written = _written()
        assert FAKE_TOKENS[0] not in [t["text"] for t in written["tokens"]]
        assert [t["text"] for t in written["tokens"]] == [FAKE_TOKENS[1], FAKE_TOKENS[2]]
        assert [t["seed_index"] for t in written["tokens"]] == [1, 2]
        assert written["gate"]["conf"] == 30.0

        # (b) reopen and save at c0: the seed on disk is untouched, so the token comes back
        served = client.get_json("/api/image/aaaa0001")
        assert [t["text"] for t in served["seed_tokens"]] == list(FAKE_TOKENS[:3])
        toks = _ui_reload_tokens(served["seed_tokens"], served["saved"])
        assert client.save("aaaa0001", _ui_save_body(toks, 0.0))[0] == 200
        written = _written()
        assert [t["text"] for t in written["tokens"]] == list(FAKE_TOKENS[:3])
        assert [t["box"] for t in written["tokens"]] == [t["bbox"] for t in seed]
        assert written["gate"]["conf"] == 0.0

        # (c) a DELETION still sticks across the same reopen — hidden and deleted must not
        # collapse into the same thing, or every gated save would resurrect deleted boxes
        served = client.get_json("/api/image/aaaa0001")
        toks = [t for t in _ui_reload_tokens(served["seed_tokens"], served["saved"])
                if t["seed_index"] != 1]
        assert client.save("aaaa0001", _ui_save_body(toks, 0.0, n_seed=3))[0] == 200
        assert _written()["deleted_seed_indexes"] == [1]
        assert [t["seed_index"] for t in _written()["tokens"]] == [0, 2]
        served = client.get_json("/api/image/aaaa0001")
        toks = _ui_reload_tokens(served["seed_tokens"], served["saved"])
        assert client.save("aaaa0001", _ui_save_body(toks, 0.0, n_seed=3))[0] == 200
        assert [t["seed_index"] for t in _written()["tokens"]] == [0, 2]
        assert FAKE_TOKENS[1] not in [t["text"] for t in _written()["tokens"]]

        # (d) …and it still sticks when the gate is RAISED above the deleted token's
        # confidence before the next save. This is the case the old "infer deletion from
        # absence at the saved gate" rule got wrong: conf 45 hidden at c60 read as merely
        # hidden, so the deleted box came back and would have entered gt.csv.
        served = client.get_json("/api/image/aaaa0001")
        toks = _ui_reload_tokens(served["seed_tokens"], served["saved"])
        assert 1 not in [t["seed_index"] for t in toks]
        assert client.save("aaaa0001", _ui_save_body(toks, 60.0, n_seed=3))[0] == 200
        served = client.get_json("/api/image/aaaa0001")
        assert 1 not in [t["seed_index"]
                         for t in _ui_reload_tokens(served["seed_tokens"], served["saved"])]


def test_page_and_server_gate_agree() -> None:
    """The page's `visible()` and the module's `visible()` must not drift apart.

    They gate the same tokens for two different purposes — the page decides what the human
    sees (and therefore what is written), `collect()` decides what counts as "hidden" in the
    summary. A disagreement silently mis-states the seed-quality numbers.
    """
    for conf in (None, -1.0, -0.5, 0.0, 5.0, 45.0, 59.9, 60.0, 95.0, 100.0):
        for gate in (0.0, 10.0, 30.0, 60.0, 95.0, 100.0):
            mine = _ui_visible({"confidence": conf}, gate)
            assert mine == review_gt.visible(conf, gate), (conf, gate)
    # the CONF_FALLBACK case the page calls out by name: "no signal" is never "low confidence"
    assert all(review_gt.visible(-1.0, g) for g in range(0, 101))
    assert all(review_gt.visible(None, g) for g in range(0, 101))


def test_no_length_rule_can_ever_remove_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-character `L` behaves exactly like a long token at every slider position."""
    _review_dir(tmp_path, monkeypatch)
    seed = [_seed_tok("L", (5, 5, 11, 19), 95),            # 0: single char, conf 95
            _seed_tok("GRDN1234", (5, 30, 45, 54), 95),    # 1: 8 chars, SAME confidence
            _seed_tok("R", (5, 60, 11, 74), 45),           # 2: single char, conf 45
            _seed_tok("L", (5, 90, 11, 104), -1.0)]        # 3: single char, CONF_FALLBACK
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], {})
    path = review_gt.record_path("aaaa0001")
    hand_drawn = {"text": "L", "box": [200.0, 10.0, 206.0, 24.0], "label": "PHI",
                  "confidence": None, "seed_index": None}

    with _running(images) as client:
        for gate in range(0, 101):
            toks = _ui_tokens_from_seed(seed) + [dict(hand_drawn)]
            assert client.save("aaaa0001", _ui_save_body(toks, float(gate)))[0] == 200
            written = json.loads(path.read_text(encoding="utf-8"))
            present = {t["seed_index"] for t in written["tokens"]}
            # the single char is in the record exactly when its confidence clears the gate —
            # the same rule the 8-character token at the same confidence gets
            assert (0 in present) == (gate <= 95), gate
            assert (0 in present) == (1 in present), gate
            assert (2 in present) == (gate <= 45), gate
            # CONF_FALLBACK (-1) and a hand-drawn box (None) are never gated away
            assert 3 in present, gate
            assert None in present, gate
            assert hand_drawn["box"] in [t["box"] for t in written["tokens"]], gate
            assert [t["text"] for t in written["tokens"]].count("L") == (
                (1 if gate <= 95 else 0) + 2), gate  # seed 0 (maybe) + seed 3 + hand-drawn
            assert written["gate"] == {"conf": float(gate), "len": 1, "alnum": True}

        # whatever the client claims, the recorded gate has len 1 / alnum True
        assert client.save("aaaa0001", {"state": "accepted", "tokens": [],
                                        "gate": {"conf": 30, "len": 7, "alnum": False}})[0] == 200
    assert json.loads(path.read_text(encoding="utf-8"))["gate"] == {
        "conf": 30.0, "len": 1, "alnum": True}

    # no length rule exists in either artifact
    src = inspect.getsource(review_gt)
    # the only length check in the writer is the box's arity, never a token's text
    assert [a.strip() for a in re.findall(r"\blen\(\s*([^)]*)\)",
                                          inspect.getsource(review_gt.write_record))] == ["box"]
    assert not [a for a in re.findall(r"\blen\(\s*([^)]*)\)", src) if "text" in a]
    assert "len(" not in inspect.getsource(review_gt.visible)
    assert "len" not in review_gt.STRATUM_GATES
    assert all(isinstance(v, float) for v in review_gt.STRATUM_GATES.values())
    ui = UI_HTML.read_text(encoding="utf-8")
    body = re.search(r"function visible\([^)]*\)\s*\{(.*?)\n\}", ui, re.S)
    assert body, "review_ui.html no longer has a visible() gate"
    # strip comments before grepping for a length rule: prose legitimately contains "len"
    # (as in "si-len-t"), and matching that made the check fire on a comment, not on code
    code = re.sub(r"/\*.*?\*/", "", body.group(1), flags=re.S)
    assert "len" not in code
    assert code.strip().splitlines()[-1].strip() == (
        "return t.confidence == null || t.confidence < 0 || t.confidence >= g;")


# ---------------------------------------------------------------------------
# 10. per-stratum defaults
# ---------------------------------------------------------------------------


def test_per_stratum_gate_defaults_resolve_from_the_groups_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _review_dir(tmp_path, monkeypatch)
    seeds = {i: [_seed_tok(FAKE_TOKENS[0], (5, 5, 25, 29), 90)]
             for i in ("aaaa0001", "aaaa0002", "aaaa0003")}
    renders, seed_dir = _build_set(tmp_path, "a", seeds)
    groups = _groups_csv(tmp_path / "groups.csv",
                         {"aaaa0001": "mg_2d", "aaaa0002": "ct_secondary_capture"})
    images, stats = review_gt.load_pairs([f"{renders}:{seed_dir}"], review_gt.load_groups(groups))

    assert stats["no_stratum"] == 1
    with _running(images) as client:
        # the artifact the page opens with: `default_gate` in the reopen payload
        served = {i: client.get_json(f"/api/image/{i}") for i in seeds}
    assert served["aaaa0001"]["default_gate"] == 60.0
    assert served["aaaa0001"]["stratum"] == "mg_2d"
    assert served["aaaa0002"]["default_gate"] == 0.0
    assert served["aaaa0002"]["stratum"] == "ct_secondary_capture"
    assert served["aaaa0003"]["default_gate"] == review_gt.DEFAULT_GATE_CONF == 0.0
    assert served["aaaa0003"]["stratum"] == "(unknown)"

    served_states = {i: e["gate_conf"] for i, e in images.items()}
    assert served_states == {"aaaa0001": 60.0, "aaaa0002": 0.0, "aaaa0003": 0.0}

    monkeypatch.setattr(review_gt, "serve", lambda *a, **k: None)
    assert review_gt.main(["--set", f"{renders}:{seed_dir}", "--groups", str(groups)]) == 0
    out = capsys.readouterr().out
    assert "3 images from 1 pair(s)" in out
    assert "1 image_id(s) not in the groups CSV -> default gate c0 (show everything)" in out


# ---------------------------------------------------------------------------
# 11. multi-pair union, missing seeds, duplicate ids
# ---------------------------------------------------------------------------


def test_multi_pair_union_dedupe_and_hard_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _review_dir(tmp_path, monkeypatch)
    tok = [_seed_tok(FAKE_TOKENS[0], (5, 5, 25, 29), 90)]
    r1, s1 = _build_set(tmp_path, "one", {"aaaa0001": tok, "aaaa0002": tok})
    r2, s2 = _build_set(tmp_path, "two", {"aaaa0002": tok, "aaaa0003": tok})

    # (a) union: 3 distinct ids, and the ONE duplicate is sha256-identical -> deduped
    images, stats = review_gt.load_pairs([f"{r1}:{s1}", f"{r2}:{s2}"], {})
    assert sorted(images) == ["aaaa0001", "aaaa0002", "aaaa0003"]
    assert stats["dup_identical"] == 1
    assert images["aaaa0002"]["seed"] == s1 / "aaaa0002.json"  # first --set wins
    assert images["aaaa0003"]["seed"] == s2 / "aaaa0003.json"
    with _running(images) as client:
        assert {e["image_id"] for e in client.get_json("/api/index")} == set(images)

    # (b) a manifest row whose seed JSON is missing is a STARTUP failure naming that seed
    r3, s3 = _build_set(tmp_path, "three", {"aaaa0004": tok, "aaaa0005": tok},
                        skip_seed=("aaaa0005",))
    with pytest.raises(review_gt.SetupError) as exc:
        review_gt.load_pairs([f"{r3}:{s3}"], {})
    message = str(exc.value)
    assert str(s3 / "aaaa0005.json") in message
    assert "aaaa0005" in message and "seed" in message

    # (c) same image_id, DIFFERENT pixels -> refuse, naming BOTH manifests
    r4, s4 = _build_set(tmp_path, "four", {"aaaa0001": tok}, sha_override={"aaaa0001": "1" * 64})
    with pytest.raises(review_gt.SetupError) as exc:
        review_gt.load_pairs([f"{r1}:{s1}", f"{r4}:{s4}"], {})
    message = str(exc.value)
    assert str(r1 / review_gt.MANIFEST_NAME) in message
    assert str(r4 / review_gt.MANIFEST_NAME) in message
    assert "1" * 64 in message

    # (d) a missing PNG is equally fatal, and a bad --set spec never silently loads nothing
    (r3 / "aaaa0004.png").unlink()
    with pytest.raises(review_gt.SetupError, match="aaaa0004.png"):
        review_gt.load_pairs([f"{r3}:{s3}"], {})
    with pytest.raises(review_gt.SetupError, match="--set"):
        review_gt.parse_set(str(r1))


# ---------------------------------------------------------------------------
# 12. deferral round-trip and note confidentiality
# ---------------------------------------------------------------------------


def test_a_defer_records_no_deletions_so_swept_seed_boxes_come_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred image is UNDECIDED, so it retires nothing — not tokens, not deletions.

    The page treats `deleted_seed_indexes` as permanent: `restoreSaved()` drops those seed
    indexes from the working set on every reopen. Persisting them on a *deferred* record
    would therefore make "come back to this later" destructive — sweep away the junk on an
    mg_2d frame, then press D because it turns out to be unreadable, and the seed boxes are
    gone from an image nobody has ruled on, with no way back in the UI. Same argument as the
    empty `tokens` list: re-opening a deferred image re-reads the whole seed.
    """
    _review_dir(tmp_path, monkeypatch)
    seed = [_seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 90),
            _seed_tok(FAKE_TOKENS[1], (10, 50, 50, 74), 90),
            _seed_tok(FAKE_TOKENS[2], (10, 90, 70, 114), 90)]
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    groups = _groups_csv(tmp_path / "groups.csv", {"aaaa0001": "mg_2d"})
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], review_gt.load_groups(groups))

    with _running(images) as client:
        # the page posts what it swept even when the decision is a defer
        body = _ui_save_body([], 0.0, state="deferred", defer_reason="unreadable")
        body["deleted_seed_indexes"] = [0, 1, 2]
        assert client.save("aaaa0001", body)[0] == 200

        disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
        assert disk["state"] == "deferred"
        assert disk["tokens"] == []
        assert disk["deleted_seed_indexes"] == []       # the whole point

        # and the reopen really does hand every seed box back to the reviewer
        served = client.get_json("/api/image/aaaa0001")
        assert [t["seed_index"]
                for t in _ui_reload_tokens(served["seed_tokens"], served["saved"])] == [0, 1, 2]

    # a non-deferred state still records deletions — this must not have disabled the feature
    review_gt.write_record("aaaa0001", {"state": "accepted", "tokens": [],
                                        "deleted_seed_indexes": [0, 2]})
    disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
    assert disk["deleted_seed_indexes"] == [0, 2]


def test_defer_reasons_round_trip_and_the_note_never_leaves_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    _review_dir(tmp_path, monkeypatch)
    ids = [f"aaaa000{i}" for i in range(1, len(review_gt.DEFER_REASONS) + 1)]
    seeds = {i: [_seed_tok(FAKE_TOKENS[0], (5, 5, 25, 29), 90)] for i in ids}
    renders, seed_dir = _build_set(tmp_path, "a", seeds)
    groups = _groups_csv(tmp_path / "groups.csv", dict.fromkeys(ids, "ct_axial"))
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"], review_gt.load_groups(groups))
    assert review_gt.DEFER_REASONS == ("unreadable", "ambiguous_token",
                                       "possible_non_blank_control", "needs_cal")

    capfd.readouterr()
    with _running(images) as client:
        for image_id, reason in zip(ids, review_gt.DEFER_REASONS, strict=True):
            extra = {"defer_reason": reason}
            if reason == "needs_cal":
                extra["note"] = FAKE_NOTE
            body = _ui_save_body([], 0.0, state="deferred", **extra)
            assert client.save(image_id, body)[0] == 200
            disk = json.loads(review_gt.record_path(image_id).read_text(encoding="utf-8"))
            assert disk["state"] == "deferred"
            assert disk["defer_reason"] == reason
            reopened = client.get_json(f"/api/image/{image_id}")["saved"]
            assert reopened["state"] == "deferred"
            assert reopened["defer_reason"] == reason

        # the note reproduces verbatim through the reopen endpoint...
        assert client.get_json(f"/api/image/{ids[3]}")["saved"]["note"] == FAKE_NOTE
        assert json.loads(
            review_gt.record_path(ids[3]).read_text(encoding="utf-8"))["note"] == FAKE_NOTE

        # ...and a deferral with no reason, or a reason outside the enum, is REFUSED at write
        for bad in ({"state": "deferred", "tokens": []},
                    {"state": "deferred", "tokens": [], "defer_reason": "because"},
                    {"state": "deferred", "tokens": [], "defer_reason": ""},
                    {"state": "accepted", "tokens": [], "defer_reason": "needs_cal"},
                    {"state": "not_a_state", "tokens": []}):
            code, body = client.save("aaaa0001", bad)
            assert code == 400, bad
            assert body == b"request failed"  # fixed string: no traceback, no body echo
        # the earlier good record is still exactly what it was
        disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
        assert disk["defer_reason"] == "unreadable"

    with pytest.raises(ValueError, match="defer_reason"):
        review_gt.write_record("aaaa0001", {"state": "deferred", "tokens": []})

    review_gt.print_summary(images)
    captured = capfd.readouterr()
    assert "needs_cal 4" not in captured.out
    assert "unreadable 1" in captured.out and "needs_cal 1" in captured.out
    for secret in (FAKE_NOTE, "CMFN-00421", "overlaps the ruler", *FAKE_TOKENS):
        assert secret not in captured.out + captured.err, secret


# ---------------------------------------------------------------------------
# 13. provenance fields on the written record
# ---------------------------------------------------------------------------


def test_round_defaults_to_one_and_timestamp_is_iso8601_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _review_dir(tmp_path, monkeypatch)
    payload = {"state": "accepted", "gate": {"conf": 0},
               "tokens": [{"text": FAKE_TOKENS[0], "box": [1, 2, 3, 4], "label": "PHI",
                           "confidence": 90.0, "seed_index": 0}]}
    before = datetime.now(review_gt.UTC).replace(microsecond=0)
    review_gt.write_record("aaaa0001", payload)
    after = datetime.now(review_gt.UTC)

    disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
    assert disk["round"] == 1 and isinstance(disk["round"], int)
    stamp = disk["timestamp"]
    assert stamp.endswith("+00:00"), stamp
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.tzinfo is not None
    assert before <= parsed <= after
    assert "." not in stamp  # timespec="seconds"

    # an explicit round is honoured; a falsy one falls back to round 1
    review_gt.write_record("aaaa0002", {**payload, "round": 2})
    assert json.loads(review_gt.record_path("aaaa0002").read_text(encoding="utf-8"))["round"] == 2
    review_gt.write_record("aaaa0003", {**payload, "round": 0})
    assert json.loads(review_gt.record_path("aaaa0003").read_text(encoding="utf-8"))["round"] == 1


# ---------------------------------------------------------------------------
# 14-18: the blind spots a mutation audit found — each of these passes the
# suite above if the behaviour it names is deleted from review_gt.py
# ---------------------------------------------------------------------------


def test_review_dir_is_anchored_to_the_repo_not_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PHI records must land where .gitignore and the Read-deny rules actually apply.

    Every other test monkeypatches REVIEW_DIR, so none of them would notice it going back to
    a CWD-relative path — and started from ~ or /tmp that writes PHI outside the repo, where
    nothing is ignored and nothing is denied (CLAUDE.md §6.6).
    """
    monkeypatch.chdir(tmp_path)
    assert review_gt.REVIEW_DIR.is_absolute()
    assert review_gt.REVIEW_DIR == REPO_ROOT / "ground_truth" / "review"
    assert review_gt.record_path("aaaa0001").parent == REPO_ROOT / "ground_truth" / "review"
    src = (REPO_ROOT / "ground_truth" / "review_gt.py").read_text(encoding="utf-8")
    assert 'REVIEW_DIR = Path("ground_truth' not in src


def test_duplicate_image_id_with_different_seed_provenance_is_a_hard_error(
    tmp_path: Path
) -> None:
    """Same pixels is not enough — two seeds from different Tesseract builds must not merge.

    Rule #9: a seeder version bump invalidates results the way a gt.csv change does. Keeping
    the first --set's seed silently would decide that by argument order.
    """
    seed = [_seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 90)]
    r1, s1 = _build_set(tmp_path, "p1", {"aaaa0001": seed},
                        provenance={"tesseract_version": "5.3.0"})
    sha = dict(
        line.split(",")[:1] + line.split(",")[4:5]
        for line in (r1 / review_gt.MANIFEST_NAME).read_text().splitlines()[1:]
    )
    r2, s2 = _build_set(tmp_path, "p2", {"aaaa0001": seed}, sha_override=sha,
                        provenance={"tesseract_version": "5.4.1"})
    with pytest.raises(review_gt.SetupError) as exc:
        review_gt.load_pairs([f"{r1}:{s1}", f"{r2}:{s2}"], {})
    assert "provenance" in str(exc.value)
    assert str(s1 / "aaaa0001.json") in str(exc.value)
    assert str(s2 / "aaaa0001.json") in str(exc.value)
    # identical provenance still dedupes rather than erroring
    r3, s3 = _build_set(tmp_path, "p3", {"aaaa0001": seed}, sha_override=sha)
    r4, s4 = _build_set(tmp_path, "p4", {"aaaa0001": seed}, sha_override=sha)
    _, stats = review_gt.load_pairs([f"{r3}:{s3}", f"{r4}:{s4}"], {})
    assert stats["dup_identical"] == 1


def test_write_record_rejects_malformed_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """label / empty text / seed_index validation, none of which any other test exercises.

    A duplicate seed_index makes collect() count one seed token twice; a NEGATIVE one indexes
    from the end of the seed list in Python and so compares the human's text against the wrong
    token; an empty text cannot become a gt.csv row at all.
    """
    _review_dir(tmp_path, monkeypatch)
    good = {"text": FAKE_TOKENS[0], "box": [1.0, 2.0, 3.0, 4.0], "label": "PHI",
            "confidence": 90.0, "seed_index": 0}
    for bad in (
        {**good, "label": "OTHER"},      # binary PHI/KEEP — OTHER was removed on purpose
        {**good, "label": "phi"},
        {**good, "text": "   "},
        {**good, "seed_index": -1},
    ):
        with pytest.raises(ValueError):
            review_gt.write_record("aaaa0001", {"state": "accepted", "tokens": [bad]})
    with pytest.raises(ValueError):                       # duplicate seed_index
        review_gt.write_record("aaaa0001", {"state": "accepted",
                                            "tokens": [good, {**good, "text": FAKE_TOKENS[1]}]})
    assert not review_gt.record_path("aaaa0001").exists()  # nothing written on any rejection
    # a single-character token is NOT malformed — there is no length floor anywhere
    rec = review_gt.write_record("aaaa0001",
                                 {"state": "accepted", "tokens": [{**good, "text": "L"}]})
    assert [t["text"] for t in rec["tokens"]] == ["L"]


def test_confidence_is_recorded_so_a_reopened_token_stays_gateable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping `confidence` from the record would make every reopened token ungatable.

    It would also be invisible to every progress/geometry assertion — the record still looks
    complete, and the slider silently stops working after the first save.
    """
    _review_dir(tmp_path, monkeypatch)
    rec = review_gt.write_record("aaaa0001", {"state": "accepted", "tokens": [
        {"text": FAKE_TOKENS[0], "box": [1.0, 2.0, 3.0, 4.0], "label": "PHI",
         "confidence": 45.0, "seed_index": 0},
        {"text": FAKE_TOKENS[1], "box": [5.0, 6.0, 7.0, 8.0], "label": "KEEP",
         "confidence": None, "seed_index": None},
    ]})
    disk = json.loads(review_gt.record_path("aaaa0001").read_text(encoding="utf-8"))
    assert [t["confidence"] for t in disk["tokens"]] == [45.0, None]
    assert rec["tokens"][0]["confidence"] == 45.0
    # and the recorded value drives the gate the same way the seed's does
    assert review_gt.visible(disk["tokens"][0]["confidence"], 30.0) is True
    assert review_gt.visible(disk["tokens"][0]["confidence"], 60.0) is False
    assert review_gt.visible(disk["tokens"][1]["confidence"], 100.0) is True


def test_hidden_count_uses_the_records_gate_not_the_stratum_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`hidden` must be computed from the gate the record was SAVED at.

    Using the stratum default instead reads identically whenever the two agree, which is the
    common case — so the number would look right while being unable to detect the situation it
    exists to report: a stratum whose default is hiding real text.
    """
    _review_dir(tmp_path, monkeypatch)
    seed = [_seed_tok(FAKE_TOKENS[0], (10, 10, 30, 34), 5),
            _seed_tok(FAKE_TOKENS[1], (10, 50, 50, 74), 45),
            _seed_tok(FAKE_TOKENS[2], (10, 90, 70, 114), 95)]
    renders, seed_dir = _build_set(tmp_path, "a", {"aaaa0001": seed})
    groups = _groups_csv(tmp_path / "g.csv", {"aaaa0001": "mg_2d"})   # default c60
    images, _ = review_gt.load_pairs([f"{renders}:{seed_dir}"],
                                     review_gt.load_groups(groups))
    assert images["aaaa0001"]["gate_conf"] == 60.0
    # saved at c10 — far from the stratum default, and only the conf-5 token is hidden
    review_gt.write_record("aaaa0001", {
        "state": "accepted", "gate": {"conf": 10.0},
        "tokens": [{"text": t["text"], "box": t["bbox"], "label": "PHI",
                    "confidence": t["confidence"], "seed_index": k}
                   for k, t in enumerate(seed) if t["confidence"] >= 10.0],
    })
    stats = review_gt.collect(images)
    assert stats["mg_2d"]["hidden"] == 1        # at the stratum default c60 it would be 2
    assert stats["mg_2d"]["non_default_gate"] == 1
    assert stats["ALL"]["hidden"] == 1
