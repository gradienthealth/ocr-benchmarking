"""Phase 13i — tests for the experiment driver. 🟢 SYNTHETIC FIXTURES ONLY.

Everything here runs on `tests/synthetic.py`'s fabricated images and fake `CMFN`/`GRDN`-style
tokens. No render, no `gt.csv` content, no manifest row, and — the one that matters most —
**no live Gemini call, ever**. The Gemini arm is exercised through an injected stub transport
that returns a `GeminiReply` built in this file; nothing in this module can reach the network.

The two tests that touch the real `ground_truth/gt.csv` compute a **sha256** of it and read the
PHI-free `image_id,has_text` roster. They never read a row, never load a token, and skip
themselves when the (gitignored) artifact is absent.

What is pinned here, in the order the driver checks it:
  - a wrong `gt.csv` hash aborts, and so does a wrong scored-set (scope) hash;
  - `--all-arms` never selects the BAA-gated arm, and `--synthetic` refuses it outright;
  - `gemini:real` without `OCR_BAA_CLEARED_GEMINI` aborts, as does a missing Vertex project or
    location — before any crop is encoded;
  - `ServedVersionMismatch` propagates out of the driver instead of being swallowed;
  - accuracy is unreachable without a negative-control floor;
  - a run that dies halfway emits no table and records `INCOMPLETE`;
  - arm selection produces exactly the tables selected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRIVER_PATH = REPO_ROOT / "scripts" / "run_experiment.py"
if not DRIVER_PATH.is_file():
    # ⚠️ `scripts/` is gitignored (`.gitignore:139`, "internal detail Gradient wouldn't want
    # public"), so the 13i driver is NOT tracked while this test file IS. A fresh clone
    # therefore has the tests and not the code they cover. This skip exists to say so out
    # loud rather than let the suite report green over an absent driver — it is a flag, not
    # a pass. Resolving it is a repo-policy decision (track the driver, or don't track its
    # tests); nothing here quietly edits .gitignore to paper over it.
    pytest.skip(
        f"{DRIVER_PATH} is absent — scripts/ is gitignored, so the 13i driver is untracked. "
        "The whole 13i test suite is therefore NOT running.",
        allow_module_level=True,
    )

from scripts import run_experiment as drv  # noqa: E402

DRIVER_SRC = DRIVER_PATH.read_text(encoding="utf-8")


# --- helpers --------------------------------------------------------------------------------


def synth(tmp_path: Path) -> dict:
    """The on-disk synthetic scene the reader arms are exercised against."""
    return drv._synthetic_setup(tmp_path / "scene")


def control_boxes(setup: dict, n: int = 2) -> dict[str, list]:
    return {
        img.id: drv.control_boxes_for(img.w, img.h, n) for img in setup["blank_images"]
    }


class StubTransport:
    """A Gemini transport that never touches the network. Records what it was handed."""

    def __init__(self, versions: list[str | None], text: str = "CMFN-0042") -> None:
        self.versions = list(versions)
        self.text = text
        self.calls = 0

    def __call__(self, call):
        from harness.readers.read_gemini import GeminiReply

        version = self.versions[min(self.calls, len(self.versions) - 1)]
        self.calls += 1
        return GeminiReply(
            text=self.text,
            model_version=version,
            prompt_tokens=100,
            output_tokens=8,
            thought_tokens=64,
            blocked=False,
            finish_reason="STOP",
        )


# --- the hash gate --------------------------------------------------------------------------


def test_wrong_gt_hash_aborts(tmp_path: Path):
    bogus = tmp_path / "gt.csv"
    bogus.write_text("image_id,token_text\nsynth-000,CMFN\n", encoding="utf-8")
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.verify_frozen_gt(bogus)
    assert "HASH MISMATCH" in str(exc.value)
    assert drv.GT_V1_SHA256 in str(exc.value)


def test_missing_gt_aborts(tmp_path: Path):
    with pytest.raises(drv.ExperimentAbort):
        drv.verify_frozen_gt(tmp_path / "nope.csv")


def test_wrong_scored_set_aborts():
    """The content hash cannot catch a run over a subset — this is what does."""
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.verify_scored_set(["synth-000", "synth-001"])
    assert "SCORED-SET HASH MISMATCH" in str(exc.value)


def test_verify_only_aborts_on_a_deliberately_wrong_hash(tmp_path: Path):
    presence = tmp_path / "presence.csv"
    presence.write_text("image_id,has_text\nsynth-000,1\n", encoding="utf-8")
    bogus = tmp_path / "gt.csv"
    bogus.write_text("image_id\nsynth-000\n", encoding="utf-8")
    with pytest.raises(drv.ExperimentAbort):
        drv.main(["--verify-only", "--gt", str(bogus), "--text-presence", str(presence)])


@pytest.mark.skipif(not drv.FROZEN_GT.is_file(), reason="gt.csv is gitignored; not in this tree")
def test_verify_only_on_the_real_gt_v1_prints_the_identity_stamp(capsys):
    """Hashes the frozen artifact and reads the PHI-free roster. No row, no token, no pixel."""
    presence = REPO_ROOT / "ground_truth" / "text_presence_v2.csv"
    if not presence.is_file():
        pytest.skip("text_presence_v2.csv absent")
    assert drv.main(["--verify-only", "--text-presence", str(presence)]) == 0
    out = capsys.readouterr().out
    assert drv.GT_V1_SHA256 in out
    assert drv.GT_V1_SET_SHA256 in out
    assert "gt_v1" in out
    assert "VERIFIED" in out


# --- arm selection --------------------------------------------------------------------------


def test_all_arms_never_selects_the_gated_arm():
    args = drv.parse_args(["--all-arms", "--out", "x"])
    selected = drv.resolve_arms(args)
    assert set(selected) & set(drv.GATED_ARMS) == set()
    assert set(selected) == set(drv.END_TO_END_ARMS) | set(drv.LOCAL_READER_ARMS)


def test_gated_arm_requires_being_named():
    args = drv.parse_args(["--arm", "gemini:real", "--out", "x"])
    assert drv.resolve_arms(args) == ["gemini:real"]


def test_synthetic_mode_refuses_the_gated_arm():
    args = drv.parse_args(["--synthetic", "--arm", "gemini:real", "--out", "x"])
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.resolve_arms(args)
    assert "BAA-gated" in str(exc.value)


def test_unknown_arm_aborts():
    args = drv.parse_args(["--arm", "surya:stock", "--out", "x"])
    with pytest.raises(drv.ExperimentAbort):
        drv.resolve_arms(args)


def test_no_arm_selected_aborts():
    with pytest.raises(drv.ExperimentAbort):
        drv.resolve_arms(drv.parse_args(["--out", "x"]))


# --- the driver never opens the gate --------------------------------------------------------


def test_driver_never_sets_the_baa_variable():
    """A grep, deliberately: the guarantee is about the source, not about one code path."""
    forbidden = ("os.environ[", "os.putenv", "environ.setdefault", "setenv(", "environ.update")
    for pattern in forbidden:
        assert pattern not in DRIVER_SRC, f"driver must never set the environment: {pattern!r}"
    # Every mention of the process environment in this file is a READ. Two of them: the BAA
    # gate and the Vertex config, both in `preflight_gemini`.
    assert DRIVER_SRC.count("os.environ") == DRIVER_SRC.count("os.environ.get(") == 2
    assert DRIVER_SRC.count("os.environ.get(BAA_ENV_VAR)") == 1


def test_no_env_file_or_makefile_sets_the_gate():
    for name in (".env", "Makefile", "makefile"):
        assert not (REPO_ROOT / name).is_file() or drv.BAA_ENV_VAR not in (
            REPO_ROOT / name
        ).read_text(encoding="utf-8", errors="replace")


# --- the Gemini preflight -------------------------------------------------------------------


def test_gemini_without_clearance_aborts(monkeypatch):
    monkeypatch.delenv(drv.BAA_ENV_VAR, raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.preflight_gemini()
    assert drv.BAA_ENV_VAR in str(exc.value)
    assert "D-12.1" in str(exc.value)


def test_gemini_gate_is_not_truthiness(monkeypatch):
    """'0' and 'no' are the values someone types to keep the gate SHUT."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    for value in ("0", "no", "false", "", "true"):
        monkeypatch.setenv(drv.BAA_ENV_VAR, value)
        with pytest.raises(drv.ExperimentAbort):
            drv.preflight_gemini()


@pytest.mark.parametrize("missing", ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"])
def test_gemini_without_vertex_config_aborts(monkeypatch, missing):
    monkeypatch.setenv(drv.BAA_ENV_VAR, "1")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.preflight_gemini()
    assert "Vertex" in str(exc.value) or "VERTEX" in str(exc.value)
    assert "AI Studio" in str(exc.value)


@pytest.mark.parametrize("arm", sorted(drv.GENERATIVE_ARMS))
def test_a_generative_arm_with_no_blank_frames_aborts(arm):
    """Not Gemini-only: no blank frames means no invention floor, for any generative arm."""
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.preflight_generative([arm], set())
    assert "floor" in str(exc.value)
    # a non-generative arm has no such requirement
    drv.preflight_generative(["doctr:stock", "reader:svtrv2"], set())


def test_build_reader_gemini_raises_without_clearance(monkeypatch):
    """Even bypassing the preflight, the reader's own gate shuts the door."""
    from harness.readers.read_gemini import BAAGateError

    monkeypatch.delenv(drv.BAA_ENV_VAR, raising=False)
    with pytest.raises(BAAGateError):
        drv.build_reader("gemini:real")


# --- the Gemini arm, on synthetic pixels with a stub transport ------------------------------


def test_gemini_arm_records_served_version_and_measured_cost(tmp_path, monkeypatch):
    from harness.readers.read_gemini import CropSource, GeminiReader

    monkeypatch.setenv(drv.BAA_ENV_VAR, "1")
    setup = synth(tmp_path)
    transport = StubTransport(["gemini-2.5-pro"])
    reader = GeminiReader(source=CropSource.REAL, transport=transport)

    result = drv.run_reader_arm(
        "gemini:real",
        setup["images"],
        setup["blank_images"],
        setup["gt"],
        setup["allowlist"],
        control_boxes(setup),
        tmp_path / "arm",
        setup["identity"],
        reader=reader,
    )
    assert result["served_model_version"] == "gemini-2.5-pro"
    assert result["usage_summary"]["thought_tokens"] > 0
    # The floor ran first and travelled into the accuracy artifact.
    assert result["negative_control_first"]["n_control_crops"] == 2
    assert result["floor"]["phase"] == "negative_control"

    # Cost SCOPE: usage_summary() is cumulative over both passes, so the table's column must be
    # the accuracy pass alone or the hosted arm silently carries its negative control's bill
    # into a cost-per-arm comparison against local readers that have no such line.
    assert result["measured_cost_usd_floor"] > 0
    assert result["measured_cost_usd"] > 0
    assert result["measured_cost_usd"] == pytest.approx(
        result["measured_cost_usd_total"] - result["measured_cost_usd_floor"]
    )
    assert result["measured_cost_usd"] < result["measured_cost_usd_total"]
    # and the column really is the scope-matched number
    cost, basis = drv._cost_cells(result)
    assert basis.startswith("measured")
    assert cost == f"${result['measured_cost_usd']:.6f}"


def test_served_version_mismatch_propagates(tmp_path, monkeypatch):
    """A revision change mid-run must kill the run — not be caught, retried, or downgraded."""
    from harness.readers.read_gemini import CropSource, GeminiReader, ServedVersionMismatch

    monkeypatch.setenv(drv.BAA_ENV_VAR, "1")
    setup = synth(tmp_path)
    transport = StubTransport(["gemini-2.5-pro", "gemini-2.5-pro-002"])
    reader = GeminiReader(source=CropSource.REAL, transport=transport)

    with pytest.raises(ServedVersionMismatch):
        drv.run_reader_arm(
            "gemini:real",
            setup["images"],
            setup["blank_images"],
            setup["gt"],
            setup["allowlist"],
            control_boxes(setup),
            tmp_path / "arm",
            setup["identity"],
            reader=reader,
        )


def test_gemini_arm_writes_no_model_output(tmp_path, monkeypatch):
    """The predicted string on a real crop IS the patient identifier. Nothing persists it."""
    from harness.readers.read_gemini import CropSource, GeminiReader

    monkeypatch.setenv(drv.BAA_ENV_VAR, "1")
    setup = synth(tmp_path)
    secret = "ZZTOPSECRET42"
    reader = GeminiReader(
        source=CropSource.REAL, transport=StubTransport(["gemini-2.5-pro"], text=secret)
    )
    drv.run_reader_arm(
        "gemini:real",
        setup["images"],
        setup["blank_images"],
        setup["gt"],
        setup["allowlist"],
        control_boxes(setup),
        tmp_path / "arm",
        setup["identity"],
        reader=reader,
    )
    for path in (tmp_path / "arm").rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(encoding="utf-8", errors="replace")
            assert "raw_response" not in path.read_text(encoding="utf-8", errors="replace")


def test_reader_gate_is_rechecked_on_every_crop(tmp_path, monkeypatch):
    """Clearance revoked mid-run stops the next crop, not just the first."""
    from harness.readers.read_gemini import BAAGateError, CropSource, GeminiReader

    monkeypatch.setenv(drv.BAA_ENV_VAR, "1")
    setup = synth(tmp_path)
    reader = GeminiReader(source=CropSource.REAL, transport=StubTransport(["gemini-2.5-pro"]))
    monkeypatch.delenv(drv.BAA_ENV_VAR, raising=False)
    with pytest.raises(BAAGateError):
        drv.run_reader_arm(
            "gemini:real",
            setup["images"],
            setup["blank_images"],
            setup["gt"],
            setup["allowlist"],
            control_boxes(setup),
            tmp_path / "arm",
            setup["identity"],
            reader=reader,
        )


# --- floor before accuracy ------------------------------------------------------------------


def test_accuracy_is_refused_without_a_floor(tmp_path):
    setup = synth(tmp_path)
    reader = drv._synthetic_reader("synthetic:reader-good")
    for bad_floor in ({}, {"phase": "accuracy"}, {"phase": None}):
        with pytest.raises(drv.ExperimentAbort) as exc:
            drv.reader_accuracy(
                bad_floor,
                "synthetic:reader-good",
                reader,
                setup["images"],
                setup["gt"],
                setup["allowlist"],
                tmp_path / "arm",
                setup["identity"],
            )
        assert "negative-control" in str(exc.value)


def test_floor_runs_before_accuracy_and_is_carried_into_the_table(tmp_path):
    setup = synth(tmp_path)
    result = drv.run_reader_arm(
        "synthetic:reader-inventive",
        setup["images"],
        setup["blank_images"],
        setup["gt"],
        setup["allowlist"],
        control_boxes(setup, n=3),
        tmp_path / "arm",
        setup["identity"],
        reader=drv._synthetic_reader("synthetic:reader-inventive"),
    )
    assert (tmp_path / "arm" / "floor.json").is_file()
    assert result["floor"]["inventions"] == 3  # the inventive reader answers every control box
    table = drv.table_reader([result])
    assert "Control crops" in table and "Inventions" in table
    assert "ORACLE" in table


@pytest.mark.parametrize("n", [0, -1, len(drv.POSITIONS) + 1])
def test_bad_boxes_per_image_is_refused(tmp_path, n):
    """0 measures no floor at all; -1 slices the position grid from the wrong end."""
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.main([
            "--synthetic", "--arm", "synthetic:reader-good",
            "--boxes-per-image", str(n), "--out", str(tmp_path / "run"),
        ])
    assert "--boxes-per-image" in str(exc.value)


def test_a_floor_that_sent_no_crops_is_not_a_floor(tmp_path):
    """The structural backstop: every route to a zero-crop floor is refused, not just the flag.

    A floor step that ran and sent zero crops reports `inventions: 0` in exactly the shape a
    measured zero has. Under Table C's prose ("measured before any accuracy number") that reads
    as "it never invents" when it means "we never asked" — the ct_scout mistake (CLAUDE.md §8).
    """
    setup = synth(tmp_path)
    reader = drv._synthetic_reader("synthetic:reader-good")
    floor = drv.reader_floor(
        "synthetic:reader-good", reader, setup["blank_images"], {},  # no boxes for any frame
        tmp_path / "arm", setup["identity"],
    )
    assert floor["n_control_crops"] == 0 and floor["inventions"] == 0
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.reader_accuracy(
            floor, "synthetic:reader-good", reader, setup["images"], setup["gt"],
            setup["allowlist"], tmp_path / "arm", setup["identity"],
        )
    assert "never measured" in str(exc.value)
    assert "absence, not a zero" in str(exc.value)


# --- partial runs ---------------------------------------------------------------------------


def test_a_run_that_dies_halfway_emits_no_table(tmp_path, monkeypatch):
    out = tmp_path / "run"
    real_runner = drv._synthetic_runner

    def explode(arm, gt):
        if arm.endswith("noisy"):
            raise RuntimeError("engine fell over")
        return real_runner(arm, gt)

    monkeypatch.setattr(drv, "_synthetic_runner", explode)
    with pytest.raises(RuntimeError):
        drv.main([
            "--synthetic",
            "--arm", "synthetic:e2e-good",
            "--arm", "synthetic:e2e-noisy",
            "--out", str(out),
        ])

    assert not (out / "TABLES.md").exists(), "a partial run must emit no table"
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "INCOMPLETE"
    assert status["tables_written"] is False
    assert status["failed_arm"] == "synthetic:e2e-noisy"
    assert status["arms_pending"] == ["synthetic:e2e-noisy"]
    assert status["failure_type"] == "RuntimeError"


def test_finish_refuses_when_an_arm_produced_no_result(tmp_path):
    ledger = drv.RunLedger(tmp_path, ["a", "b"], {"dataset_id": "x"})
    ledger.start()
    ledger.complete_arm("a")
    with pytest.raises(drv.ExperimentAbort) as exc:
        ledger.finish()
    assert "no result" in str(exc.value)


def test_a_dirty_out_dir_is_refused(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "TABLES.md").write_text("stale", encoding="utf-8")
    with pytest.raises(drv.ExperimentAbort) as exc:
        drv.prepare_out_dir(out)
    assert "not empty" in str(exc.value)
    assert (out / "TABLES.md").read_text(encoding="utf-8") == "stale"  # nothing deleted


# --- arm selection produces exactly the tables selected -------------------------------------


def test_end_to_end_only_selection_yields_no_reader_rows(tmp_path):
    out = tmp_path / "run"
    assert drv.main(["--synthetic", "--arm", "synthetic:e2e-good", "--out", str(out)]) == 0
    tables = (out / "TABLES.md").read_text(encoding="utf-8")
    assert "synthetic:e2e-good" in tables
    assert "synthetic:e2e-noisy" not in tables
    assert "_(no reader arm ran)_" in tables
    assert "Table A" in tables and "Table B" in tables and "Table C" in tables


def test_reader_only_selection_yields_no_end_to_end_rows(tmp_path):
    out = tmp_path / "run"
    assert drv.main(["--synthetic", "--arm", "synthetic:reader-good", "--out", str(out)]) == 0
    tables = (out / "TABLES.md").read_text(encoding="utf-8")
    assert "_(no end-to-end arm ran)_" in tables
    assert "synthetic:reader-good" in tables


def test_synthetic_dry_run_produces_three_stamped_tables(tmp_path):
    out = tmp_path / "run"
    assert drv.main(["--synthetic", "--all-arms", "--out", str(out)]) == 0
    tables = (out / "TABLES.md").read_text(encoding="utf-8")

    assert "Table A — end-to-end" in tables
    assert "Table B — detector-only" in tables
    assert "Table C — reader arm" in tables
    # the identity stamp, on the tables themselves
    for field in ("Artifact identity", "gt.csv sha256", "Scored-set sha256", "Images scored"):
        assert field in tables
    # a synthetic run must never wear gt_v1's identity
    assert "SYNTHETIC-SCENE (not gt_v1)" in tables
    assert drv.GT_V1_SHA256 not in tables

    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE" and status["tables_written"] is True
    assert status["arms_pending"] == []


def test_written_artifacts_carry_no_token_text(tmp_path):
    """The scored strings are fake, and even so they must not reach a written artifact."""
    from tests import synthetic

    out = tmp_path / "run"
    drv.main(["--synthetic", "--all-arms", "--out", str(out)])
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in synthetic.FAKE_TOKENS + (synthetic.NOT_ALLOWLISTED,):
                assert token not in text, f"{path.name} leaked a token string"
            assert "raw_response" not in text


# --- what this driver must not have changed --------------------------------------------------


def test_driver_adds_no_scoring_logic():
    """It wires; it does not score. No IoU, no normalize, no metric arithmetic lives here."""
    for banned in ("def score(", "def match(", "def normalize(", "iou =", "def aggregate("):
        assert banned not in DRIVER_SRC
