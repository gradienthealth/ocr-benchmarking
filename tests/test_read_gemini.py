"""Phase 13g — tests for the BAA-gated Gemini reader. 100% synthetic, 100% offline.

Every image here is drawn by `tests/synthetic.py` from obviously-fake tokens ("CMFN",
"GRDN1234", "ACC-0001"), and every model reply comes from a spy transport that is a plain
Python function. No network call is made by this file, no credential is read, and no real
render is touched.

**The first test in this file is the point of the whole prompt**: a crop declared REAL
must raise before anything can reach the network, unless a human has set
`OCR_BAA_CLEARED_GEMINI=1` in the environment. Everything after it exists to make sure
that raise cannot be routed around — by a fixture, an injected transport, an error
handler, a retry, or a mislabelled crop.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
from PIL import Image

from harness.contract import GTToken
from harness.cost import PRICES, estimate_cost
from harness.harness import ImageRef
from harness.reading import crop_for_reading, read_image
from harness.runners.read_gemini import (
    BAA_ENV_VAR,
    PINNED_MODEL,
    BAAGateError,
    CropSource,
    GeminiCall,
    GeminiReader,
    GeminiReply,
    ServedVersionMismatch,
    SyntheticProvenanceError,
    crop_digest,
)
from tests import synthetic

MODULE = Path(__file__).resolve().parents[1] / "harness" / "runners" / "read_gemini.py"


class SpyTransport:
    """Stands in for the network. Records every call it is handed; returns canned text.

    A spy rather than a mock: the assertions that matter are about calls that must NEVER
    happen, and `calls == []` is the only way to state that.
    """

    def __init__(self, replies: list[str] | None = None, *, model_version: str | None = None):
        self.replies = list(replies or [])
        self.calls: list[GeminiCall] = []
        self.model_version = model_version

    def __call__(self, call: GeminiCall) -> GeminiReply:
        self.calls.append(call)
        text = self.replies[len(self.calls) - 1] if len(self.calls) <= len(self.replies) else ""
        return GeminiReply(
            text=text,
            model_version=self.model_version if self.model_version is not None else call.model,
            prompt_tokens=270,
            output_tokens=6,
            thought_tokens=128,
            blocked=False,
        )


def _synthetic_crops(tokens=("CMFN", "GRDN1234")) -> list[tuple[str, Image.Image]]:
    """Fake scene -> the exact crops a reader would be handed, via the pinned pipeline."""
    image, gt = synthetic.make_synthetic_image(tokens)
    return [(t.token_text, crop_for_reading(image, t.bbox)) for t in gt]


def _synthetic_reader(transport, tokens=("CMFN", "GRDN1234")) -> tuple[GeminiReader, list]:
    crops = _synthetic_crops(tokens)
    reader = GeminiReader(
        source=CropSource.SYNTHETIC,
        synthetic_digests=frozenset(crop_digest(c) for _, c in crops),
        transport=transport,
    )
    return reader, crops


# --- THE GATE ---------------------------------------------------------------------------


def test_a_real_crop_cannot_be_read_without_the_human_set_env_var(monkeypatch):
    """THE test this prompt exists for: no flag, no egress — construction already refuses.

    `source=CropSource.REAL` is the only declaration under which a render off the scanner
    can be handed to this reader. Absent `OCR_BAA_CLEARED_GEMINI=1` it raises here, at
    construction, before a client, a credential, or a crop is anywhere near it.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"])

    with pytest.raises(BAAGateError) as excinfo:
        GeminiReader(source=CropSource.REAL, transport=spy)

    assert BAA_ENV_VAR in str(excinfo.value)
    assert "CLAUDE.md" in str(excinfo.value) and "§4" in str(excinfo.value)
    assert spy.calls == []  # nothing was sent, by construction


def test_the_gate_is_rechecked_at_every_read_not_just_at_construction(monkeypatch):
    """A reader built while the gate was open stops the moment the gate closes.

    Construction-time-only checking would leave a long-lived reader running past a
    revoked clearance — and the object outlives the check by design (one construction,
    thousands of crops).
    """
    monkeypatch.setenv(BAA_ENV_VAR, "1")
    spy = SpyTransport(["CMFN", "CMFN"])
    reader = GeminiReader(source=CropSource.REAL, transport=spy)
    _, crop = _synthetic_crops(("CMFN",))[0]

    assert reader.read(crop) == "CMFN"  # gate open: one call goes out
    monkeypatch.delenv(BAA_ENV_VAR)

    with pytest.raises(BAAGateError):
        reader.read(crop)

    assert len(spy.calls) == 1  # the second read never reached the transport


@pytest.mark.parametrize("value", ["", "0", "no", "true", "yes", "TRUE", " 1", "1 ", "01"])
def test_only_the_exact_string_1_opens_the_gate(monkeypatch, value):
    """Truthiness is not clearance.

    `if os.environ.get(VAR):` would open the gate on the string "0" and on "no" — the two
    values someone would most plausibly type to keep it SHUT.
    """
    monkeypatch.setenv(BAA_ENV_VAR, value)
    with pytest.raises(BAAGateError):
        GeminiReader(source=CropSource.REAL, transport=SpyTransport())


def test_source_must_be_declared_explicitly(monkeypatch):
    """There is no default `source`: nobody gets network access by forgetting a keyword."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    with pytest.raises(TypeError):
        GeminiReader(transport=SpyTransport())  # type: ignore[call-arg]


def test_an_injected_transport_does_not_bypass_the_gate(monkeypatch):
    """The gate is checked in `read()`, above the transport — not inside the live client.

    If the check lived in the live transport, passing any other callable would walk
    straight past it. Tests inject transports constantly; that is exactly the hole.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    with pytest.raises(BAAGateError):
        GeminiReader(source=CropSource.REAL, transport=lambda call: GeminiReply.empty())


def test_a_real_crop_cannot_be_laundered_through_the_synthetic_path(monkeypatch):
    """The synthetic path takes pre-registered pixels ONLY — not a caller's promise.

    Declaring `SYNTHETIC` is what buys "no flag needed", so the declaration cannot be the
    only thing standing between a render and the network. The reader hashes each crop and
    refuses any crop it was not given the digest of up front, so a frame off the scanner
    fails on its pixels, not on the caller's honesty.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"])
    reader, _ = _synthetic_reader(spy)

    # Stands in for a real render's crop: never registered, so its digest is unknown.
    unregistered = Image.new("RGB", (120, 48), color=(17, 17, 17))

    with pytest.raises(SyntheticProvenanceError) as excinfo:
        reader.read(unregistered)

    assert spy.calls == []
    assert BAA_ENV_VAR in str(excinfo.value)  # tells the caller which door is the real one


def test_the_synthetic_path_needs_no_flag(monkeypatch):
    """Running against `tests/` fixtures must be the easy path, or nobody will use it."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN", "GRDN1234"])
    reader, crops = _synthetic_reader(spy)

    assert [reader.read(c) for _, c in crops] == ["CMFN", "GRDN1234"]
    assert len(spy.calls) == 2


def test_synthetic_mode_requires_a_non_empty_digest_set(monkeypatch):
    """An empty allowlist is a reader that can send nothing — fail at construction, loudly.

    Silently accepting `frozenset()` would produce a reader that raises on its first crop
    with a confusing provenance error instead of the real cause.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="synthetic_digests"):
        GeminiReader(
            source=CropSource.SYNTHETIC, synthetic_digests=frozenset(), transport=SpyTransport()
        )


def test_digests_are_ignored_on_the_real_path(monkeypatch):
    """A REAL reader takes no digest allowlist: the env var is its only key.

    Accepting one would invite "register the real crops and skip the flag" — the exact
    laundering route the synthetic check exists to close.
    """
    monkeypatch.setenv(BAA_ENV_VAR, "1")
    with pytest.raises(ValueError, match="synthetic_digests"):
        GeminiReader(
            source=CropSource.REAL,
            synthetic_digests=frozenset({"abc"}),
            transport=SpyTransport(),
        )


def test_crop_digest_is_content_addressed_not_object_identity():
    """Two identical crops hash alike; one changed pixel does not.

    The registration check must survive a crop being regenerated (the benchmark builds
    them twice) while still keying on the actual pixels that would be transmitted.
    """
    (_, a), (_, b) = _synthetic_crops(("CMFN", "GRDN1234"))
    a2 = a.copy()
    assert crop_digest(a) == crop_digest(a2)
    assert crop_digest(a) != crop_digest(b)


# --- rule #9: version identity ------------------------------------------------------------


def test_an_alias_has_no_version_until_the_api_answers(monkeypatch):
    """Vertex serves `gemini-2.5-pro` with no dated revision, so the pin moves to the reply.

    Reading `version` before any call must RAISE rather than return the alias: every caller
    of `version` is stamping an identity onto a result row, and a row stamped
    `gemini-2.5-pro` says only "some 2.5 Pro" — the ambiguity rule #9 exists to forbid.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"], model_version="gemini-2.5-pro-002")
    reader, crops = _synthetic_reader(spy)

    with pytest.raises(ServedVersionMismatch, match="alias"):
        _ = reader.version

    reader.read(crops[0][1])
    assert reader.version == "gemini-2.5-pro-002"  # pinned from the response
    assert reader.served_model_version == "gemini-2.5-pro-002"


def test_a_requested_id_that_names_a_revision_is_its_own_version(monkeypatch):
    """If Google ever publishes dated revisions again, the request pins it as before."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    crops = _synthetic_crops(("CMFN",))
    reader = GeminiReader(
        source=CropSource.SYNTHETIC,
        synthetic_digests=frozenset(crop_digest(c) for _, c in crops),
        transport=SpyTransport(["CMFN"]),
        model="gemini-2.5-pro-preview-06-05",
    )
    assert reader.version == "gemini-2.5-pro-preview-06-05"  # no call needed


def test_a_revision_change_mid_run_aborts(monkeypatch):
    """Google re-pointing the alias halfway through is an engine bump, not a hiccup.

    We cannot stop it happening. We can refuse to average the two halves into one number
    and call the result a measurement.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)

    versions = iter(["gemini-2.5-pro-002", "gemini-2.5-pro-003"])

    def drifting(call: GeminiCall) -> GeminiReply:
        return GeminiReply(
            text="CMFN", model_version=next(versions), prompt_tokens=270,
            output_tokens=6, thought_tokens=128, blocked=False,
        )

    reader, crops = _synthetic_reader(drifting)
    reader.read(crops[0][1])

    with pytest.raises(ServedVersionMismatch) as excinfo:
        reader.read(crops[1][1])

    assert "gemini-2.5-pro-002" in str(excinfo.value)
    assert "gemini-2.5-pro-003" in str(excinfo.value)


def test_an_alias_with_no_reported_revision_is_fatal(monkeypatch):
    """Unattributable output is not a measurement — it is a number with no provenance."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"], model_version="")
    reader, crops = _synthetic_reader(spy)

    with pytest.raises(ServedVersionMismatch, match="no model_version"):
        reader.read(crops[0][1])


def test_the_pinned_default_is_what_vertex_actually_serves():
    """Guards the 2026-08-11 finding: the dated preview revisions 404 on Vertex now."""
    assert PINNED_MODEL == "gemini-2.5-pro"
    assert not re.search(r"-\d{2}-\d{2}$|-\d{3}$", PINNED_MODEL)


def test_version_source_is_none(monkeypatch):
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    reader, _ = _synthetic_reader(SpyTransport())
    assert reader.version_source is None  # no library ships this model; see the module docstring


# --- determinism + the request itself ------------------------------------------------------


def test_the_request_is_deterministic_and_carries_no_sampling(monkeypatch):
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"])
    reader, crops = _synthetic_reader(spy)
    reader.read(crops[0][1])

    (call,) = spy.calls
    assert call.temperature == 0.0
    assert call.seed is not None
    assert call.model == PINNED_MODEL


def test_the_transmitted_bytes_are_exactly_the_crop_it_was_handed(monkeypatch):
    """No second preprocessing stage: what goes on the wire is the pinned crop, losslessly.

    A reader that re-scaled or JPEG-compressed here would be reading different pixels from
    every other reader in the arm, which is the one thing `crop_for_reading` exists to stop.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN"])
    reader, crops = _synthetic_reader(spy)
    _, crop = crops[0]
    reader.read(crop)

    sent = Image.open(io.BytesIO(spy.calls[0].image_png))
    assert sent.format == "PNG"  # lossless
    assert sent.size == crop.size
    assert crop_digest(sent.convert("RGB")) == crop_digest(crop)


def test_config_declares_every_knob_that_changes_the_output(monkeypatch):
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    reader, _ = _synthetic_reader(SpyTransport())
    config = reader.config()

    for key in ("model", "temperature", "seed", "max_output_tokens", "thinking_budget", "prompt"):
        assert key in config, key
    # A detection threshold must never appear here: this arm is handed its boxes, and a
    # decode-side score shares nothing with a detector's confidence (13g constraints).
    assert not [k for k in config if "det" in k or "confidence" in k]


def test_config_hash_splits_two_different_prompts(monkeypatch):
    """The prompt text IS the config: change it and the numbers are from a different arm."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    crops = _synthetic_crops(("CMFN",))
    digests = frozenset(crop_digest(c) for _, c in crops)
    common = {"source": CropSource.SYNTHETIC, "synthetic_digests": digests}

    a = GeminiReader(**common, transport=SpyTransport(), prompt="Read the text.")
    b = GeminiReader(**common, transport=SpyTransport(), prompt="Transcribe the text.")
    assert a.config_hash() != b.config_hash()


def test_config_id_is_derived_from_the_source_not_hand_set(monkeypatch):
    """A hand-typed label eventually lies. The arm's label follows the declaration."""
    monkeypatch.setenv(BAA_ENV_VAR, "1")
    real = GeminiReader(source=CropSource.REAL, transport=SpyTransport())
    monkeypatch.delenv(BAA_ENV_VAR)
    synth, _ = _synthetic_reader(SpyTransport())

    assert real.config_id != synth.config_id
    assert "synthetic" in synth.config_id


# --- what a refusal, a block, and an empty read mean ----------------------------------------


def test_a_blocked_or_empty_response_is_an_omission_not_a_crash(monkeypatch):
    """Safety filters fire on medical imagery. An empty read is a real answer — scored.

    Returning "" routes it to the omission axis via the existing scorer. Raising instead
    would abort a whole run on one crop; retrying would spend money to be blocked again.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)

    def blocking(call: GeminiCall) -> GeminiReply:
        return GeminiReply(
            text="", model_version=call.model, prompt_tokens=270,
            output_tokens=0, thought_tokens=0, blocked=True,
        )

    reader, crops = _synthetic_reader(blocking)
    assert reader.read(crops[0][1]) == ""
    assert reader.usage_summary()["blocked_calls"] == 1


def test_a_transport_failure_is_not_retried(monkeypatch):
    """One crop, one call. A retry loop is a second egress event on the same pixels.

    It would also make cost and latency depend on the network's mood, which the reported
    per-crop numbers cannot express.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    attempts = []

    def flaky(call: GeminiCall) -> GeminiReply:
        attempts.append(call)
        raise RuntimeError("503 backend unavailable")

    reader, crops = _synthetic_reader(flaky)
    with pytest.raises(RuntimeError, match="503"):
        reader.read(crops[0][1])
    assert len(attempts) == 1


# --- money ---------------------------------------------------------------------------------


def test_measured_token_usage_is_recorded_per_call(monkeypatch):
    """The estimate prices the input; only the API can report what was actually billed.

    2.5 Pro bills thinking tokens as output, and thinking is not optional on Pro — an
    input-side estimate alone would understate this arm's real cost.
    """
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN", "GRDN1234"])
    reader, crops = _synthetic_reader(spy)
    for _, crop in crops:
        reader.read(crop)

    usage = reader.usage_summary()
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 540
    assert usage["output_tokens"] == 12
    assert usage["thought_tokens"] == 256
    assert usage["measured_cost_usd"] > 0


def test_the_arm_is_priced_per_crop_through_the_existing_cost_table(monkeypatch):
    """No pricing logic in the runner: `@priced("gemini")` + the `PRICES` table, as always."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    spy = SpyTransport(["CMFN", "GRDN1234"])
    reader, crops = _synthetic_reader(spy)

    image, gt = synthetic.make_synthetic_image(("CMFN", "GRDN1234"))
    ref = ImageRef(
        id="synth-000", path="unused", w=image.width, h=image.height,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )
    per_crop = estimate_cost(ref, reader.read)

    assert PRICES["gemini_in"] > 0
    assert per_crop > 0


# --- the source-level guarantees the prompt asks a grep for -----------------------------------


def test_the_module_never_reads_a_credential_file():
    """Credentials come from the environment only — never a path this module opens."""
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in (".config/gcloud", ".aws/credentials", "application_default_credentials"):
        assert forbidden not in source, forbidden


def test_the_module_never_persists_a_response_body():
    """A cloud response holds read-back token text (CLAUDE.md §3): it is scored, not stored.

    Asserted against the source rather than behaviour because the failure mode is a debug
    line someone adds later, not a code path a test would naturally exercise.
    """
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in ("write_text(", "json.dump", "open(", "logging", "print("):
        assert forbidden not in source, forbidden


def test_the_gate_check_is_not_conditional_on_a_test_environment():
    """No `PYTEST_CURRENT_TEST`, no `if TESTING:` — a test must not be able to open the gate.

    Matched on word boundaries, not substrings: "CI" lives inside "specific" and "decision",
    and a test that fails on an innocent word gets deleted rather than fixed.
    """
    source = MODULE.read_text(encoding="utf-8")
    for forbidden in ("PYTEST", "pytest", "TESTING", "CI", "DEBUG"):
        assert not re.search(rf"\b{forbidden}\b", source), forbidden


# --- it is a Reader, and it scores through the frozen path --------------------------------


def test_it_scores_through_read_image_like_any_other_reader(tmp_path, monkeypatch):
    """End of the line: the same `read_image` -> `score()` path, tagged as a reading arm."""
    monkeypatch.delenv(BAA_ENV_VAR, raising=False)
    image, gt = synthetic.make_synthetic_image(("CMFN", "GRDN1234"), label="KEEP")
    path = tmp_path / "img.png"
    image.save(path)

    crops = [crop_for_reading(image, t.bbox) for t in gt]
    reader = GeminiReader(
        source=CropSource.SYNTHETIC,
        synthetic_digests=frozenset(crop_digest(c) for c in crops),
        transport=SpyTransport(["CMFN", "WRONG"]),
    )
    ref = ImageRef(
        id="synth-000", path=str(path), w=image.width, h=image.height,
        stratum="synth_ct_axial", modality="CT", vendor="FakeVendorA", frame_idx=0,
    )

    row = read_image(
        str(path), gt, reader, allowlist=set(synthetic.FAKE_TOKENS), image_ref=ref
    )

    assert row["keep_total"] == 2
    assert row["keep_exact_match_count"] == 1
    assert row["false_redaction_count"] == 1  # the misread token is destroyed downstream
    assert row["iou_thr"] is None  # oracle detector: no matcher ran


def test_gt_token_shape_is_unchanged_by_this_arm():
    """Sanity: this arm consumes the frozen GT schema, it does not extend it."""
    assert list(GTToken.__dataclass_fields__)[:3] == ["image_id", "series_uid", "modality"]
