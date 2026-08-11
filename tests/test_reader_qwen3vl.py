"""Phase 13d — tests for the Qwen3-VL reader. 100% synthetic: fake images, fake tokens.

Two tiers, and the split matters:

**Fast tier (always runs where `transformers` is importable).** The model and processor are
replaced by fakes, so the arm's *contract* — abstain handling, identity, one-crop-only, no
egress, no stray files — is proven without 16 GB of weights. A fake cannot accidentally read
well and hide a contract bug, which is the same reason `test_reading.py` scores through stubs.

**Weight tier (`QWEN3VL_WEIGHTS=1`).** Real greedy decodes of real synthetic crops: a known
token, determinism, abstention on a blank crop, charset sanity. Skipped by default because the
weights are a 9 GB local download and pytest must stay runnable in the shared benchmark venv,
which deliberately does not have this arm's dependencies installed.

Every accuracy signal in here is SYNTHETIC and must not be quoted as this arm's accuracy.
Qwen2.5-VL-7B scored 0.962/0.943 on synthetic imprinted text and 0.719/0.858 on real burned-in
text (arXiv 2511.02014); the gap between those two numbers is exactly the gap between this file
and Arnav's run.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="Qwen3-VL reader arm not installed in this venv")
torch = pytest.importorskip("torch", reason="Qwen3-VL reader arm not installed in this venv")

from harness.aggregate import aggregate  # noqa: E402
from harness.readers import read_qwen3vl  # noqa: E402
from harness.readers.read_qwen3vl import ABSTAIN_MARKER, Qwen3VLReader  # noqa: E402
from harness.reading import arm_config_hash, crop_for_reading, read_image  # noqa: E402
from tests import synthetic  # noqa: E402
from tests.test_reading import _image_ref, _saved  # noqa: E402

WEIGHTS = os.environ.get("QWEN3VL_WEIGHTS") == "1"
needs_weights = pytest.mark.skipif(
    not WEIGHTS, reason="set QWEN3VL_WEIGHTS=1 to run against the real 4B checkpoint"
)


# --- fakes ---------------------------------------------------------------------------------


class FakeProcessor:
    """The minimum of the processor API `read()` uses, recording what it was handed."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.images_seen: list[object] = []
        self.texts_seen: list[str] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "<chat>"

    def __call__(self, text, images, return_tensors="pt"):
        self.texts_seen.extend(text)
        self.images_seen.extend(images)
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return self.reply


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs: dict = {}

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.zeros((1, 6), dtype=torch.long)


def _reader(reply: str, **kw) -> Qwen3VLReader:
    """A reader whose weights are fakes — `_load()` is never reached."""
    reader = Qwen3VLReader(**kw)
    reader._processor = FakeProcessor(reply)
    reader._model = FakeModel()
    return reader


def _crop(token: str = "CMFN-00421"):
    image, gt = synthetic.make_synthetic_image((token,))
    return crop_for_reading(image, gt[0].bbox)


# --- the abstain contract ------------------------------------------------------------------


def test_abstain_marker_becomes_an_empty_string():
    """The one mapping the arm performs: `<<NOTEXT>>` in, omission out.

    An omission is a real answer — the reader saying "nothing here" — and it lands on the
    Found axis, never the Added one. A model that instead guessed would land on Added, which
    is the dangerous axis.
    """
    assert _reader(ABSTAIN_MARKER).read(_crop()) == ""


def test_abstain_marker_is_recognized_with_surrounding_whitespace():
    """`normalize()` (NFC + strip) is what decides, so a trailing newline still abstains."""
    assert _reader(f"  {ABSTAIN_MARKER}\n").read(_crop()) == ""


@pytest.mark.parametrize(
    "reply",
    [
        "<<NOTEXT>>.",
        '"<<NOTEXT>>"',
        "`<<NOTEXT>>`",
        "<< NOTEXT >>",
        "<<notext>>",
        "  <<NOTEXT>>  \n",
    ],
)
def test_a_decorated_abstain_marker_still_abstains(reply):
    """Each of these is the model obeying the instruction, with punctuation.

    Scoring them as reads is not a wash: on a control box a non-empty return is counted as
    INVENTED text, so a model that abstains politely would inflate the very hallucination
    floor that gates whether this arm's accuracy may be quoted.
    """
    assert _reader(reply).read(_crop()) == ""


def test_free_form_refusal_prose_is_NOT_treated_as_abstention():
    """Deliberate, and the asymmetry is the reason.

    Matching arbitrary prose means guessing at intent; a wrong guess there HIDES an invention,
    which is worse than inflating a floor that is already documented as an upper bound.
    """
    assert _reader("There is no readable text.").read(_crop()) == "There is no readable text."


def test_empty_and_whitespace_only_replies_are_also_empty():
    """A silent model and an abstaining one are scored identically: an omission, not a read."""
    assert _reader("").read(_crop()) == ""
    assert _reader("   \n\t ").read(_crop()) == ""


def test_a_chatty_reply_is_returned_intact_and_scores_as_a_misread():
    """The deliberate non-feature: no preface stripping.

    Peeling "The text reads:" off would be an undeclared knob that flatters this arm against
    recognizers that cannot be chatty at all. It is returned as-is, and on a KEEP token that
    is a false redaction — the headline metric, correctly charged to the model.
    """
    reply = "The text reads: CMFN-00421"
    assert _reader(reply).read(_crop()) == reply


def test_a_chatty_read_is_scored_as_a_false_redaction(tmp_path):
    """End-to-end through the frozen scorer, so the claim above is measured, not asserted."""
    image, gt = synthetic.make_synthetic_image(("CMFN",), label="KEEP")
    path = _saved(tmp_path, image)
    row = read_image(
        str(path),
        gt,
        _reader("The text reads: CMFN"),
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-000", path, image),
    )
    assert row["keep_total"] == 1
    assert row["false_redaction_count"] == 1
    assert row["omission_count"] == 0  # it answered; it just answered wrong


def test_abstention_is_an_omission_not_a_hallucination(tmp_path):
    image, gt = synthetic.make_synthetic_image(("CMFN",), label="KEEP")
    path = _saved(tmp_path, image)
    row = read_image(
        str(path),
        gt,
        _reader(ABSTAIN_MARKER),
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-000", path, image),
    )
    assert row["omission_count"] == 1
    assert row["added_count"] == 0


def test_abstaining_on_a_control_box_leaves_the_invention_floor_at_zero(tmp_path):
    """The negative control's happy path: blank frame, control boxes, nothing invented."""
    image, _ = synthetic.make_blank_image()
    path = _saved(tmp_path, image, "blank.png")
    row = read_image(
        str(path),
        [],
        _reader(ABSTAIN_MARKER),
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-blank", path, image),
        control_boxes=[(10.0, 10.0, 120.0, 40.0), (200.0, 300.0, 320.0, 330.0)],
    )
    assert row["added_count"] == 0


def test_text_on_a_control_box_is_counted_as_invention(tmp_path):
    """And the failure path: any non-empty read on a blank frame IS the hallucination floor."""
    image, _ = synthetic.make_blank_image()
    path = _saved(tmp_path, image, "blank.png")
    row = read_image(
        str(path),
        [],
        _reader("ACC-0001"),
        allowlist=set(synthetic.FAKE_TOKENS),
        image_ref=_image_ref("synth-blank", path, image),
        control_boxes=[(10.0, 10.0, 120.0, 40.0)],
    )
    assert row["added_count"] == 1


# --- identity (D-13.5, rule #9) --------------------------------------------------------------


def test_version_is_read_from_the_library_named_by_version_source():
    """Rule #9's "never hand-type a version", checked the way the runner contract checks it."""
    import importlib

    reader = Qwen3VLReader()
    module = importlib.import_module(reader.version_source)
    assert reader.version == module.__version__


IDENTITY_ONLY_ARGS = {"config_id"}
# Constructor arguments that are NOT output knobs and so must stay out of `config()`:
# `config_id` is the arm's label and is already part of `aggregate()`'s identity tuple on its
# own. Hashing it would split one measurement in two the moment someone renamed it.


def test_config_declares_every_constructor_knob():
    """An incomplete `config()` is the one remaining way to get a false identity (D-13.5)."""
    params = {p for p in inspect.signature(Qwen3VLReader.__init__).parameters if p != "self"}
    declared = set(Qwen3VLReader().config())
    missing = params - declared - IDENTITY_ONLY_ARGS
    assert missing == set(), f"undeclared knobs: {sorted(missing)}"
    assert IDENTITY_ONLY_ARGS.isdisjoint(declared), "an arm label must not be hashed as a knob"


def test_config_pins_the_model_revision_and_carries_the_prompt_in_full():
    """The model revision has no `__version__` to live in, so the hash is its only guard."""
    config = Qwen3VLReader().config()
    assert re.fullmatch(r"[0-9a-f]{40}", str(config["revision"])), "pin a commit sha, not a branch"
    assert config["prompt"] == read_qwen3vl.PROMPT_V1
    assert config["do_sample"] is False and config["num_beams"] == 1


def test_the_torch_version_is_part_of_the_identity():
    """fp32 CPU decode arithmetic is a torch property, so a torch bump is a possible output bump."""
    assert Qwen3VLReader().config()["torch_version"] == torch.__version__


def test_model_name_follows_model_id_so_a_second_candidate_is_not_mislabelled():
    """D-11.1 is a four-way A/B run through this one class; a hardcoded name would mislabel it.

    `aggregate()` would still refuse to blend the two (their hashes differ), so the damage is
    not a corrupted average — it is a correct number printed under the wrong model's name.
    """
    assert Qwen3VLReader().model_name == "qwen3-vl-4b-instruct"
    assert Qwen3VLReader(model_id="Qwen/Qwen3-VL-8B-Instruct").model_name == "qwen3-vl-8b-instruct"


def test_a_relabelled_arm_cannot_pool_with_the_default_one():
    """How the floor run keeps out of the accuracy run's rows: a distinct `config_id`.

    The label is in `aggregate()`'s identity tuple, so relabelling is enough to make the two
    un-blendable — without pretending some knob changed.
    """
    floor = Qwen3VLReader(config_id="floor-4box")
    assert floor.config_id != Qwen3VLReader().config_id
    assert floor.config_hash() == Qwen3VLReader().config_hash()  # same knobs, different arm


def test_two_prompts_are_two_arms_and_cannot_be_blended(tmp_path):
    """A prompt change is an arm change (13a) — `aggregate()` must refuse to average them."""
    image, gt = synthetic.make_synthetic_image(("CMFN",))
    path = _saved(tmp_path, image)
    img_ref = _image_ref("synth-000", path, image)

    def rows_for(reader):
        row = read_image(
            str(path), gt, reader, allowlist=set(synthetic.FAKE_TOKENS), image_ref=img_ref
        )
        row.update(
            stratum=img_ref.stratum,
            model_name=reader.model_name,
            version=reader.version,
            config_id=reader.config_id,
            config_hash=arm_config_hash(reader),
            verifier_model_name=None,
            verifier_version=None,
            verifier_elapsed=None,
        )
        return [row]

    a = _reader("CMFN")
    b = _reader("CMFN", prompt="Read the text.", prompt_id="v2-test")
    assert a.config_hash() != b.config_hash()

    with pytest.raises(ValueError, match="refusing to blend"):
        aggregate(rows_for(a) + rows_for(b))


def test_two_revisions_are_two_arms():
    """A silent checkpoint bump is rule #9's failure mode; the hash is what makes it loud."""
    pinned = Qwen3VLReader()
    other = Qwen3VLReader(revision="0" * 40)
    assert pinned.config_hash() != other.config_hash()


def test_thread_count_is_part_of_the_identity():
    """On CPU it changes reduction order, hence the arithmetic, hence possibly the argmax."""
    assert Qwen3VLReader().config_hash() != Qwen3VLReader(torch_num_threads=1).config_hash()


# --- the hard constraints: no egress, one crop, no stray files -------------------------------


def _executable_source() -> str:
    """The module's CODE with every docstring and comment removed.

    The prose in this reader is full of the words a naive grep for egress would flag ("HTTP
    client", "API key", "self-hosted"). Stripping docstrings and comments means the egress
    test asserts something about what the module DOES, which is the only thing that can leak.
    """
    tree = ast.parse(Path(read_qwen3vl.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body.pop(0)
    return ast.unparse(tree)  # unparse never emits comments


def test_the_module_contains_no_network_client():
    """CLAUDE.md §4: Qwen3-VL is ✅ only if self-hosted. Grep, not trust.

    Also catches the subtler version of the mistake — an `openai`-style client pointed at a
    local server is still an HTTP path, and a `base_url` knob is how one gets added later.
    """
    code = _executable_source()
    banned = (
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "api_key",
        "base_url",
        "endpoint",
        "openai",
        "http://",
        "https://",
    )
    hits = [word for word in banned if word in code]
    assert hits == [], f"possible egress path in the reader: {hits}"


def test_weights_are_loaded_local_files_only():
    """A missing checkpoint must fail loudly, not silently fetch whatever is at that sha."""
    # Counted on the executable source so the docstring's own mention is not what passes.
    assert _executable_source().count("local_files_only=True") == 2  # processor AND model


def test_the_model_sees_exactly_one_crop_and_nothing_else():
    """The arm's definition and minimum-necessary at once: no frame, no path, no batch."""
    reader = _reader("CMFN")
    crop = _crop()
    reader.read(crop)

    processor = reader._processor
    assert processor.images_seen == [crop]
    assert len(processor.texts_seen) == 1
    assert crop.size[1] == 48  # what it saw is the pinned 48px crop, not the frame


def test_the_prompt_reaches_the_model_from_the_constant_not_the_call_site():
    reader = _reader("CMFN")
    reader.read(_crop())
    content = reader._processor.messages[0]["content"]
    assert {"type": "text", "text": read_qwen3vl.PROMPT_V1} in content


def test_decoding_is_greedy():
    reader = _reader("CMFN")
    reader.read(_crop())
    kwargs = reader._model.generate_kwargs
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["max_new_tokens"] == read_qwen3vl.MAX_NEW_TOKENS


def test_no_confidence_is_extracted_anywhere():
    """A decode logprob is not a detection score and must never share a threshold with one.

    `output_scores` / `return_dict_in_generate` are the only two ways to get one out of
    `generate()`; neither is set, so there is nothing for a downstream threshold to grab.
    """
    code = _executable_source()
    assert "output_scores" not in code
    assert "return_dict_in_generate" not in code
    assert "logprob" not in code and "confidence" not in code


def test_reading_writes_no_file_the_caller_did_not_ask_for(tmp_path, monkeypatch):
    """On real data a stray file is engine-read token text landing somewhere unmanaged."""
    monkeypatch.chdir(tmp_path)
    _reader("CMFN").read(_crop())
    assert list(Path(tmp_path).iterdir()) == []


def test_constructing_a_reader_does_not_load_the_weights():
    """The experiment CLI builds every arm just to hash it; a 16 GB load per build is fatal."""
    reader = Qwen3VLReader()
    assert reader._model is None and reader._processor is None


# --- the processor, without the weights ------------------------------------------------------


@pytest.fixture(scope="module")
def real_processor():
    """The real processor only (a few MB of config + tokenizer), never the 9 GB of weights."""
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            read_qwen3vl.MODEL_ID, revision=read_qwen3vl.REVISION, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 — any load failure means "not cached here"
        pytest.skip(f"Qwen3-VL processor is not in the local HF cache: {type(exc).__name__}")


def test_the_chat_template_really_produces_a_vision_placeholder(real_processor):
    """The failure this catches is invisible: a template that drops the image part.

    The model would then answer from the prompt text alone — 100% invention — and every test
    driven by a fake processor would still pass, because a fake accepts any message shape.
    """
    chat = real_processor.apply_chat_template(
        Qwen3VLReader()._messages(), tokenize=False, add_generation_prompt=True
    )
    assert "vision_start" in chat or "image_pad" in chat, chat[:200]
    assert read_qwen3vl.PROMPT_V1 in chat


def test_exactly_one_image_placeholder_is_emitted(real_processor):
    """One crop per call is the arm's definition; two placeholders would mean two images."""
    chat = real_processor.apply_chat_template(
        Qwen3VLReader()._messages(), tokenize=False, add_generation_prompt=True
    )
    assert chat.count("vision_start") == 1


# --- weight-backed tier ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_reader():
    reader = Qwen3VLReader()
    reader._ensure_loaded()
    return reader


@needs_weights
def test_real_read_of_a_known_synthetic_token(real_reader):
    """SYNTHETIC accuracy — a sanity check that the wiring works, never a reported number."""
    assert "CMFN" in real_reader.read(_crop("CMFN-00421")).upper()


@needs_weights
def test_real_reads_are_deterministic(real_reader):
    crop = _crop("GRDN5678")
    assert real_reader.read(crop) == real_reader.read(crop)


@needs_weights
def test_real_abstention_on_a_blank_crop(real_reader):
    """Whether an untrained VLM actually abstains is measured here, never assumed."""
    blank, _ = synthetic.make_blank_image()
    assert real_reader.read(crop_for_reading(blank, (10.0, 10.0, 140.0, 44.0))) == ""


@needs_weights
def test_real_read_stays_in_the_latin_alphanumeric_charset(real_reader):
    """A VLM answering in another script on a Latin-digit token is a mode worth catching."""
    text = real_reader.read(_crop("ACC-0099"))
    assert all(ord(ch) < 128 for ch in text), "non-ASCII leaked into a Latin-digit read"
