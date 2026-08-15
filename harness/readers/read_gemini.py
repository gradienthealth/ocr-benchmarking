"""Phase 13g — the Gemini 2.5 Pro reader. 🔴 BAA-GATED — synthetic crops only until cleared.

This is a **Phase 12 arm, not part of the local bake-off**. Every crop it reads leaves the
environment: the pixels of a burned-in patient ID are sent to Google and read back as text.
Under CLAUDE.md §4 that is permissible only when the Google Cloud BAA is signed AND the
specific product being called is on Google's covered-products list. "Google has a BAA" is
not sufficient for a specific model on a specific surface — which is why this module makes
the surface part of the gate (see `_live_transport`).

THE GATE — the point of this module
---------------------------------------------------------------------------------
Every reader is constructed with an explicit `source=`, and there is no default:

  `CropSource.REAL`       a render off the scanner. Requires the environment variable
                          `OCR_BAA_CLEARED_GEMINI=1`, set by a human who has completed the
                          D-12.1 checklist. Checked at construction AND immediately before
                          every single call, above the transport, so an injected transport,
                          a subclass, or a long-lived reader outliving its clearance cannot
                          route around it. Absent the variable, it raises `BAAGateError`.

  `CropSource.SYNTHETIC`  fake `CMFN`-style fixtures. Needs no flag — but it does not accept
                          the caller's word for it either. The reader is given the set of
                          crop digests it may transmit, up front, and hashes every crop
                          against that set. A real render fails on its pixels rather than on
                          anyone's honesty, so "declare it synthetic" is not a bypass.

  The residual hole, stated plainly: someone who hashes real crops and passes those digests
  in gets a live call without the flag. That is forgery, not an accident — there is no
  code path in this repo that produces such a set — but the guarantee is "a real render
  cannot reach the network by mistake," not "cannot reach it by deliberate act."

WHAT THIS MODULE NEVER DOES
---------------------------------------------------------------------------------
- Never persists or logs a response body. A reply holds the read-back token text, which is
  PHI on a real crop (CLAUDE.md §3); it is scored and dropped. There is no `raw_response`
  here, nothing is written to disk, and nothing is printed — the reading arm's `Reader`
  contract returns one string, which is all this arm has.
- Never opens a credential file. Credentials come from the process environment only.
- Never retries. A retry is a second egress event on the same pixels, and it makes cost and
  latency depend on the network's mood rather than on the model.

RULE #9 WITH NO `__version__` TO SOURCE FROM
---------------------------------------------------------------------------------
No installed library ships this model, so `version_source` is None and the shared
version-sourcing check in `tests/test_readers.py` cannot apply — this reader is named in
that test's exemption list, which is how opting out stays a visible, reviewable edit rather
than a default. Setting `version_source = "google.genai"` would be worse than useless: the
SDK version says nothing about what the model returns, so it would attach a confident,
wrong provenance to every row.

What replaces it: the revision is pinned from the RESPONSE. Vertex serves only the alias
`gemini-2.5-pro` (verified 2026-08-11 — the dated preview revisions 404), so there is no
revision to put in the request. The first reply's `model_version` becomes this arm's
`version`; reading `version` before that raises rather than stamping a placeholder onto a
result row; a later reply naming a different revision aborts the run; and an alias plus a
reply carrying no revision at all is fatal, because an unattributable number is not a
measurement.

Known limitation, to be stated in any write-up rather than implied away: Vertex reports
`model_version` as the alias itself, so this arm's identity is really "whatever
`gemini-2.5-pro` was on the run date." The mechanism would catch a mid-run change; it
cannot manufacture precision Google does not publish. The run date is part of the version.

COST
---------------------------------------------------------------------------------
`harness.cost` prices the input side from the crop's dimensions, as for every other arm.
It cannot see the output side, and on 2.5 Pro thinking tokens are billed as output and
cannot be switched off — so the reader also accumulates the token counts the API actually
reports and exposes them via `usage_summary()`. Those measured numbers, not the estimate,
are what the arm should be priced on. Token counts are integers: PHI-free.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import string
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PIL import Image

from harness.cost import PRICES, priced
from harness.reading import Reader

BAA_ENV_VAR = "OCR_BAA_CLEARED_GEMINI"
# Set to exactly "1", by a human, after the D-12.1 checklist is confirmed in writing.
# Deliberately not read from any committed file and never written by this repo: a value
# that lives only in one person's shell is a value nobody can set by merging a branch.

GATE_VALUE = "1"
# Compared with `==`, never with truthiness. `if os.environ.get(VAR):` would open the gate
# on the strings "0" and "no" — the two values someone would type to keep it shut.

PINNED_MODEL = "gemini-2.5-pro"
# An ALIAS, not a revision — and deliberately so, because Vertex offers nothing else.
# Verified 2026-08-11 against `gcloud ai model-garden models list` on gradient-health-central:
# the only 2.5 Pro entry is `google/gemini-2.5-pro@default`. The dated preview revisions
# (`-preview-06-05` and friends) were retired when the model went GA and now 404.
#
# This does not weaken rule #9, it relocates it. An alias re-points server-side without
# notice, so the revision cannot be pinned in the REQUEST; it is pinned from the RESPONSE
# instead. `version` is unknown until the API answers, becomes whatever revision answered,
# and any later call reporting a different one aborts the run (see `_record`). The
# guarantee that matters is unchanged: every reported number names the exact revision that
# produced it, and one arm never mixes two.

_PINNED_REVISION = re.compile(r"-(\d{2}-\d{2}|\d{3})$")
# Classifies an id, no longer rejects one. An id ending in a dated (`-06-05`) or numbered
# (`-001`) revision names its own version, so a reply that reports no revision is
# tolerable. An alias like `gemini-2.5-pro` names nothing, so a reply that reports no
# revision leaves the arm with no identity at all — and that is fatal, not tolerable.

PROMPT = (
    "Transcribe the characters in this image exactly as they appear. "
    "Reply with the characters only: no explanation, no punctuation you do not see, "
    "no guessing. If nothing is legible, reply with nothing at all."
)
# Part of the arm's identity, hashed into `config_hash`: two prompts are two arms.
# Phrased to suppress the failure this arm is most likely to have — a generative model
# completing a plausible identifier rather than reporting an unreadable one.

MAX_OUTPUT_TOKENS = 512
# **Thinking tokens count against this ceiling on 2.5 Pro**, and thinking cannot be turned
# off — so this is not "how long may the answer be," it is "how much thinking plus answer."
# The first live run (2026-08-11) used 64 with a 128-token thinking budget: 5 of 18 crops
# came back empty, because the budget was spent before a single character was emitted.
# 512 leaves room for the floor of thinking plus a short token. Raising it does not raise
# the bill much — output is billed per token used, not per token allowed.

THINKING_BUDGET = 128
# 2.5 Pro cannot have thinking disabled; 128 is its floor. Pinned rather than left to the
# default because the default is a server-side choice that can move under us, and thinking
# tokens are billed as output.

SEED = 13
# Fixed so a repeat run of the same crop asks for the same decode. Determinism here is a
# request, not a guarantee: a hosted model can still drift, which is what the served-version
# cross-check is for.


class CropSource(Enum):
    """What kind of pixels this reader is allowed to transmit. No default anywhere."""

    SYNTHETIC = "synthetic"
    REAL = "real"


class BAAGateError(RuntimeError):
    """Raised when a real crop is handed to a reader whose BAA gate is shut."""


class SyntheticProvenanceError(RuntimeError):
    """Raised when a crop offered to the synthetic path was not registered up front."""


class ServedVersionMismatch(RuntimeError):
    """Raised when the API answers from a different revision than the pinned one."""


@dataclass(frozen=True)
class GeminiCall:
    """One request, fully specified. Carries the exact bytes that go on the wire."""

    image_png: bytes
    model: str
    prompt: str
    temperature: float
    seed: int
    max_output_tokens: int
    thinking_budget: int


@dataclass(frozen=True)
class GeminiReply:
    """One response, reduced to the scored string plus PHI-free accounting.

    The response object itself never leaves the transport: `text` is the only content that
    escapes, and it is returned to the scorer, not stored.
    """

    text: str
    model_version: str | None
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    blocked: bool

    finish_reason: str | None = None
    # Why generation stopped, verbatim from the API. Carried because an empty read has at
    # least two causes that demand opposite responses: `MAX_TOKENS` is our ceiling being too
    # low (a config bug, fix it and re-run) while `SAFETY` is a filter firing on medical
    # imagery (a real property of the arm, report it). Counting them together as "blocked"
    # hides the first behind the second. PHI-free: an enum name, never content.

    @classmethod
    def empty(cls) -> GeminiReply:
        """A reply that read nothing — the shape a stub needs when it has nothing to say."""
        return cls(
            text="",
            model_version=None,
            prompt_tokens=0,
            output_tokens=0,
            thought_tokens=0,
            blocked=False,
        )


Transport = Callable[[GeminiCall], GeminiReply]


def crop_digest(crop: Image.Image) -> str:
    """Content address of the pixels that would be transmitted.

    Hashes the raw RGB bytes and the size, not the encoded file, so a crop regenerated by
    the benchmark hashes identically to the one registered earlier while a single changed
    pixel does not.
    """
    rgb = crop.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{rgb.width}x{rgb.height}|".encode())
    h.update(rgb.tobytes())
    return h.hexdigest()


def _encode(crop: Image.Image) -> bytes:
    """PNG, losslessly. The crop was produced by the pinned `crop_for_reading` pipeline.

    Re-scaling or JPEG-encoding here would mean this arm reads different pixels than every
    other reader in the arm — the one thing that pipeline exists to prevent.
    """
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _live_transport(call: GeminiCall) -> GeminiReply:
    """The real network call. Vertex AI only — the surface the Google BAA can cover.

    The Gemini Developer API (AI Studio, `GEMINI_API_KEY`) is a separate product and is NOT
    on the Google Cloud covered-products list, so a real render must never go there. This
    transport therefore takes its configuration from `GOOGLE_CLOUD_PROJECT` /
    `GOOGLE_CLOUD_LOCATION` and lets the client resolve credentials from the environment
    itself — this module opens no credential file and holds no key.

    Imported lazily so the whole arm, tests included, works with the SDK absent.
    """
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        raise BAAGateError(
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must both be set: this arm "
            "calls Gemini on Vertex AI, the surface a Google Cloud BAA covers. The Gemini "
            "Developer API (AI Studio) is a different product and is not covered — see "
            "CLAUDE.md §4 and the D-12.1 checklist."
        )

    client = genai.Client(vertexai=True, project=project, location=location)
    response = client.models.generate_content(
        model=call.model,
        contents=[
            types.Part.from_bytes(data=call.image_png, mime_type="image/png"),
            call.prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=call.temperature,
            seed=call.seed,
            candidate_count=1,
            max_output_tokens=call.max_output_tokens,
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(thinking_budget=call.thinking_budget),
        ),
    )

    usage = response.usage_metadata
    text = response.text
    candidates = response.candidates or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    return GeminiReply(
        finish_reason=str(finish) if finish is not None else None,
        # `.text` is None when the reply was filtered or empty; "" scores as an omission,
        # which is a real answer. The response object is not kept beyond this expression.
        text=text or "",
        model_version=response.model_version,
        prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        thought_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        blocked=not text,
    )


class GeminiReader(Reader):
    """Gemini 2.5 Pro as a reading-arm recognizer: one crop in, one string out.

    Expected to lose to the local readers — a projected ~59% exact match on 8 characters.
    The point of the arm is to have the number measured rather than assumed; it is not a
    candidate to promote, and it cannot run on a real frame until the gate is opened.
    """

    model_name = "gemini-2.5-pro"

    version_source = None
    # No installed library carries this model's version — see the module docstring for the
    # two checks that replace the usual `__version__` cross-check.

    def __init__(
        self,
        *,
        source: CropSource,
        synthetic_digests: frozenset[str] | None = None,
        transport: Transport | None = None,
        model: str = PINNED_MODEL,
        prompt: str = PROMPT,
    ) -> None:
        """Build a reader for one declared crop source. `source` is required, always.

        Raises:
            BAAGateError: `source=REAL` while `OCR_BAA_CLEARED_GEMINI` is not exactly "1".
            ValueError: a REAL reader handed a digest allowlist; a SYNTHETIC reader handed
                none.
        """
        if source is CropSource.REAL:
            if synthetic_digests is not None:
                raise ValueError(
                    "a REAL reader takes no synthetic_digests: registering crops must never "
                    f"become a way to skip {BAA_ENV_VAR}."
                )
            _require_clearance()
        else:
            if not synthetic_digests:
                raise ValueError(
                    "synthetic_digests must be a non-empty set of crop digests: the "
                    "synthetic path transmits registered pixels only, so an empty set is a "
                    "reader that can read nothing."
                )

        self.source = source
        self.requested_model = model
        # What goes in the REQUEST. Not the arm's version unless it happens to name a
        # revision — see `version` below.
        self.config_id = f"gemini-{source.value}"
        # Derived from the declaration, never hand-set: a label somebody types by hand
        # eventually disagrees with the thing it labels.
        self.prompt = prompt
        self._digests = frozenset(synthetic_digests or ())
        self._transport: Transport = transport or _live_transport

        self.served_model_version: str | None = None
        # Filled from the first reply that reports one; stays None if the API says nothing.

        self._version: str | None = model if _PINNED_REVISION.search(model) else None
        # A requested id that names a revision IS the version. An alias has none until the
        # API answers, which is why `version` raises rather than guessing.

        self._calls = 0
        self._blocked = 0
        self._finish_reasons: dict[str, int] = {}
        self._prompt_tokens = 0
        self._output_tokens = 0
        self._thought_tokens = 0

    @property
    def version(self) -> str:
        """The revision that actually produced this arm's numbers. Never a guess.

        Raises rather than returning a placeholder when the requested id is an alias and no
        call has been made yet. Every caller of `version` is stamping an identity onto a
        result row, and a row stamped `gemini-2.5-pro` says only "some 2.5 Pro" — which is
        precisely the ambiguity rule #9 forbids. `run_reading()` reads this after the first
        crop is read, so the ordinary path never sees the raise.
        """
        if self._version is None:
            raise ServedVersionMismatch(
                f"{self.requested_model!r} is an alias, so this arm has no version until "
                "the API reports the revision that answered. Read one crop first. If a "
                "reply already came back and this still raises, the API returned no "
                "`model_version` and the arm cannot be scored: there would be no way to "
                "say which revision produced the numbers (rule #9)."
            )
        return self._version

    @priced("gemini")
    def read(self, crop: Image.Image) -> str:
        """Read one preprocessed crop. The gate is checked here, above the transport.

        Order matters and is the whole design: authorization first, encoding second,
        network last. Nothing touches the wire until the crop has earned the right to be
        there — either the human-set variable is present, or the pixels were registered.
        """
        self._authorize(crop)

        call = GeminiCall(
            image_png=_encode(crop),
            model=self.requested_model,
            prompt=self.prompt,
            temperature=0.0,
            seed=SEED,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_budget=THINKING_BUDGET,
        )
        reply = self._transport(call)  # no retry: see the module docstring

        self._record(reply)
        return reply.text

    def _authorize(self, crop: Image.Image) -> None:
        """Whichever door this reader came in by, it is checked again on every crop."""
        if self.source is CropSource.REAL:
            _require_clearance()
            return

        if crop_digest(crop) not in self._digests:
            raise SyntheticProvenanceError(
                "this crop was not registered as synthetic, so it will not be transmitted. "
                "The synthetic path sends pre-registered fixture pixels only. To read real "
                f"renders, build the reader with source=CropSource.REAL and set {BAA_ENV_VAR}"
                "=1 after completing the D-12.1 checklist (CLAUDE.md §4)."
            )

    def _record(self, reply: GeminiReply) -> None:
        """Accumulate PHI-free accounting and pin the revision from the response.

        The first reply that names a revision fixes this arm's `version`; every later reply
        must name the same one. That is the whole of rule #9 for a hosted model: we cannot
        stop Google re-pointing an alias mid-run, but we can refuse to average the two
        halves into one number and call it a measurement.
        """
        if reply.model_version:
            if self._version is None:
                self._version = reply.model_version
            elif reply.model_version != self._version:
                raise ServedVersionMismatch(
                    f"this arm has been answering as {self._version!r} and just answered as "
                    f"{reply.model_version!r}. A revision change mid-run invalidates the "
                    "rows already scored, the way a gt.csv change does (rule #9): re-run "
                    "against one revision, do not mix them."
                )
            self.served_model_version = reply.model_version
        elif self._version is None:
            raise ServedVersionMismatch(
                f"the API reported no model_version and {self.requested_model!r} is an "
                "alias, so nothing here knows which revision produced this read. An "
                "unattributable number is not a measurement (rule #9) — pin a dated "
                "revision if the API offers one, or fix the transport so it passes the "
                "reported version through."
            )

        self._calls += 1
        self._blocked += int(reply.blocked)
        if reply.finish_reason:
            self._finish_reasons[reply.finish_reason] = (
                self._finish_reasons.get(reply.finish_reason, 0) + 1
            )
        self._prompt_tokens += reply.prompt_tokens
        self._output_tokens += reply.output_tokens
        self._thought_tokens += reply.thought_tokens

    def usage_summary(self) -> dict[str, object]:
        """What the API said it billed — counts only, never content. Safe to report.

        `measured_cost_usd` is the honest number for this arm: `harness.cost` can only see
        the input side, and on 2.5 Pro the thinking tokens it cannot see are billed as
        output and are not optional.
        """
        billable_output = self._output_tokens + self._thought_tokens
        return {
            "calls": self._calls,
            "blocked_calls": self._blocked,
            "finish_reasons": dict(sorted(self._finish_reasons.items())),
            "prompt_tokens": self._prompt_tokens,
            "output_tokens": self._output_tokens,
            "thought_tokens": self._thought_tokens,
            "measured_cost_usd": (
                self._prompt_tokens * PRICES["gemini_in"]
                + billable_output * PRICES["gemini_out"]
            ),
            "served_model_version": self.served_model_version,
        }

    def charset(self) -> frozenset[str]:
        """Single characters this reader can emit. Structurally unconstrained, not measured.

        Same contract as 13c's and 13d's readers so `tests/test_readers.py`'s shared charset
        check applies here unchanged — but the honest answer is a different KIND of answer
        than theirs, and that difference is worth stating rather than hiding behind a
        matching signature.

        A CTC recognizer has a decoder dictionary that can be read off the model; Qwen3-VL
        has a tokenizer whose single-character pieces can be enumerated. This model is behind
        an API: there is no dictionary to inspect, and a general text model is not restricted
        to one. So this returns printable ASCII as a claim about the decoder's *structure*
        (nothing constrains it to a subset), not as a measurement of an artifact we hold.

        The check still earns its place for the other readers, and failing it is impossible
        here — which is the point worth flagging: **this arm's risk is the opposite one.** A
        recognizer that cannot spell `A-Z0-9-` scores badly for a reason unrelated to reading;
        a generative model can spell anything, including a plausible identifier that is not
        in the image. Invention is what to watch here, and it is measured by the negative
        control, never by this method.
        """
        return frozenset(string.printable.strip())

    def config(self) -> dict[str, object]:
        """Every knob that changes what this reader outputs (D-13.5).

        `source` is in here because the two sources are genuinely different measurements —
        fixture glyphs and scanner glyphs — and their rows must never merge into one arm.
        No detection threshold appears: this arm is handed its boxes, and whatever
        decode-side score a generative model might expose shares nothing with a detector's
        confidence, so the two must never meet in one number.
        """
        return {
            "model": self.requested_model,
            # The REQUESTED id, not the served revision: the served one lands in `version`,
            # which `aggregate()` already keys on separately. Putting it here too would make
            # the config hash change the moment Google re-points the alias, splitting one
            # config into two for a reason that is not a config change.
            "source": self.source.value,
            "prompt": self.prompt,
            "temperature": 0.0,
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking_budget": THINKING_BUDGET,
            "response_mime_type": "text/plain",
            "candidate_count": 1,
        }


def _require_clearance() -> None:
    """The gate. One function, one comparison, called at construction and at every read."""
    if os.environ.get(BAA_ENV_VAR) != GATE_VALUE:
        raise BAAGateError(
            f"{BAA_ENV_VAR} is not set to {GATE_VALUE!r}, so no real render may be sent to "
            "Gemini. This arm is cloud egress: the crop leaves the environment and the "
            "burned-in identifier is read back by a third party. Under CLAUDE.md §4 that "
            "requires a signed Google Cloud BAA naming the product actually called "
            "(Gemini on Vertex AI), confirmed in writing per the D-12.1 checklist. A human "
            "sets this variable after that confirmation — nothing in this repo sets it. To "
            "exercise the arm meanwhile, use source=CropSource.SYNTHETIC with registered "
            "fixture crops, which needs no clearance."
        )
