"""Phase 13d — Arm C reader: Qwen3-VL, self-hosted, one crop at a time. 🟢 PHI-free code.

The code here is PHI-free; a RUN of it is PHI-touching, because the crops come from real
renders. That is the whole design: the weights load from the local HuggingFace cache and the
forward pass happens in-process, so no pixel and no read-back token ever leaves the
environment. CLAUDE.md §4 marks Qwen3-VL ✅ **only if self-hosted** — this module is what
"self-hosted" means in code. There is no HTTP client in it, no API key, no base URL, and no
remote inference path to configure; `tests/test_reader_qwen3vl.py` greps for all of those.

WHY A VLM IS ON THE READING ARM AT ALL
---------------------------------------------------------------------------------
Two reasons, and they are different questions that this one arm answers at once:

1. **Arm C of the bake-off.** docTR and PP-OCRv6 read with a CRNN/SVTR recognizer trained on
   text lines. A VLM reads with a language model, which can *invent* a plausible token where a
   CRNN can only garble one. On an allowlist redaction pipeline an invented-but-plausible ID is
   worse than a garbled one, so the invention rate has to be measured, not assumed.
2. **Evidence for D-11.1** (reopened 2026-08-09) — which VLM becomes the Phase-11 verifier. The
   incumbent pick was never measured on this kind of text. Its predecessor, Qwen2.5-VL-7B,
   scored 0.962/0.943 precision/recall on *synthetic* imprinted text and collapsed to
   0.719/0.858 on *real* burned-in text (arXiv 2511.02014). Which is also the warning label on
   every synthetic number this module's own tests produce.

`model_id` is a constructor argument precisely because D-11.1 is a four-way A/B (4B / 8B /
RolmOCR / InternVL3.5-8B): a second candidate is a second instance with its own identity, not a
fork of this file.

MEASURE THE FLOOR BEFORE QUOTING THE ACCURACY
---------------------------------------------------------------------------------
plan.md requires every generative arm to run its negative control first. For a reader the
control is not "a blank image" — a reader handed only GT boxes cannot invent a *location*, so
its `added_count` is structurally zero and it would appear to hallucinate at exactly 0.0. The
floor comes from control boxes on confirmed-blank frames; `experiments/reader_negative_control.py`
is the procedure, and `harness/reading.py:read_image` refuses control boxes anywhere else.

WHAT COUNTS AS THIS ARM'S IDENTITY (D-13.5, rule #9)
---------------------------------------------------------------------------------
`version` is the transformers version, cross-checked against `version_source` by the shared
contract test — the "never hand-type a version" rule. The **model** revision cannot live there
(it is not any module's `__version__`), so it lives in `config()`, which is hashed into
`config_hash`; `aggregate()` then refuses to blend two revisions the way it refuses to blend two
engines. Callers that write run metadata write `reader.config()` verbatim so the sha stays
human-readable and not only hashed.

The prompt is part of that identity: two prompts are two arms, so it is a module constant with
an id, never a string at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from harness.contract import normalize
from harness.reading import Reader

if TYPE_CHECKING:  # importing the model stack is expensive; keep it out of type-check time
    from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# The 4B Instruct checkpoint. Apache-2.0, no revenue gate (unlike Surya 2 — CLAUDE.md rule #10).

REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
# EXACT commit sha, never a branch name. `revision="main"` is a model that changes under the
# results — the silent version bump rule #9 exists to stop, with no version string to notice it
# by. Everything the checkpoint carries is pinned by this: weights, tokenizer, chat template,
# AND `preprocessor_config.json`, which owns the image-side resizing the processor applies on
# top of our crop. Bump this only the way an engine bump is done — loudly, and re-run.

PROMPT_ID = "v1"
PROMPT_V1 = (
    "Transcribe the text in this image exactly as it appears. "
    "Output only the transcribed characters: no explanation, no quotes, no extra words. "
    "If there is no readable text, output exactly <<NOTEXT>> and nothing else."
)
# THE prompt. A module constant with an id because a prompt change is an arm change (13a): the
# same weights under two prompts are two different measurements, and `config()` puts the full
# text into the hash so the two cannot merge.

ABSTAIN_MARKER = "<<NOTEXT>>"
# The abstain contract. "No untrained VLM abstains reliably" (arXiv 2511.19806) — this gives
# abstention a defined form so the arm can MEASURE how often it happens instead of assuming it.
# Angle-bracket-doubled so it cannot collide with a real read: clinical burned-in tokens are
# alphanumerics, dashes, dots and slashes.

MAX_NEW_TOKENS = 32
# Burned-in tokens are short (an accession number, an ID, a laterality letter). 32 is roomy for
# every one of them and caps the damage when the model decides to write a sentence instead —
# which is a real failure mode this arm is here to measure, not to hide.

DTYPE = "float32"
# CPU default. This box is avx2-only (no avx512_bf16, no AMX), so bf16 is emulated per-op: it
# halves memory and buys no speed, while changing the arithmetic. fp32 on 4B is ~16 GB resident.
# Declared in `config()` because it changes the decode, not just the runtime.

TORCH_NUM_THREADS = 8
# Also declared in `config()`, which looks like a violation of "don't declare knobs that cannot
# change the output" — device, batch size, log level. It isn't: on CPU the thread count changes
# the reduction ORDER inside matmuls, and a different order is different floating point, which
# can flip an argmax on a near-tie. A thread count is a device knob everywhere except here.


class Qwen3VLReader(Reader):
    """Qwen3-VL reading ONE crop at a time, greedily, entirely in-process.

    One crop per `read()` call and nothing else: the crop is the only pixels the model ever
    receives. That is the arm's definition (a reader is measured with detection held at zero)
    and it is also HIPAA minimum-necessary — the model never sees the frame the crop came from.
    """

    version_source = "transformers"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = REVISION,
        prompt: str = PROMPT_V1,
        prompt_id: str = PROMPT_ID,
        abstain_marker: str = ABSTAIN_MARKER,
        max_new_tokens: int = MAX_NEW_TOKENS,
        dtype: str = DTYPE,
        torch_num_threads: int = TORCH_NUM_THREADS,
        config_id: str = "stock",
    ) -> None:
        import torch
        import transformers

        self.model_id = model_id
        self.revision = revision
        self.prompt = prompt
        self.prompt_id = prompt_id
        self.abstain_marker = abstain_marker
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.torch_num_threads = torch_num_threads

        self.model_name = model_id.rsplit("/", 1)[-1].lower()
        # DERIVED from `model_id`, never a class constant. `model_id` is the knob that makes
        # this class serve the whole D-11.1 A/B (4B / 8B / RolmOCR / InternVL), and a hardcoded
        # name would stamp "qwen3-vl-4b-instruct" onto every row, table and report of an 8B run.
        # `aggregate()` would still refuse to blend them (the hashes differ), so the failure is
        # not a corrupted average — it is worse in one way: a correct number under a wrong name.

        self.config_id = config_id
        # Part of `aggregate()`'s identity tuple, so it is the lever a caller uses to declare
        # "this run is a different arm" without pretending a knob changed — the negative-control
        # script uses it to keep a floor run from ever pooling with an accuracy run.

        self.version = transformers.__version__
        # Read from the installed library, never typed (rule #9). `version_source` names the
        # same module and the shared contract test asserts the two agree.

        self.torch_version = torch.__version__
        # Also read, not typed. It goes in `config()` because fp32 CPU decode arithmetic is a
        # torch property: a torch bump can change a near-tie argmax with no other input moving,
        # and rule #9 says a result that could differ must not share an identity with one that
        # came before it.

        self._model: Any | None = None
        self._processor: Any | None = None
        # Loaded on first `read()`. Constructing a reader must stay cheap: the experiment CLI
        # builds every arm it might run in order to hash their identities, and a 16 GB load per
        # construction would make listing the arms cost more than running one.

    # -- identity -------------------------------------------------------------------------

    def config(self) -> dict[str, object]:
        """Every knob that can change what this reader outputs (D-13.5).

        `revision` is in here rather than in `version` because it is not any module's
        `__version__` — see the module docstring. `prompt` is in here in full, not as a
        digest of itself, so a diff of two runs' metadata shows what actually changed.

        `config_id` is deliberately NOT here: it is the arm's human label, already part of
        `aggregate()`'s identity tuple in its own right, and hashing a label would make two
        byte-identical configurations look like different measurements because someone renamed
        one of them.
        """
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "abstain_marker": self.abstain_marker,
            "max_new_tokens": self.max_new_tokens,
            "dtype": self.dtype,
            "torch_num_threads": self.torch_num_threads,
            "torch_version": self.torch_version,
            "do_sample": False,
            "num_beams": 1,
        }

    # -- model ----------------------------------------------------------------------------

    def _load(self) -> tuple[Any, Any]:
        """Load weights + processor from the local HF cache at the pinned revision.

        Local files only. `local_files_only=True` is not a performance choice: it makes a
        missing or moved checkpoint fail here, with a message telling a human to fetch it,
        instead of silently pulling whatever is at that revision mid-run.

        `torch.set_num_threads` is process-global, which is normally a reason not to call it
        from a library. It is called here anyway because the thread count is declared in
        `config()` (it changes CPU reduction order, hence the arithmetic): a declared knob that
        the code does not actually set would be a lie in the identity hash.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.set_num_threads(self.torch_num_threads)

        processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=getattr(torch, self.dtype),
            local_files_only=True,
        )
        model.eval()
        return model, processor

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            self._model, self._processor = self._load()
        return self._model, self._processor

    # -- the interface --------------------------------------------------------------------

    def _messages(self) -> list[dict[str, Any]]:
        """The one chat turn every read sends: one image part, then THE prompt. Never two.

        A separate method so a test can assert the chat template actually turns this into a
        vision placeholder. If it silently did not, the model would answer from the prompt text
        alone — every read would be invention, and every number the arm produced would be
        meaningless while looking completely normal.
        """
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
            }
        ]

    def read(self, crop: Image.Image) -> str:
        """Read one preprocessed crop. Greedy decode, no sampling, no second guess.

        Returns the model's decoded string as-is except for two things: surrounding whitespace
        (which `normalize()` would strip at scoring time anyway) and the abstain marker, which
        becomes `""` — an omission, which is a real answer.

        Deliberately NOTHING else is stripped. A model that answers "The text reads: CMFN-0042"
        scores as a misread, and on a KEEP token that is a false redaction — the headline
        metric. Peeling the preface off would be an undeclared knob that flatters this arm
        against the CRNN readers, which cannot be chatty in the first place. The chattiness is
        measured instead (`experiments/qwen3vl_cpu_timing.py` counts it).

        No confidence is extracted. A decode logprob is not a detection score, and nothing
        downstream could consume one without eventually sharing a threshold with docTR's
        detector confidence; `Reader.read` returns a bare string and this arm keeps it that way.
        """
        import torch

        model, processor = self._ensure_loaded()

        chat = processor.apply_chat_template(
            self._messages(), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat], images=[crop], return_tensors="pt")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy: the same crop must give the same string, always
                num_beams=1,
                # Explicit Nones override the checkpoint's own generation_config, which ships
                # sampling defaults. No seed is set anywhere and none is needed: with sampling
                # off there is no RNG in the path, and a seed would only imply otherwise.
                temperature=None,
                top_p=None,
                top_k=None,
            )

        prompt_len = inputs["input_ids"].shape[1]
        text = processor.decode(generated[0][prompt_len:], skip_special_tokens=True)

        if self._is_abstention(text):
            return ""
        return text.strip()

    def _is_abstention(self, text: str) -> bool:
        """True when the reply IS the abstain marker, allowing for near-miss decorations.

        Exact equality after `normalize()` is too strict in one direction that matters. An
        instruction-tuned model that abstains correctly still tends to punctuate: `"<<NOTEXT>>"`,
        `<<NOTEXT>>.`, ``` `<<NOTEXT>>` ```, `<< NOTEXT >>`. Each of those is the model doing
        exactly what it was told, and scoring it as a READ has an asymmetric cost — on a control
        box it is counted as INVENTED text, which inflates the hallucination floor, the number
        that gates whether this arm's accuracy may be quoted at all.

        So the comparison strips surrounding whitespace, quotes and backticks, drops trailing
        sentence punctuation, removes inner spaces, and casefolds. All of that applies ONLY to
        the marker test — a reply that is not the marker is returned untouched, so no real read
        is ever cleaned up (see `read`'s docstring on why chatty reads stay chatty).

        What this deliberately does NOT do is recognize free-form refusals ("There is no
        readable text."). Matching arbitrary prose would mean guessing at intent, and a wrong
        guess in that direction hides an invention instead of merely inflating the floor. Such
        replies are scored as reads, and the floor is therefore an UPPER bound — stated as such
        in `experiments/reader_negative_control.py`.
        """
        candidate = normalize(text).strip("\"'`“”‘’ \t").rstrip(".!;:")
        candidate = "".join(candidate.split())
        return candidate.casefold() == "".join(self.abstain_marker.split()).casefold()
