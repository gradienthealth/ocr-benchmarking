"""Phase 10e tests for ground_truth/build_gt.py — SYNTHETIC fixtures only.

Every image, token, UID and institution below is fabricated. No render, no `.dcm`, no real
review record, no real `manifest.csv` is touched: the whole world is built under `tmp_path`
from fake `CMFN-…`-style tokens (CLAUDE.md §5). `build_gt.py` is never run against real data
here — a human does that, and Claude reads only its PHI-free summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ground_truth.build_gt import BuildError, artifact_paths, build, main, make_parser

# --- the synthetic world ---------------------------------------------------------------

W, H = 640, 480

# image_id is 8 hex in the real pipeline; any stable opaque string works here.
IMG_MG_A = "aaaa1111"      # mg_2d, two tokens
IMG_MG_B = "bbbb2222"      # mg_2d, one token
IMG_CTRL_BLANK = "cccc3333"  # ct_axial, no tokens -> blank-control survivor
IMG_CTRL_TEXT = "dddd4444"   # ct_axial, HAS text -> leaves the control set
IMG_SCOUT = "eeee5555"     # ct_scout, no tokens -> a stratum with zero text-bearing images

ALL_IMAGES = [IMG_MG_A, IMG_MG_B, IMG_CTRL_BLANK, IMG_CTRL_TEXT, IMG_SCOUT]

STRATA = {
    IMG_MG_A: "mg_2d",
    IMG_MG_B: "mg_2d",
    IMG_CTRL_BLANK: "ct_axial",
    IMG_CTRL_TEXT: "ct_axial",
    IMG_SCOUT: "ct_scout",
}
VENDORS = {
    IMG_MG_A: "ACME_SCANNER",
    IMG_MG_B: "ACME_SCANNER",
    IMG_CTRL_BLANK: "BETACORP",
    IMG_CTRL_TEXT: "BETACORP",
    IMG_SCOUT: "BETACORP",
}
MODALITIES = {
    IMG_MG_A: "MG", IMG_MG_B: "MG",
    IMG_CTRL_BLANK: "CT", IMG_CTRL_TEXT: "CT", IMG_SCOUT: "CT",
}

# The frame the renderer actually produced. IMG_MG_B deliberately disagrees with the
# manifest's middle_frame_index below, which is the 9-of-199 case on the real data.
RENDERED_FRAME = {
    IMG_MG_A: 5, IMG_MG_B: 2, IMG_CTRL_BLANK: 10, IMG_CTRL_TEXT: 10, IMG_SCOUT: 0,
}
MANIFEST_FRAME = dict(RENDERED_FRAME, **{IMG_MG_B: 47})

# Fake tokens. Case-varied and whitespace-bearing on purpose: token_text is stored RAW.
TOKENS = {
    IMG_MG_A: [
        {"text": "CMFN-00421", "box": [10.0, 20.0, 110.0, 44.0], "label": "KEEP",
         "confidence": 91.0, "seed_index": 0},
        {"text": "acc-0099 ", "box": [10.0, 60.0, 120.0, 84.0], "label": "PHI",
         "confidence": 77.0, "seed_index": 1},
    ],
    IMG_MG_B: [
        {"text": "GRDN5678", "box": [5.0, 5.0, 95.0, 29.0], "label": "KEEP",
         "confidence": 88.0, "seed_index": 0},
    ],
    IMG_CTRL_BLANK: [],
    IMG_CTRL_TEXT: [
        {"text": "L", "box": [3.0, 3.0, 15.0, 21.0], "label": "KEEP",
         "confidence": 60.0, "seed_index": None},
    ],
    IMG_SCOUT: [],
}

_MANIFEST_COLUMNS = [
    "series_uid", "modality", "strata", "manufacturer", "manufacturer_model",
    "body_part", "series_description", "number_of_frames", "middle_frame_index",
    "photometric_interpretation", "sop_class_uid", "study_uid", "institution_name",
    "study_date", "source_tar_path", "dest_tar_path",
]


def uid_for(image_id: str) -> str:
    return f"1.2.3.FAKE.SERIES.{image_id}"


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(header))
        writer.writerows(rows)


class World:
    """A complete synthetic 10b/10c/10d output tree plus a fake manifest."""

    def __init__(self, root: Path, images: list[str]):
        self.root = root
        self.images = list(images)
        self.renders = root / "renders"
        self.seed = root / "seed"
        self.gt_dir = root / "ground_truth"
        self.review = self.gt_dir / "review"
        self.round2 = self.gt_dir / "review_r2"
        for d in (self.renders, self.seed, self.review, self.gt_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.out = self.gt_dir / "gt.csv"
        self.groups = root / "groups.csv"
        self.backmap = root / "backmap.csv"
        self.manifest = root / "manifest.csv"
        self.drawn = root / "drawn.csv"
        self.write_all()

    # -- writers (each callable again after mutation) -------------------------------

    def write_all(self) -> None:
        self.write_renders()
        self.write_seed()
        self.write_reviews()
        self.write_groups()
        self.write_backmap()
        self.write_manifest()
        self.write_drawn(len(self.images))

    def write_renders(self) -> None:
        _write_csv(
            self.renders / "render_manifest.csv",
            ("image_id", "frame_idx", "w", "h", "sha256", "fallback_used"),
            [[i, RENDERED_FRAME[i], W, H, hashlib.sha256(i.encode()).hexdigest(), 1]
             for i in self.images],
        )
        for i in self.images:
            (self.renders / f"{i}.png").write_bytes(b"")  # presence only; never opened

    def write_seed(self) -> None:
        for i in self.images:
            seed_tokens = [
                {"text": t["text"], "bbox": list(t["box"]),
                 "confidence": t["confidence"], "label": "PHI"}
                for t in TOKENS[i] if t["seed_index"] is not None
            ]
            (self.seed / f"{i}.json").write_text(json.dumps({
                "image_id": i, "w": W, "h": H,
                "provenance": {"tesseract_version": "0.0.0-synthetic"},
                "tokens": seed_tokens,
            }), encoding="utf-8")

    def write_reviews(self, states: dict[str, str] | None = None) -> None:
        states = states or {}
        for i in self.images:
            self.write_review(i, state=states.get(i, "edited"))

    def write_review(self, image_id: str, *, state: str = "edited",
                     tokens: list[dict] | None = None, directory: Path | None = None) -> None:
        record = {
            "image_id": image_id,
            "state": state,
            "round": 1,
            "timestamp": "2026-08-09T00:00:00Z",
            "gate": {"conf": 0.0, "len": 1, "alnum": False},
            "tokens": [] if state == "deferred"
                      else (TOKENS[image_id] if tokens is None else tokens),
            "deleted_seed_indexes": [],
        }
        if state == "deferred":
            record["defer_reason"] = "ambiguous_token"
        (directory or self.review).mkdir(parents=True, exist_ok=True)
        (directory or self.review).joinpath(f"{image_id}.json").write_text(
            json.dumps(record), encoding="utf-8")

    def write_groups(self) -> None:
        _write_csv(self.groups, ("image_id", "group"),
                   [[i, STRATA[i]] for i in self.images])

    def write_backmap(self) -> None:
        _write_csv(self.backmap, ("image_id", "series_uid", "sop_instance_uid",
                                  "src_frame_idx", "id_len"),
                   [[i, uid_for(i), f"1.2.3.FAKE.SOP.{i}", RENDERED_FRAME[i], 8]
                    for i in self.images])

    def write_manifest(self, extra_rows: int = 0) -> None:
        rows = [
            [uid_for(i), MODALITIES[i], STRATA[i], VENDORS[i], "M1", "CHEST", "desc",
             "60", str(MANIFEST_FRAME[i]), "MONOCHROME2", "1.2.840.FAKE.SOPCLASS",
             "1.2.3.FAKE.STUDY", "FAKE_GENERAL_HOSPITAL", "1999-12-31",
             "gs://x/s.tar", "gs://y/d.tar"]
            for i in self.images
        ]
        for n in range(extra_rows):
            rows.append(
                [f"1.2.3.FAKE.SERIES.extra{n}", "CT", "ct_axial", "BETACORP", "M1",
                 "CHEST", "desc", "60", "30", "MONOCHROME2", "1.2.840.FAKE.SOPCLASS",
                 "1.2.3.FAKE.STUDY", "FAKE_GENERAL_HOSPITAL", "1999-12-31",
                 "gs://x/s.tar", "gs://y/d.tar"])
        _write_csv(self.manifest, _MANIFEST_COLUMNS, rows)

    def write_drawn(self, n: int) -> None:
        _write_csv(self.drawn, ("series_uid",),
                   [[f"1.2.3.FAKE.SERIES.drawn{k}"] for k in range(n)])

    # -- running --------------------------------------------------------------------

    def argv(self) -> list[str]:
        return [
            "--set", f"{self.renders}:{self.seed}",
            "--groups", str(self.groups),
            "--backmap", str(self.backmap),
            "--manifest", str(self.manifest),
            "--drawn", str(self.drawn),
            "--out", str(self.out),
            "--review-dir", str(self.review),
            "--round2-dir", str(self.round2),
            # Required (no default): it names a COMMITTED per-scope artifact that the build
            # overwrites in place, so it is spelled out on every invocation.
            "--text-presence-name", "text_presence_v2.csv",
        ]

    def run(self) -> str:
        return build(make_parser().parse_args(self.argv()))

    def rows(self) -> list[dict]:
        with self.out.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def presence(self) -> dict[str, str]:
        path = self.gt_dir / "text_presence_v2.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            return {r["image_id"]: r["has_text"] for r in csv.DictReader(fh)}


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path, ALL_IMAGES)


# --- the gate ---------------------------------------------------------------------------


def test_missing_review_record_fails_and_writes_nothing(world):
    (world.review / f"{IMG_MG_A}.json").unlink()
    with pytest.raises(BuildError, match="NO review record"):
        world.run()
    assert not world.out.exists()
    assert not world.out.with_suffix(".csv.part").exists()


def test_deferred_review_record_fails(world):
    world.write_review(IMG_MG_A, state="deferred")
    with pytest.raises(BuildError, match="deferred"):
        world.run()
    assert not world.out.exists()


def test_orphan_review_record_fails(world):
    """The bidirectional half: a record for an image outside the scored set."""
    world.write_review(IMG_MG_A, state="edited")
    (world.review / "ffff9999.json").write_text(json.dumps({
        "image_id": "ffff9999", "state": "edited", "round": 1, "timestamp": "t",
        "gate": {"conf": 0.0}, "tokens": [], "deleted_seed_indexes": [],
    }), encoding="utf-8")
    with pytest.raises(BuildError, match="NOT in the scored set"):
        world.run()
    assert not world.out.exists()


def test_part_file_is_not_parsed_as_a_record(world):
    """A killed 10d save leaves `<id>.json.<pid>.<tid>.part`; the glob must ignore it."""
    (world.review / f"{IMG_MG_A}.json.123.456.part").write_text("{ truncated",
                                                                encoding="utf-8")
    world.run()
    assert world.out.exists()


def test_record_whose_inner_image_id_disagrees_with_its_filename_fails(world):
    """A copied or renamed record must not pass the gate.

    Without this check the gate sees no missing, no orphan and no deferred record, and
    `gt.csv` gets one image's tokens joined to another image's series_uid and vendor —
    in-bounds boxes, so 10a validates clean and both hashes look healthy.
    """
    stolen = json.loads((world.review / f"{IMG_MG_A}.json").read_text())
    (world.review / f"{IMG_MG_B}.json").write_text(json.dumps(stolen), encoding="utf-8")
    with pytest.raises(BuildError, match="does not\n?\\s*match its filename|declares image_id"):
        world.run()
    assert not world.out.exists()


def test_seed_quality_block_follows_review_dir(world, tmp_path):
    """`collect()` resolves records through a module global; --review-dir must win.

    Otherwise the provenance block silently describes a different set of records than the
    one that produced gt.csv — all zeros if that directory is empty, or a stale batch's
    numbers presented as this build's provenance.
    """
    summary = world.run()
    # The tell: resolved against the wrong directory these records are simply not found,
    # so every image lands in `unreviewed` and both edit counters read 0.
    assert "images    5  accepted    0  edited    5  deferred   0  unreviewed   0" in summary
    import ground_truth.review_gt as review_gt_module
    assert review_gt_module.REVIEW_DIR != world.review   # global restored afterwards


def test_unexpected_exception_message_is_withheld(world, capsys, monkeypatch):
    """An exception message can quote a field value, so only the type is printed."""
    import ground_truth.build_gt as build_gt_module

    def boom(*_a, **_k):
        raise RuntimeError("token_text=CMFN-00421 blew up")

    monkeypatch.setattr(build_gt_module, "build", boom)
    assert main(world.argv()) == 1
    err = capsys.readouterr().err
    assert "unexpected RuntimeError" in err
    assert "CMFN-00421" not in err


def test_setup_error_is_caught_not_raised_as_a_traceback(world, capsys):
    (world.renders / f"{IMG_MG_A}.png").unlink()
    assert main(world.argv()) == 1
    assert "SetupError" in capsys.readouterr().err


def test_exit_code_is_nonzero_on_a_failed_gate(world, capsys):
    (world.review / f"{IMG_MG_A}.json").unlink()
    assert main(world.argv()) == 1
    assert not world.out.exists()


# --- reconciliation ---------------------------------------------------------------------


def test_drawn_but_unrendered_is_reported_and_not_fatal(world):
    world.write_drawn(len(ALL_IMAGES) + 3)
    summary = world.run()
    assert world.out.exists()
    assert f"drawn {len(ALL_IMAGES) + 3}" in summary
    assert "3 drawn series never rendered" in summary


def test_frame_idx_comes_from_the_render_manifest(world):
    summary = world.run()
    row = next(r for r in world.rows() if r["image_id"] == IMG_MG_B)
    assert row["frame_idx"] == str(RENDERED_FRAME[IMG_MG_B]) != str(MANIFEST_FRAME[IMG_MG_B])
    assert "frame_idx disagreements vs manifest.middle_frame_index: 1" in summary


def test_manifest_counts_are_printed_from_the_loaded_file(world):
    world.write_manifest(extra_rows=4)
    summary = world.run()
    assert f"rows {len(ALL_IMAGES) + 4}" in summary


# --- rows -------------------------------------------------------------------------------


def test_vendor_stratum_modality_come_from_the_manifest_adapter(world):
    """Not from the filename, not from the review JSON (CLAUDE.md rule #7)."""
    world.write_review(IMG_MG_A, tokens=[
        dict(TOKENS[IMG_MG_A][0], **{"vendor": "WRONGCORP", "stratum": "wrong_stratum"}),
    ])
    row = (world.run(), world.rows())[1]
    row = next(r for r in row if r["image_id"] == IMG_MG_A)
    assert (row["vendor"], row["stratum"], row["modality"]) == ("ACME_SCANNER", "mg_2d", "MG")


def test_token_text_round_trips_byte_identical(world):
    world.run()
    texts = {r["image_id"]: r["token_text"] for r in world.rows()}
    assert texts[IMG_MG_B] == "GRDN5678"
    raw = [r["token_text"] for r in world.rows() if r["image_id"] == IMG_MG_A]
    # Case variation and the trailing space both survive: normalize() is applied at
    # match time, never here.
    assert raw == ["CMFN-00421", "acc-0099 "]


def test_invalid_label_is_rejected(world):
    world.write_review(IMG_MG_A, tokens=[dict(TOKENS[IMG_MG_A][0], label="OTHER")])
    with pytest.raises(BuildError, match="frozen 10a spec"):
        world.run()
    assert not world.out.exists()


def test_part_candidate_is_removed_when_validation_refuses(world):
    """The pre-validation `.part` holds the FULL table — it must not survive a refusal."""
    world.write_review(IMG_MG_A, tokens=[dict(TOKENS[IMG_MG_A][0], label="OTHER")])
    with pytest.raises(BuildError, match="frozen 10a spec"):
        world.run()
    assert not world.out.exists()
    assert not Path(str(world.out) + ".part").exists()


def test_part_candidate_is_removed_when_the_validator_itself_raises(world, monkeypatch):
    """`finally`, not the failure branch: a raising validator left the file behind before."""
    import ground_truth.build_gt as build_gt_module

    def boom(*_a, **_k):
        raise RuntimeError("validator died mid-check")

    monkeypatch.setattr(build_gt_module, "validate_gt", boom)
    with pytest.raises(RuntimeError):
        world.run()
    assert not Path(str(world.out) + ".part").exists()


def test_a_successful_build_leaves_no_part_behind(world):
    world.run()
    assert world.out.exists()
    assert not Path(str(world.out) + ".part").exists()


def test_text_presence_name_is_required(world):
    """No default: forgetting it must not silently overwrite another scope's denominator."""
    argv = [a for a in world.argv() if a not in ("--text-presence-name",
                                                 "text_presence_v2.csv")]
    with pytest.raises(SystemExit) as exc:
        make_parser().parse_args(argv)
    assert exc.value.code == 2


def _with_presence_name(world, name: str) -> list[str]:
    argv = world.argv()
    argv[argv.index("text_presence_v2.csv")] = name
    return argv


@pytest.mark.parametrize("name", ["../escaped.csv", "/tmp/escaped.csv", "sub/x.csv", ""])
def test_text_presence_name_rejects_anything_but_a_bare_filename(world, name):
    with pytest.raises(BuildError, match="bare filename"):
        build(make_parser().parse_args(_with_presence_name(world, name)))


@pytest.mark.parametrize("name", ["gt.csv", "gt.csv.sha256", "gt_set.sha256",
                                  "gt_summary.txt"])
def test_text_presence_name_may_not_collide_with_an_artifact(world, name):
    """It is written last; colliding would overwrite a just-frozen file after its hash."""
    with pytest.raises(BuildError, match="collides"):
        build(make_parser().parse_args(_with_presence_name(world, name)))


def test_a_rejected_presence_name_writes_nothing_at_all(world):
    """The check must precede write_gt: otherwise gt.csv + both hashes are already replaced,
    and the corrected re-run reads its own new hash as `prev` and prints no invalidation."""
    with pytest.raises(BuildError):
        build(make_parser().parse_args(_with_presence_name(world, "../escaped.csv")))
    assert not world.out.exists()
    assert not Path(str(world.out) + ".sha256").exists()
    assert not (world.gt_dir / "gt_set.sha256").exists()
    assert not (world.gt_dir / "gt_summary.txt").exists()
    assert not (world.gt_dir.parent / "escaped.csv").exists()


def test_summary_names_the_text_presence_file(world):
    """Otherwise nothing afterwards records which denominator this build rewrote."""
    assert "text_presence_v2.csv" in world.run()


def test_non_numeric_render_geometry_is_refused_without_echoing_the_value(world):
    """`int("...")` quotes what it choked on, and main() prints ValueError verbatim."""
    path = world.renders / "render_manifest.csv"
    text = path.read_text(encoding="utf-8").replace(",0,", ",CMFN-00421,", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BuildError) as exc:
        world.run()
    assert "CMFN-00421" not in str(exc.value)
    assert "non-integer frame_idx/w/h" in str(exc.value)


def test_non_numeric_token_box_is_refused_without_echoing_the_value(world):
    """A column-shifted record can put token text where a coordinate belongs."""
    bad = dict(TOKENS[IMG_MG_A][0], box=["CMFN-00421", 1.0, 2.0, 3.0])
    world.write_review(IMG_MG_A, tokens=[bad])
    with pytest.raises(BuildError) as exc:
        world.run()
    assert "CMFN-00421" not in str(exc.value)
    assert "token box that is not four numbers" in str(exc.value)


def test_columns_match_the_frozen_spec(world):
    from ground_truth.gt_schema import COLUMNS
    world.run()
    with world.out.open(newline="", encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == COLUMNS


# --- blank controls ---------------------------------------------------------------------


def test_blank_control_contributes_zero_rows(world):
    world.run()
    assert not [r for r in world.rows() if r["image_id"] == IMG_CTRL_BLANK]


def test_control_with_text_gets_rows_and_leaves_the_control_set(world):
    summary = world.run()
    assert [r for r in world.rows() if r["image_id"] == IMG_CTRL_TEXT]
    # 2 ct_axial images, 1 of which has text -> 1 survivor.
    assert "blank-control survivors (ct_axial with zero rows): 1 of 2" in summary
    assert IMG_CTRL_TEXT in summary


def test_single_character_token_survives(world):
    """`L` is real burned-in text and a misread of it is a false redaction."""
    world.run()
    assert [r for r in world.rows()
            if r["image_id"] == IMG_CTRL_TEXT and r["token_text"] == "L"]


# --- text_presence + zero-text warning ---------------------------------------------------


def test_text_presence_lists_every_scored_image(world):
    world.run()
    presence = world.presence()
    assert set(presence) == set(ALL_IMAGES)
    assert presence[IMG_MG_A] == "1"
    assert presence[IMG_CTRL_BLANK] == "0"
    assert presence[IMG_SCOUT] == "0"


def test_zero_text_stratum_produces_the_named_warning(world):
    summary = world.run()
    assert "WARNING  stratum 'ct_scout' has NO text-bearing images" in summary
    assert "'mg_2d'" not in summary.split("WARNING")[-1].split("\n")[0]


# --- hashes -----------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text().strip()


def test_content_hash_matches_the_written_file(world):
    world.run()
    expected = hashlib.sha256(world.out.read_bytes()).hexdigest()
    assert _read(Path(str(world.out) + ".sha256")) == expected
    assert len(expected) == 64 and expected == expected.lower()


def test_rerun_over_unchanged_inputs_reproduces_both_hashes(world):
    world.run()
    gt_hash = _read(Path(str(world.out) + ".sha256"))
    set_hash = _read(world.gt_dir / "gt_set.sha256")
    first_bytes = world.out.read_bytes()
    world.run()
    assert world.out.read_bytes() == first_bytes
    assert _read(Path(str(world.out) + ".sha256")) == gt_hash
    assert _read(world.gt_dir / "gt_set.sha256") == set_hash


def test_set_hash_changes_when_a_blank_image_joins_but_gt_csv_does_not(tmp_path):
    """The case that motivates a second hash: scope moved, contents did not."""
    smaller = World(tmp_path / "a", [IMG_MG_A, IMG_MG_B])
    smaller.run()
    bytes_small = smaller.out.read_bytes()
    set_small = _read(smaller.gt_dir / "gt_set.sha256")

    bigger = World(tmp_path / "b", [IMG_MG_A, IMG_MG_B, IMG_CTRL_BLANK])
    bigger.run()

    assert bigger.out.read_bytes() == bytes_small          # identical contents
    assert _read(bigger.gt_dir / "gt_set.sha256") != set_small   # different scope


def test_changed_hash_is_surfaced_as_an_invalidation(world):
    world.run()
    world.write_review(IMG_MG_B, tokens=[dict(TOKENS[IMG_MG_B][0], text="CMFN-99999")])
    summary = world.run()
    assert "EVERY NUMBER PREVIOUSLY REPORTED AGAINST THIS gt.csv IS VOID" in summary
    assert "gt.csv.sha256" in summary


# --- D-10.7 self-agreement ---------------------------------------------------------------


def test_absent_round2_prints_the_sentence_and_no_number(world):
    summary = world.run()
    assert "self-agreement: not yet run (round 2 absent)" in summary
    assert "text agreement" not in summary


def test_empty_round2_dir_also_prints_the_sentence(world):
    world.round2.mkdir(parents=True, exist_ok=True)
    summary = world.run()
    assert "self-agreement: not yet run (round 2 absent)" in summary


def test_round2_present_produces_a_computed_figure(world):
    # Same boxes, one text disagreement and one label disagreement.
    world.write_review(IMG_MG_A, directory=world.round2, tokens=[
        dict(TOKENS[IMG_MG_A][0], text="CMFN-00427"),
        dict(TOKENS[IMG_MG_A][1], label="KEEP"),
    ])
    summary = world.run()
    assert "tokens paired at IoU>=0.5 2" in summary
    assert "text agreement   1/2" in summary
    assert "label agreement  1/2" in summary
    assert "round-1 only 0   round-2 only 0" in summary


def test_deferred_round2_record_is_excluded_not_counted_as_disagreement(world):
    """An undecided re-review has an empty token list; counting it would turn every
    round-1 token into a miss and quietly depress the agreement figure."""
    world.write_review(IMG_MG_A, directory=world.round2, tokens=[
        dict(TOKENS[IMG_MG_A][0]), dict(TOKENS[IMG_MG_A][1]),
    ])
    world.write_review(IMG_MG_B, directory=world.round2, state="deferred")
    summary = world.run()
    assert "tokens paired at IoU>=0.5 2" in summary
    assert "text agreement   2/2" in summary
    assert "1 round-2 record(s) are DEFERRED and excluded" in summary


def test_round_distribution_is_reported(world):
    summary = world.run()
    assert "annotation rounds in the review dir: round 1: 5" in summary


def test_more_rendered_than_drawn_is_reported_without_a_negative_count(world):
    world.write_drawn(2)
    summary = world.run()
    assert "-" not in summary.split("MORE images rendered")[0].split("NOTE")[-1]
    assert "3 MORE images rendered than the drawn file lists" in summary


def test_round2_never_feeds_gt_csv(world):
    world.write_review(IMG_MG_A, directory=world.round2, tokens=[
        {"text": "ROUND2-ONLY", "box": [200.0, 200.0, 300.0, 224.0], "label": "PHI",
         "confidence": 50.0, "seed_index": None},
    ])
    world.run()
    assert not [r for r in world.rows() if r["token_text"] == "ROUND2-ONLY"]


# --- summary is PHI-free -----------------------------------------------------------------


def test_summary_carries_no_token_text_and_no_uid(world):
    summary = world.run()
    for token in ("CMFN-00421", "acc-0099", "GRDN5678"):
        assert token not in summary
    assert "1.2.3.FAKE.SERIES" not in summary
    assert "FAKE_GENERAL_HOSPITAL" not in summary
    assert "1999-12-31" not in summary


def test_summary_is_written_to_a_file_as_well_as_returned(world):
    summary = world.run()
    assert (world.gt_dir / "gt_summary.txt").read_text().strip() == summary.strip()


# --- Phase 13h / D-13.4: a second artifact must not clobber the first --------------------
# The dev slice is built by THIS script with a different `--out`. Until 13h the three
# sibling names were hardcoded, so building it would have overwritten gt_v1's frozen scope
# hash and summary — a rule-#8 invalidation performed by a tool rather than by a human.


def test_artifact_paths_reproduce_the_historical_gt_names_exactly():
    """The default build's four filenames are unchanged. gt_v1's artifacts must not churn."""
    gt_hash, set_hash, summary = artifact_paths(Path("ground_truth/gt.csv"))
    assert gt_hash.name == "gt.csv.sha256"
    assert set_hash.name == "gt_set.sha256"
    assert summary.name == "gt_summary.txt"


def test_artifact_paths_are_derived_from_out():
    gt_hash, set_hash, summary = artifact_paths(Path("ground_truth/dev_v1.csv"))
    assert gt_hash.name == "dev_v1.csv.sha256"
    assert set_hash.name == "dev_v1_set.sha256"
    assert summary.name == "dev_v1_summary.txt"


def test_artifact_paths_handles_an_out_without_a_csv_suffix():
    _, set_hash, summary = artifact_paths(Path("ground_truth/devslice"))
    assert set_hash.name == "devslice_set.sha256"
    assert summary.name == "devslice_summary.txt"


def test_a_dev_slice_build_does_not_touch_the_scored_sets_artifacts(world):
    """THE regression this fix exists for: build gt.csv, then build a dev slice beside it.

    Both hash files and both summaries must survive, holding their own values. If the dev
    build overwrote `gt_set.sha256`, every number reported against gt_v1 would be scored
    against a scope hash that no longer describes it — and nothing would say so.
    """
    world.run()
    frozen = {
        p.name: p.read_bytes()
        for p in (world.gt_dir / "gt.csv.sha256",
                  world.gt_dir / "gt_set.sha256",
                  world.gt_dir / "gt_summary.txt")
    }

    dev_argv = [a for a in world.argv()]
    dev_argv[dev_argv.index("--out") + 1] = str(world.gt_dir / "dev_v1.csv")
    dev_argv[dev_argv.index("--text-presence-name") + 1] = "text_presence_dev_v1.csv"
    build(make_parser().parse_args(dev_argv))

    for name, content in frozen.items():
        assert (world.gt_dir / name).read_bytes() == content, f"{name} was clobbered"
    assert (world.gt_dir / "dev_v1_set.sha256").is_file()
    assert (world.gt_dir / "dev_v1_summary.txt").is_file()


def test_dev_slice_summary_names_itself_not_gt_csv(world):
    """A dev-slice summary that says "gt.csv" would be filed as the scored set's."""
    dev_argv = [a for a in world.argv()]
    dev_argv[dev_argv.index("--out") + 1] = str(world.gt_dir / "dev_v1.csv")
    dev_argv[dev_argv.index("--text-presence-name") + 1] = "text_presence_dev_v1.csv"
    summary = build(make_parser().parse_args(dev_argv))
    assert "dev_v1.csv sha256" in summary
    assert "dev_v1_set.sha256" in summary
    assert "Phase 10e — gt.csv build summary" not in summary


def test_text_presence_name_may_not_collide_with_the_dev_slice_artifacts(world):
    """The reserved set tracks --out: a dev build reserves ITS names, not gt.csv's."""
    dev_argv = [a for a in world.argv()]
    dev_argv[dev_argv.index("--out") + 1] = str(world.gt_dir / "dev_v1.csv")
    dev_argv[dev_argv.index("--text-presence-name") + 1] = "dev_v1_set.sha256"
    with pytest.raises(BuildError, match="collides"):
        build(make_parser().parse_args(dev_argv))
