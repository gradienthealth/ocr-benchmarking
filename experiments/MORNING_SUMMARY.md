# Morning summary — assembled 2026-08-11T16:26:17Z

Everything below is PHI-free aggregate output. Source of truth for slides 15-17.
Deck must be off this workstation before its 09:21 local hard shutdown.

## Run status — read this first

- `results/gt_v1_reader_parseq_20260811_1621` — status **COMPLETE**, TABLES.md present
- `results/gt_v1_reader_svtrv2_20260811_1616` — status **COMPLETE**, TABLES.md present
- `results/gt_v1_easyocr_tuned_20260811_1508` — status **COMPLETE**, TABLES.md present
- `results/gt_v1_paddle_20260811_1340` — status **COMPLETE**, TABLES.md present
- `results/gt_v1_paddle_20260811_0812` — status **RUNNING**, TABLES.md MISSING
- `results/gt_v1_easyocr_20260811_0812` — status **COMPLETE**, TABLES.md present
- `results/gt_v1_doctr_20260811_0812` — status **COMPLETE**, TABLES.md present
- `results/preflight_doctr_20260811_080703` — status **COMPLETE** (preflight docTR-only insurance run)

A status of COMPLETE means its tables were written. INCOMPLETE means the run died
partway and that directory has NO table — run_experiment.py writes all-or-nothing.


---

# FROM: results/gt_v1_reader_parseq_20260811_1621

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T16:21:11+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

_(no end-to-end arm ran)_

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**


## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

_(no end-to-end arm ran)_

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

| Arm                 | Config                | Version | Images | GT tokens | Read | Empty | FR rate | KEEP exact | Control crops | Inventions | Invention/crop | Cost (USD) | Cost basis         |
| ------------------- | --------------------- | ------- | ------ | --------- | ---- | ----- | ------- | ---------- | ------------- | ---------- | -------------- | ---------- | ------------------ |
| reader:doctr-parseq | parseq [f21c41a269b9] | v1.0.1  | 102    | 2351      | 2350 | 1     | 18.04%  | 81.96%     | 388           | 242        | 62.37%         | $0.000000  | estimated (pixels) |

---

# FROM: results/gt_v1_reader_svtrv2_20260811_1616

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T16:17:00+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

_(no end-to-end arm ran)_

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**


## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

_(no end-to-end arm ran)_

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

| Arm           | Config                | Version | Images | GT tokens | Read | Empty | FR rate | KEEP exact | Control crops | Inventions | Invention/crop | Cost (USD) | Cost basis         |
| ------------- | --------------------- | ------- | ------ | --------- | ---- | ----- | ------- | ---------- | ------------- | ---------- | -------------- | ---------- | ------------------ |
| reader:svtrv2 | server [c9e1258be513] | 0.1.5   | 102    | 2351      | 2334 | 17    | 19.21%  | 80.79%     | 388           | 34         | 8.76%          | $0.000000  | estimated (pixels) |

---

# FROM: results/gt_v1_easyocr_tuned_20260811_1508

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T15:08:09+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

| Arm           | Config                | Version | Images | FR rate | KEEP exact | Found | Omission | Added | Latency p95 (s) | Cost (USD) | Cost basis         |
| ------------- | --------------------- | ------- | ------ | ------- | ---------- | ----- | -------- | ----- | --------------- | ---------- | ------------------ |
| easyocr:tuned | thr0.2 [741280cbf20c] | 1.7.2   | 199    | 94.22%  | 5.78%      | 210   | 2141     | 1310  | 22.6809         | $0.000000  | estimated (pixels) |

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**

- **easyocr:tuned** — floor: 48 invented word(s) over 97 confirmed-blank frame(s), 0.495 per image.

## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

| Arm           | IoU | Text imgs | Blank | GT tokens | Found | Missed | Recall | Boxes | Added | Precision | Floor boxes/img |
| ------------- | --- | --------- | ----- | --------- | ----- | ------ | ------ | ----- | ----- | --------- | --------------- |
| easyocr:tuned | 0.5 | 102       | 97    | 2351      | 210   | 2141   | 8.93%  | 1472  | 1262  | 14.27%    | 0.495           |

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

_(no reader arm ran)_

---

# FROM: results/gt_v1_paddle_20260811_1340

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T13:40:49+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

| Arm                   | Config               | Version | Images | FR rate | KEEP exact | Found | Omission | Added | Latency p95 (s) | Cost (USD) | Cost basis         |
| --------------------- | -------------------- | ------- | ------ | ------- | ---------- | ----- | -------- | ----- | --------------- | ---------- | ------------------ |
| pp-ocrv6_medium:stock | stock [9b0d1bec510b] | 3.7.0   | 199    | 61.43%  | 38.57%     | 1075  | 1276     | 2502  | 116.9391        | $0.000000  | estimated (pixels) |

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**

- **pp-ocrv6_medium:stock** — floor: 44 invented word(s) over 97 confirmed-blank frame(s), 0.454 per image.

## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

| Arm                   | IoU | Text imgs | Blank | GT tokens | Found | Missed | Recall | Boxes | Added | Precision | Floor boxes/img |
| --------------------- | --- | --------- | ----- | --------- | ----- | ------ | ------ | ----- | ----- | --------- | --------------- |
| pp-ocrv6_medium:stock | 0.5 | 102       | 97    | 2351      | 1075  | 1276   | 45.73% | 3533  | 2458  | 30.43%    | 0.454           |

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

_(no reader arm ran)_

---

# FROM: results/gt_v1_easyocr_20260811_0812

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T08:17:34+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

| Arm           | Config               | Version | Images | FR rate | KEEP exact | Found | Omission | Added | Latency p95 (s) | Cost (USD) | Cost basis         |
| ------------- | -------------------- | ------- | ------ | ------- | ---------- | ----- | -------- | ----- | --------------- | ---------- | ------------------ |
| easyocr:stock | stock [f24d5b06a787] | 1.7.2   | 199    | 76.03%  | 23.97%     | 818   | 1533     | 707   | 24.6888         | $0.000000  | estimated (pixels) |

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**

- **easyocr:stock** — floor: 7 invented word(s) over 97 confirmed-blank frame(s), 0.072 per image.

## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

| Arm           | IoU | Text imgs | Blank | GT tokens | Found | Missed | Recall | Boxes | Added | Precision | Floor boxes/img |
| ------------- | --- | --------- | ----- | --------- | ----- | ------ | ------ | ----- | ----- | --------- | --------------- |
| easyocr:stock | 0.5 | 102       | 97    | 2351      | 818   | 1533   | 34.79% | 1518  | 700   | 53.89%    | 0.072           |

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

_(no reader arm ran)_

---

# FROM: results/gt_v1_doctr_20260811_0812

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T08:12:22+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

| Arm         | Config               | Version | Images | FR rate | KEEP exact | Found | Omission | Added | Latency p95 (s) | Cost (USD) | Cost basis         |
| ----------- | -------------------- | ------- | ------ | ------- | ---------- | ----- | -------- | ----- | --------------- | ---------- | ------------------ |
| doctr:stock | stock [a1df589d1b52] | v1.0.1  | 199    | 39.53%  | 60.47%     | 1757  | 594      | 605   | 2.2324          | $0.000000  | estimated (pixels) |

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**

- **doctr:stock** — floor: 31 invented word(s) over 97 confirmed-blank frame(s), 0.320 per image.

## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

| Arm         | IoU | Text imgs | Blank | GT tokens | Found | Missed | Recall | Boxes | Added | Precision | Floor boxes/img |
| ----------- | --- | --------- | ----- | --------- | ----- | ------ | ------ | ----- | ----- | --------- | --------------- |
| doctr:stock | 0.5 | 102       | 97    | 2351      | 1757  | 594    | 74.73% | 2331  | 574   | 75.38%    | 0.320           |

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

_(no reader arm ran)_

---

# FROM: results/preflight_doctr_20260811_080703

# OCR benchmark — gt_v1

| Artifact identity        | Value                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Dataset                  | gt_v1                                                            |
| gt.csv sha256            | afc6598990eb5e0a2790e2aaa0ee962172c1ca3be600ab0b3e500b912d94759e |
| Scored-set sha256        | e0bf95035e036c02ee80e75065ea5190a53821bfd043facd9582f373880435ea |
| Images scored            | 199                                                              |
|   of which text-bearing  | 102                                                              |
|   of which blank control | 97                                                               |
| GT tokens                | 2351                                                             |
| Generated at             | 2026-08-11T08:07:08+00:00                                        |

> **Three questions, three tables, never blended.** Table A ranks engines end to end; Table B asks only whether the text was *found*; Table C scores *reading* under an oracle detector. A number from one table must never be quoted against another's.

## Table A — end-to-end (find + read)

The pipeline question: did the engine locate the token *and* read it correctly. *Found*/*Omission* is the omission axis, *Added* the hallucination axis — two quantities, **never averaged**. False-redaction rate is plan.md's headline; KEEP exact-match is the reading-quality metric the current PM focus ranks on. Both are shown; neither is folded into the other.

| Arm         | Config               | Version | Images | FR rate | KEEP exact | Found | Omission | Added | Latency p95 (s) | Cost (USD) | Cost basis         |
| ----------- | -------------------- | ------- | ------ | ------- | ---------- | ----- | -------- | ----- | --------------- | ---------- | ------------------ |
| doctr:stock | stock [a1df589d1b52] | v1.0.1  | 199    | 39.53%  | 60.47%     | 1757  | 594      | 605   | 2.2782          | $0.000000  | estimated (pixels) |

**Negative control (step 2) — measured in the same pass, since a blank frame is already a negative control for an engine that draws its own boxes:**

- **doctr:stock** — floor: 31 invented word(s) over 97 confirmed-blank frame(s), 0.320 per image.

## Table B — detector-only (find, strings ignored)

A different question from Table A, on the same rows: did the engine *find* the text at all, regardless of whether it read it. Matched by the same matcher at the same IoU bar, so a loss here is a detection loss, not a reading one. **Never merged with Table A** — a detection recall and an end-to-end rate have different denominators. Reader arms do not appear: they are handed the GT box, run no matcher, and would print a saturated 100% recall that measures nothing.

| Arm         | IoU | Text imgs | Blank | GT tokens | Found | Missed | Recall | Boxes | Added | Precision | Floor boxes/img |
| ----------- | --- | --------- | ----- | --------- | ----- | ------ | ------ | ----- | ----- | --------- | --------------- |
| doctr:stock | 0.5 | 102       | 97    | 2351      | 1757  | 594    | 74.73% | 2331  | 574   | 75.38%    | 0.320           |

## Table C — reader arm (ORACLE boxes, string only)

**Not comparable to Table A.** Every reader here is handed `gt_v1`'s human-drawn box, so detection error is held at zero by construction and detection metrics are undefined for this arm. *Read* / *Empty* mean the reader returned text / returned nothing on a given box — not that a detector found it.

*Control crops* / *Inventions* are the hallucination floor, measured on confirmed-blank frames **before** any accuracy number in this table was computed. A reader handed only GT boxes can never invent a location, so without control boxes its invention rate would be structurally zero — a number meaning "we never asked".

_(no reader arm ran)_

---

# DEV-SLICE TUNING FOOTNOTE (22 images, NOT the scored set): experiments/sweep_dev_v1_doctr

```
=== Phase 13h — stock vs tuned, dev-slice sweep (PHI-free) ===

dev set            dev_v1.csv  sha256 1733f78acc03318d2c832fda2b07ee19d47accb916e49b707e7bf7364dfea7df
images             22
ranked on          false_redaction_rate (lower is better)
stratum guard      a config is disqualified if any stratum regresses > 0.02

D-13.4: tuned on a dev slice drawn from OUTSIDE gt_v1, so these thresholds were
chosen without seeing the scored set. The winners below are FROZEN — re-tuning
after seeing gt_v1 numbers reintroduces the bias the slice was annotated to avoid.

-- doctr v1.0.1 ---------------------------------------------------
configurations tried: 5

  config_id  hash            false_red  keep_exact   added  omitted   med_ms
  stock      a1df589d1b52       0.2792      0.7208      89       73     2279
  tuned      0c49d61a22e3       0.2741      0.7259      87       71     2776
             {"bin_thresh": 0.3, "box_thresh": 0.1}
  tuned      98ae6f3e303b       0.2741      0.7259      85       71     2792
             {"bin_thresh": 0.3, "box_thresh": 0.3}
  tuned      f1a49bfe575c       0.2792      0.7208      89       73     2787
             {"bin_thresh": 0.1, "box_thresh": 0.05}
  tuned      ebcb518ce354       0.2741      0.7259      85       71     2835
             {"bin_thresh": 0.5, "box_thresh": 0.3}

  WINNER: {"bin_thresh": 0.3, "box_thresh": 0.1}
    config_hash 0c49d61a22e3
    false_redaction_rate: 0.2792 (stock) -> 0.2741
    Report this as its own finding — the delta is not folded into the
    headline number, and stock and tuned are separate rows, never averaged.

-- per-stratum, winner vs stock ------------------------------------------
(the pooled number above can improve while a stratum is destroyed — CLAUDE.md §8)
  doctr:
    stratum                       stock      tuned      delta
    ct_scout                        n/a        n/a        n/a
    ct_secondary_capture         0.1481     0.1481    +0.0000
    mg_2d                        0.4000     0.3000    +0.1000
    mg_tomo                         n/a        n/a        n/a
    us_ge                        0.2481     0.2403    +0.0078
    us_philips                   0.2000     0.2000    +0.0000
    us_samsung                   0.3704     0.3704    +0.0000
    us_siemens                   0.3667     0.3667    +0.0000
    us_sonosite                  0.4167     0.4167    +0.0000
    us_toshiba_canon             0.3766     0.3766    +0.0000

```

---

# DEV-SLICE TUNING FOOTNOTE (22 images, NOT the scored set): experiments/sweep_dev_v1_easyocr

```
=== Phase 13h — stock vs tuned, dev-slice sweep (PHI-free) ===

dev set            dev_v1.csv  sha256 1733f78acc03318d2c832fda2b07ee19d47accb916e49b707e7bf7364dfea7df
images             22
ranked on          false_redaction_rate (lower is better)
stratum guard      a config is disqualified if any stratum regresses > 0.02

D-13.4: tuned on a dev slice drawn from OUTSIDE gt_v1, so these thresholds were
chosen without seeing the scored set. The winners below are FROZEN — re-tuning
after seeing gt_v1 numbers reintroduces the bias the slice was annotated to avoid.

-- easyocr 1.7.2 -------------------------------------------------
configurations tried: 2

  config_id  hash            false_red  keep_exact   added  omitted   med_ms
  stock      f24d5b06a787       0.7132      0.2868     138      293    12966
             {"low_text": 0.4, "text_threshold": 0.7}
  thr0.2     741280cbf20c       0.9365      0.0635     280      427    10261
             {"low_text": 0.2, "text_threshold": 0.2}

  DISQUALIFIED on the per-stratum guard (pooled numbers looked fine):
    {"low_text": 0.2, "text_threshold": 0.2}
      worst: us_siemens 0.6000 -> 0.9333 (delta -0.3333)

  WINNER: none — every candidate regressed a stratum. The stock arm stands;
  do NOT freeze a tuned config for this engine on this evidence.

-- per-stratum, winner vs stock ------------------------------------------
(the pooled number above can improve while a stratum is destroyed — CLAUDE.md §8)
  easyocr:   (no winner — stock arm only)
    stratum                       stock      tuned      delta
    ct_scout                        n/a        n/a        n/a
    ct_secondary_capture         0.7963        n/a        n/a
    mg_2d                        0.8000        n/a        n/a
    mg_tomo                         n/a        n/a        n/a
    us_ge                        0.6822        n/a        n/a
    us_philips                   0.7091        n/a        n/a
    us_samsung                   0.7407        n/a        n/a
    us_siemens                   0.6000        n/a        n/a
    us_sonosite                  0.6667        n/a        n/a
    us_toshiba_canon             0.7403        n/a        n/a

```
