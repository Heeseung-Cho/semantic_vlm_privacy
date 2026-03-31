# Semantic Pipeline Status

## Current Direction

Current main direction is a no-train VLM-guided detector pipeline:

`VLM semantic controller -> Grounding DINO proposals -> optional VLM reranking -> optional SAM`

At this stage, SAM is treated as optional post-processing, not a core component.
The current focus is on whether VLM-derived semantics can reliably improve detector proposal quality and null-aware behavior.

## Why This Direction

The current evidence supports this framing:

- Detector-only grounding with exact category text is unstable for several private classes.
- Many BIV private classes are visually similar, especially paper-like document categories.
- BIV-Priv-Seg contains intentional null cases due to real BLV capture conditions, so existence judgment matters.
- LLM2Seg appears closer to a semantic-detector bridge than to direct support-conditioned few-shot grounding.
- Zero-shot VLM semantics already show useful family-level guidance even without additional training.

## Baseline Findings

### 1. Pretrained vs finetuned G-DINO on support set

Support-set detector comparison showed that pretrained Grounding DINO is currently stronger than the epoch-11 finetuned checkpoint.

Observed support-set summary:

- pretrained:
  - mean_iou: 0.5467
  - hit50: 0.5625
  - nonempty detections: 11 / 16
- epoch11 finetuned:
  - mean_iou: 0.1851
  - hit50: 0.1875
  - nonempty detections: 4 / 16

Interpretation:

- current finetuning degraded open-vocabulary transfer on the private support categories
- pretrained OGC checkpoint remains the stronger detector baseline for the current pipeline

### 2. Qwen direct cue path

The earlier `Qwen -> cue -> G-DINO -> SAM` direct path worked in some cases, but its main bottleneck was cue generation reliability.

What was observed:

- when Qwen produced a good cue, G-DINO often localized correctly
- the main bottleneck was not always detector localization, but unstable cue recall and prompt contamination
- support-conditioned direct prompt routing was much weaker than expected

This makes Qwen more suitable as a semantic controller or reranker than as a direct grounding model in the current no-train setting.

## No-Train Semantic Pipeline

## Implemented structure

A new no-train path has been added:

- semantic controller
  - query-only mode
  - support-query mode
- semantic cue refinement
- null-prior handling
- proposal generation with multiple prompts
- optional reranking with support images
- optional SAM disable switch for bbox-only inspection

Relevant files:

- `challenge/semantic/semantic_gdino_sam.py`
- `challenge/semantic/run_semantic_gdino_sam_pipeline.py`

## What the semantic controller returns

The controller currently produces:

- `family`
- `summary`
- `proposal_prompts`
- `null_likely`
- inferred profile:
  - `paper_like`
  - `text_bearing`
  - `object_like`

This profile is now used to prioritize detector prompts differently depending on whether the case looks document-like or object-like.

## Query-only zero-shot sanity result

Running the semantic pipeline in `query_only` mode on support images produced a meaningful improvement over the initial raw semantic attempt.

Qualitative result:

- several support images produced good family-level cues
- good examples included receipt-like paper, business card, and pill bottle cases
- proposal overlays became substantially more sensible after cue filtering and prompt prioritization

Examples observed:

- `10.jpeg`: receipt-like paper family was reasonable and proposal aligned with the receipt area
- `225.jpeg`: business-card-like semantics produced a good card proposal
- `668.jpeg`: pill-bottle-like semantics emerged, though proposal drift remained

Failure cases still observed:

- `730`: still missed
- `136`: became worse under strict null policy, hand region selected instead
- `668`: still drifted toward a wrong instance/region in some runs

## What improved

Compared to the first semantic query-only attempt:

- low-signal prompts such as generic attributes or background terms are now filtered
- prompt lists are shorter and more detector-friendly
- null-aware gating is explicit and controllable
- support overlays and semantic overlays are closer to the correct canvas/orientation
- the zero-shot structure now looks usable instead of purely exploratory

## Interpretation

The current zero-shot result is meaningful because it works without any additional training.

This supports the following claim:

- family-level semantics are already useful for private-object localization
- exact category names are often too brittle
- a VLM-first semantic controller is justified, especially because null detection is part of the real task

## Paper-like vs Object-like

This distinction is now central.

### Paper-like / document-like

Representative categories:

- bank_statement
- local_newspaper
- doctors_prescription
- bills_or_receipt
- mortgage_invest_report
- letters_with_address
- medical_record_doc
- transcript
- business_card (borderline)

Expected behavior:

- detector can often find broad document regions
- subtype disambiguation is hard
- text-bearing semantics help more than exact class names
- null gating may be useful but can become too aggressive

### Object-like / device-like

Representative categories:

- pregnancy_test_box
- condom_box
- pregnancy_test
- empty_pill_bottle
- condom_w_plastic_bag
- credit_or_debit_card
- tattoo_sleeve

Expected behavior:

- shape prior matters more
- concrete object nouns should be prioritized
- strict null gating may hurt if the object is small or partially visible

## Current Improvement Points

### Priority 1. Null policy should become profile-dependent

Current `strict|skip|ignore` is global.
That is too coarse.

Observed implication:

- some paper-like cases may benefit from stronger null gating
- some object-like cases are harmed by strict null gating

Next step:

- test profile-conditioned null policy
- example:
  - paper-like -> strict
  - object-like -> ignore or weaker gating

### Priority 2. Proposal drift cases need reranking

Cases like `668` suggest that semantic family is acceptable but the selected detector proposal is wrong.

Next step:

- use support-conditioned reranking only after current query-only cue stage is stable
- reranking should solve proposal drift, not semantic family generation

### Priority 3. Prompt refinement for document families

Paper-like categories remain highly ambiguous.

Next step:

- emphasize document-family cues and short content descriptors
- avoid exact over-specific private names too early
- compare broad prompts such as:
  - financial document
  - medical form
  - receipt-like paper
  - newspaper page

### Priority 4. Inspect null cases explicitly

Cases like `730` need separate review.

Next step:

- distinguish true null-like cases from semantic failures
- log whether the controller failed semantically or whether the detector failed after a plausible cue

### Priority 5. Keep SAM detached until bbox logic stabilizes

SAM caused memory pressure and is not the main decision component.

Next step:

- continue bbox-only debugging
- reattach SAM only after semantic cueing and detector proposal quality are stable

## Recommended Immediate Plan

1. Keep pretrained G-DINO as the detector baseline.
2. Continue with query-only semantic-controller experiments.
3. Compare `null_policy=strict` vs `null_policy=ignore` by semantic profile.
4. Identify proposal-drift cases that are good candidates for support-conditioned reranking.
5. Only after that, enable support-conditioned reranking.
6. Reattach SAM last.

## Working Conclusion

The no-train semantic controller path is now viable.
It is not yet robust, but it is already much stronger than the first raw attempt and is strong enough to justify continuing along the VLM-guided detector direction.

The most promising next step is not end-to-end training, but refining cue selection and null handling, then using support-conditioned reranking to correct detector proposal drift.
