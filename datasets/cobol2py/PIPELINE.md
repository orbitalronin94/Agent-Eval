# Cobol2py dataset pipeline

This pipeline turns the repository's public, redistributable differential fixtures into evidence-backed behavior records. It is the first implementation phase of the Agent-Eval dataset roadmap.

## First model candidate

- Model: `onnx-community/Qwen3-0.6B-DQ-ONNX`
- Base family: Qwen3-0.6B
- Format: ONNX dynamic quantization, Transformers.js compatible
- Published repository size: approximately 370 MB; ONNX weights folder approximately 356 MB
- License lineage: Apache-2.0 from Qwen3; re-check the upstream model card and files before redistribution
- Pinned model revision used by the proposal job: `d6beae7`

This is an initial candidate selected for its size, reasoning-capable Qwen3 family and compatibility with RONIN-IA's existing Transformers.js runtime. It is not declared the best model until a controlled task benchmark compares it with alternatives. The model is used for hypotheses and annotations only, never as the semantic oracle.

## Automated flow

1. Checkout the current Cobol2py main revision and record its exact SHA.
2. Compile each public fixture with GnuCOBOL and capture stdout bytes and exit status.
3. Run the current deterministic translator and capture generated Python behavior.
4. Emit `train.jsonl`, `dev.jsonl` and `holdout.jsonl`, plus a hash manifest.
5. Prompt Qwen3 only with eligible train/dev counterexamples (maximum 20 per run).
6. Upload records and proposals as a 90-day Actions artifact.

The builder uses GnuCOBOL as the oracle. Exact matches and mismatches are computed from stdout bytes and process exit status, not from a model judge.

## Split isolation

Cases with identical normalized semantic-feature signatures are kept together, then indivisible groups are greedily assigned to a deterministic 70/15/15 train/dev/holdout target. This avoids the earlier broad ID-prefix grouping that put 91 of 105 cases into train. The annotation script rejects any holdout record. Holdout is emitted for independent evaluation but is never passed to the model, used to choose a prompt, or used to select a translator rule.

## Epistemic and review rules

- Oracle observations are ground truth only for the recorded compiler version, fixture and execution environment.
- A model proposal is not a label, accepted rule or implementation patch.
- Human review must verify each proposed semantic claim against COBOL documentation and a minimal executable GnuCOBOL counterexample.
- Rejected hypotheses and counterexamples should be retained.
- Only after review may examples be promoted into a curated training corpus.
- Third-party corpora are not imported by this workflow; fixture source comes from the Cobol2py repository. The source license and fixture provenance remain attached to every record.

## Run it

GitHub Actions runs weekly, on relevant pushes and manually via `workflow_dispatch`. Download the artifact named `cobol2py-model-dataset-<run-id>`.

The workflow does not fine-tune weights, edit Cobol2py, merge a translator PR, or merge model outputs into RONIN-IA. Those are separate promotion gates. Quantized inference weights are not a substitute for a trainable checkpoint; any later LoRA/QLoRA experiment must use an appropriate trainable base and preserve the independent holdout.

## Next phases

1. Expand the Cobol2py corpus and add provenance/licensing checks.
2. Add a fixed reasoning benchmark for model selection (accuracy, calibration, latency, peak RAM and artifact bytes).
3. Curate accepted examples and counterexamples into a versioned train corpus.
4. Add RONIN Knowledge, scientific identifiability, behavioral robustness, adversarial security and multi-agent/RAG in separate schemas and workflows.
5. Only then run adapter experiments and evaluate merged candidates in RONIN-IA against the unchanged baseline.
