#!/usr/bin/env python3
"""Build a traceable COBOL behavior dataset from Cobol2py's public fixtures.

The GnuCOBOL executable is the oracle. The translator is only the candidate.
Holdout records are emitted separately and must never be passed to a model.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(command: list[str], *, cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def family(case_id: str) -> str:
    return re.split(r"[-_]", case_id, maxsplit=1)[0].lower()


def group_key(case: dict[str, Any]) -> str:
    # Group identical semantic feature signatures, not broad ID prefixes such as "bench".
    features = case.get("features", [])
    normalized = sorted(
        re.sub(r"[^a-z0-9]+", " ", str(feature).lower()).strip()
        for feature in features
    )
    normalized = [feature for feature in normalized if feature]
    return "|".join(normalized) if normalized else family(case["id"])


def split_families(cases: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(group_key(case), []).append(case)

    total = len(cases)
    targets = {"train": total * 0.70, "dev": total * 0.15, "holdout": total * 0.15}
    counts = {"train": 0, "dev": 0, "holdout": 0}
    assignments: dict[str, str] = {}

    # Place larger semantic groups first. For each indivisible group, choose
    # the split that minimizes normalized squared deviation from 70/15/15.
    ordered = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(item[0].encode("utf-8")).hexdigest()),
    )
    for key, members in ordered:
        size = len(members)
        choices = []
        for split in ("train", "dev", "holdout"):
            projected = dict(counts)
            projected[split] += size
            score = sum(
                ((projected[name] - targets[name]) / max(targets[name], 1.0)) ** 2
                for name in targets
            )
            choices.append((score, split))
        _, selected = min(choices)
        assignments[key] = selected
        counts[selected] += size
    return assignments

def text_or_replacement(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translator", type=Path, default=Path("cobol2py.py"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/cases.json"))
    parser.add_argument("--license", default="MIT (verified from source repository LICENSE)")
    args = parser.parse_args()

    root = args.source_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    translator = (root / args.translator).resolve()
    if not manifest_path.is_relative_to(root) or not translator.is_relative_to(root):
        parser.error("manifest and translator must remain inside source-root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases:
        parser.error("manifest contains no cases")

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    compiler = subprocess.run(["cobc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    splits = split_families(cases)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "holdout": []}

    for case in cases:
        source = (root / case["path"]).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise SystemExit(f"fixture path is missing or escapes source-root: {case['path']}")
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        group = group_key(case)
        split = splits[group]
        item: dict[str, Any] = {
            "id": case["id"],
            "source_repo": "orbitalronin94/cobol2py",
            "source_path": case["path"],
            "source_revision": revision,
            "source_license": args.license,
            "knowledge_type": "unverified_fixture",
            "dialect": "GnuCOBOL-compatible fixture; exact compiler version recorded",
            "compiler": compiler,
            "input": {"language": "COBOL", "source": source_text, "features": case.get("features", [])},
            "target": None,
            "candidate": None,
            "evidence_refs": [],
            "epistemic_status": "reference_compile_failed",
            "limitations": ["Observed only under the recorded GnuCOBOL version and CI environment."],
            "counterexamples": [],
            "split": split,
            "group_key": group,
            "supervised_eligible": False,
        }

        try:
            with __import__("tempfile").TemporaryDirectory(prefix="agent-eval-cobol-") as tmp:
                work = Path(tmp)
                reference_path = work / "reference"
                compiled = run(["cobc", "-x", "-I", str(source.parent), "-o", str(reference_path), str(source)], cwd=root)
                if compiled.returncode != 0:
                    item["evidence_refs"].append("GnuCOBOL compile failed: " + text_or_replacement(compiled.stderr)[-1000:])
                else:
                    reference = run([str(reference_path)], cwd=work)
                    item["target"] = {
                        "oracle": "GnuCOBOL",
                        "stdout_text": text_or_replacement(reference.stdout),
                        "stdout_base64": base64.b64encode(reference.stdout).decode("ascii"),
                        "returncode": reference.returncode,
                    }
                    item["evidence_refs"].append("GnuCOBOL stdout SHA-256: " + sha256(reference.stdout))
                    item["supervised_eligible"] = True
                    if reference.returncode is None:
                        item["epistemic_status"] = "reference_execution_failed"
                    else:
                        translated = run([sys.executable, str(translator), str(source), "--format", "auto"], cwd=root)
                        if translated.returncode != 0:
                            item["candidate"] = {
                                "translation_returncode": translated.returncode,
                                "translation_stderr": text_or_replacement(translated.stderr)[-2000:],
                                "stdout_text": "",
                                "stdout_base64": "",
                                "returncode": translated.returncode,
                            }
                            item["epistemic_status"] = "observed_mismatch"
                            item["counterexamples"].append("Translator did not emit a runnable program.")
                        else:
                            generated = work / "candidate.py"
                            generated.write_bytes(translated.stdout)
                            executed = run([sys.executable, str(generated)], cwd=work)
                            exact = executed.stdout == reference.stdout and executed.returncode == reference.returncode
                            item["candidate"] = {
                                "translation_returncode": translated.returncode,
                                "stdout_text": text_or_replacement(executed.stdout),
                                "stdout_base64": base64.b64encode(executed.stdout).decode("ascii"),
                                "returncode": executed.returncode,
                                "stderr": text_or_replacement(executed.stderr)[-2000:],
                                "generated_python_sha256": sha256(translated.stdout),
                            }
                            item["epistemic_status"] = "observed_exact_match" if exact else "observed_mismatch"
                            item["knowledge_type"] = "observed_behavior_pair" if exact else "counterexample"
                            if not exact:
                                item["counterexamples"].append("Generated Python differs from GnuCOBOL stdout bytes or exit status.")
        except (OSError, subprocess.TimeoutExpired) as exc:
            item["epistemic_status"] = "reference_execution_failed"
            item["evidence_refs"].append(type(exc).__name__ + ": " + str(exc))

        records[split].append(item)

    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    hashes = {}
    for split, rows in records.items():
        path = args.output / f"{split}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        encoded = payload.encode("utf-8")
        path.write_bytes(encoded)
        counts[split] = {
            "records": len(rows),
            "supervised_eligible": sum(row["supervised_eligible"] for row in rows),
            "exact_matches": sum(row["epistemic_status"] == "observed_exact_match" for row in rows),
            "counterexamples": sum(row["epistemic_status"] == "observed_mismatch" for row in rows),
        }
        hashes[path.name] = sha256(encoded)

    metadata = {
        "schema_version": 1,
        "dataset": "cobol2py-observed-behavior",
        "source_repo": "orbitalronin94/cobol2py",
        "source_revision": revision,
        "manifest_sha256": sha256(manifest_path.read_bytes()),
        "compiler": compiler,
        "split_policy": "deterministic group split by fixture family; holdout excluded from model prompts",
        "counts": counts,
        "sha256": hashes,
        "model_annotations_are_not_ground_truth": True,
    }
    (args.output / "dataset-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
