import sys
import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def verify_all():
    print("=" * 80)
    print("VERIFYING AUTONOMOUS CHECKPOINT & RESUME ARTIFACTS")
    print("=" * 80)

    # 1. Verify autonomous_state.json
    state_file = PROJECT_ROOT / "scratch" / "autonomous_state.json"
    assert state_file.exists(), "autonomous_state.json missing!"
    with open(state_file, "r") as f:
        state = json.load(f)
    print(f"[OK] autonomous_state.json is valid JSON (Status: {state.get('status')})")

    # 2. Verify RESUME.md
    resume_md = PROJECT_ROOT / "scratch" / "RESUME.md"
    assert resume_md.exists(), "RESUME.md missing!"
    print("[OK] RESUME.md exists.")

    # 3. Verify artifact_manifest.json
    manifest_file = PROJECT_ROOT / "scratch" / "artifact_manifest.json"
    assert manifest_file.exists(), "artifact_manifest.json missing!"
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    print(f"[OK] artifact_manifest.json is valid JSON with {len(manifest)} tracked paths.")

    # 4. Verify disk existence of all manifest files
    for key, rel_path in manifest.items():
        p = PROJECT_ROOT / rel_path
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [EXISTS] {key:<30} -> {rel_path} ({size_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {key:<30} -> {rel_path}")
            assert False, f"Manifest artifact missing: {rel_path}"

    # 5. Verify processed_s1 count in checkpoint vs outputs
    ck_file = PROJECT_ROOT / "scratch" / "inference_checkpoint.json"
    with open(ck_file, "r") as f:
        ck = json.load(f)
    proc_s1 = ck.get("processed_s1", 0)
    print(f"\n[OK] inference_checkpoint.json processed_s1 = {proc_s1:,}")

    with open(PROJECT_ROOT / "output" / "matching_results.tsv", "r", encoding="utf-8") as f:
        match_lines = sum(1 for _ in f) - 1
    with open(PROJECT_ROOT / "output" / "candidate_pairs.tsv", "r", encoding="utf-8") as f:
        cand_lines = sum(1 for _ in f) - 1

    print(f"[OK] matching_results.tsv line count: {match_lines:,}")
    print(f"[OK] candidate_pairs.tsv line count  : {cand_lines:,}")
    assert match_lines == proc_s1, f"Mismatch in matching rows: {match_lines} vs {proc_s1}"
    assert cand_lines == proc_s1, f"Mismatch in candidate rows: {cand_lines} vs {proc_s1}"

    print("\n" + "=" * 80)
    print("CHECKPOINT VERIFICATION COMPLETE: ALL CHECKS PASSED 100% SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    verify_all()
