#!/usr/bin/env python
"""Update sub5 config files with the recommended generator architecture.

Usage:
    python scripts/update_sub5_configs.py --steps 64 --hidden 128 [--residual-blocks 2]
    python scripts/update_sub5_configs.py --from-recommendation recommend.json

Updates ddpm.steps (and optionally hidden/residual_blocks/train_epochs) in all
configs/bigfoot_ks_sub5_flow_*.yaml files using targeted line-replacement so
the original YAML formatting, comments, and list syntax are preserved.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# Simple targeted in-place updater: finds "  key: <value>" inside the ddpm section
# and replaces the value — no YAML round-trip so formatting is fully preserved.
def _patch_yaml_lines(text: str, key: str, new_val: int | str) -> tuple[str, str | None]:
    """Return (patched_text, old_value_str) for a scalar key inside the ddpm: block."""
    in_ddpm = False
    lines = text.splitlines(keepends=True)
    out = []
    old_val = None
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        # Detect top-level 'ddpm:' section start
        if re.match(r'^ddpm\s*:', line):
            in_ddpm = True
            out.append(line)
            continue
        # Detect end of ddpm section (another top-level key at indent 0)
        if in_ddpm and indent == 0 and stripped and not stripped.startswith("#"):
            in_ddpm = False
        if in_ddpm:
            m = re.match(r'^(\s+' + re.escape(key) + r'\s*:\s*)(\S+)(.*)', line)
            if m:
                old_val = m.group(2)
                line = m.group(1) + str(new_val) + m.group(3) + "\n"
        out.append(line)
    return "".join(out), old_val


def update_file(path: Path, fields: dict[str, int], dry_run: bool) -> list[str]:
    text = path.read_text()
    changes = []
    for key, new_val in fields.items():
        patched, old_val = _patch_yaml_lines(text, key, new_val)
        if old_val is not None and str(old_val) != str(new_val):
            changes.append(f"  ddpm.{key}: {old_val} → {new_val}")
            text = patched
        elif old_val is None:
            changes.append(f"  ddpm.{key}: (not found — skipped)")
    if changes and not dry_run:
        path.write_text(text)
    return changes


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--residual-blocks", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None,
                        help="Override ddpm.train_epochs (default: auto from steps)")
    parser.add_argument("--from-recommendation", default=None,
                        help="JSON file from analyze_sweep.py --recommend-json")
    parser.add_argument("--configs-dir", default="configs",
                        help="Directory containing config files (default: configs/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes but do not write files")
    args = parser.parse_args(argv)

    steps = args.steps
    hidden = args.hidden
    residual_blocks = args.residual_blocks
    train_epochs = args.train_epochs

    if args.from_recommendation:
        rec = json.loads(Path(args.from_recommendation).read_text())
        steps = steps or rec["steps"]
        hidden = hidden or rec.get("hidden", 128)
        residual_blocks = residual_blocks or rec.get("residual_blocks", 2)
        print(f"Loaded recommendation: {rec['best_model']}")
        print(f"  steps={steps}, hidden={hidden}, rb={residual_blocks}")

    if steps is None:
        parser.error("--steps is required (or use --from-recommendation)")

    # Auto-scale train_epochs: keep proportional to ODE steps (generation is more expensive,
    # training cost per epoch is the same — 10 epochs is still fine for all step counts).
    if train_epochs is None:
        train_epochs = 10

    fields: dict[str, int] = {"steps": steps}
    if hidden is not None:
        fields["hidden"] = hidden
    if residual_blocks is not None:
        fields["residual_blocks"] = residual_blocks
    fields["train_epochs"] = train_epochs

    configs_dir = Path(args.configs_dir)
    targets = sorted(configs_dir.glob("bigfoot_ks_sub5_flow_*.yaml"))
    if not targets:
        print(f"No bigfoot_ks_sub5_flow_*.yaml files found in {configs_dir}")
        return

    n_changed = 0
    for path in targets:
        changes = update_file(path, fields, dry_run=args.dry_run)
        real_changes = [c for c in changes if "not found" not in c and "→" in c]
        if real_changes:
            print(f"\n{path.name}:")
            for c in changes:
                print(c)
            n_changed += 1
        else:
            print(f"  {path.name}: no changes")

    if args.dry_run:
        print("\n[dry-run] no files written")
    else:
        print(f"\nUpdated {n_changed}/{len(targets)} config files.")


if __name__ == "__main__":
    main()
