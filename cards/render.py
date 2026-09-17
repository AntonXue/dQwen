#!/usr/bin/env python3
"""Render the five model cards from cards/template.md and push them to the Hub.

    python cards/render.py --check   # render; diff each against the card the Hub serves on main
    python cards/render.py --push    # commit README.md (+ the modeling file) on main and both step branches

Every card is a landing page: the family table is on all five, with the current repo bolded.
The modeling file travels in the same commit so the snippet on the page matches the code
beside it. Weights are never touched; verify with the safetensors ledger afterwards.
"""
import argparse, difflib, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
ORG = "UT-IFML"
REFS = ("main", "step50000-swa", "step25000-swa")
MODELS = [  # name, parent, total params, backbone label, modeling file, family tag
    ("dQwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B", "0.75B", "hybrid", "modeling_dqwen3_5.py", "qwen3.5"),
    ("dQwen3.5-2B-Base",   "Qwen/Qwen3.5-2B",   "1.88B", "hybrid", "modeling_dqwen3_5.py", "qwen3.5"),
    ("dQwen3.5-4B-Base",   "Qwen/Qwen3.5-4B",   "4.21B", "hybrid", "modeling_dqwen3_5.py", "qwen3.5"),
    ("dQwen3.5-9B-Base",   "Qwen/Qwen3.5-9B",   "8.95B", "hybrid", "modeling_dqwen3_5.py", "qwen3.5"),
    ("dQwen3-1.7B-Base",   "Qwen/Qwen3-1.7B",   "1.72B", "full attention (control)", "modeling_dqwen3.py", "qwen3"),
]
def description(name, parent):
    short = parent.split("/")[1]
    if name.startswith("dQwen3.5"):
        return (f"A masked diffusion language model adapted from {short}. The backbone is hybrid: "
                "only its attention layers are made bidirectional, and the Gated DeltaNet layers stay causal.")
    return (f"A masked diffusion language model adapted from {short}. The backbone is full attention, "
            "and every layer is made bidirectional. It is the control model in the paper's matched "
            "comparison against the hybrid dQwen3.5-2B.")
def render(name):
    t = (HERE / "template.md").read_text()
    me = next(m for m in MODELS if m[0] == name)
    rows = []
    for n, _, total, backbone, _, _ in MODELS:
        cell = f"**{n}** (this repo)" if n == name else f"[{n}](https://huggingface.co/{ORG}/{n})"
        rows.append(f"| {cell} | {total} | {backbone} |")
    return t.format(name=name, parent=me[1], description=description(name, me[1]),
                    arch_tag="hybrid-attention" if me[3] == "hybrid" else "full-attention",
                    family_tag=me[5], family_rows="\n".join(rows))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for name, _, _, _, modeling, _ in MODELS:
        rid = f"{ORG}/{name}"; card = render(name)
        if a.check:
            served = open(hf_hub_download(rid, "README.md", force_download=True)).read()
            d = list(difflib.unified_diff(served.splitlines(), card.splitlines(), lineterm=""))
            print(f"  {rid:28s} rendered {len(card.splitlines())} lines; diff vs served main: {sum(l.startswith(('+','-')) for l in d)} changed lines")
        if a.push:
            assert api.repo_info(rid).id == rid, f"{rid} is a redirect"
            ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
                   CommitOperationAdd(path_in_repo=modeling, path_or_fileobj=str(REPO / "models" / modeling))]
            for ref in REFS:
                c = api.create_commit(repo_id=rid, operations=ops, revision=ref,
                                      commit_message="Model card: family landing page with a two-statement quickstart; generate() accepts a string prompt and loads its own tokenizer")
                print(f"  {rid:28s} {ref:14s} {c.oid[:12]}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
