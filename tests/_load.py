"""Load selected top-level definitions from the numbered study scripts.

The study scripts are Colab pipelines (they mount Drive and run on import), so
they cannot be imported.  This helper parses a script and executes ONLY the named
functions and constants, leaving the published code untouched.
"""
import ast
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(rel_path, names):
    path = os.path.join(ROOT, rel_path)
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & set(names):
                keep.append(node)
    found = set()
    for n in keep:
        found |= ({n.name} if isinstance(n, ast.FunctionDef)
                  else {t.id for t in n.targets if isinstance(t, ast.Name)})
    missing = set(names) - found
    assert not missing, f"{rel_path}: could not find {sorted(missing)}"
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns
