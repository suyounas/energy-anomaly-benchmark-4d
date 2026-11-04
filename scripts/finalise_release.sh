#!/usr/bin/env bash
# Fill the commit hash and Zenodo DOI into the paper and CITATION.cff after the release exists.
#   bash scripts/finalise_release.sh v1.0.0 10.5281/zenodo.1234567
# Needs: the tag exists locally (git tag), and the DOI that Zenodo minted for that GitHub release.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:?usage: finalise_release.sh <tag> <zenodo-doi>}"
DOI="${2:?usage: finalise_release.sh <tag> <zenodo-doi>}"
[[ "$DOI" == 10.5281/zenodo.* ]] || { echo "DOI should look like 10.5281/zenodo.1234567"; exit 1; }
HASH="$(git rev-list -n 1 "$TAG" | cut -c1-10)"
[[ -n "$HASH" ]] || { echo "tag $TAG not found"; exit 1; }
python3 - "$TAG" "$HASH" "$DOI" <<'PY'
import sys, re
tag, h, doi = sys.argv[1:]
p = "paper/revised/main.tex"; s = open(p).read()
old_pat = re.compile(r"commit \\texttt\{\\textless COMMIT-HASH\\textgreater\}, archived DOI \\texttt\{\\textless ZENODO-DOI\\textgreater\}\. \\pending\{fill commit hash and Zenodo DOI after the final push\}")
assert old_pat.search(s), "placeholder sentence not found in main.tex (already filled?)"
new = rf"release \texttt{{{tag}}} (commit \texttt{{{h}}}), archived at \url{{https://doi.org/{doi}}}."
s = old_pat.sub(lambda m: new, s)
open(p, "w").write(s)
c = open("CITATION.cff").read()
c = re.sub(r"^# After the Zenodo release.*$", f'doi: "{doi}"', c, flags=re.M)
open("CITATION.cff", "w").write(c)
print("main.tex and CITATION.cff updated:", tag, h, doi)
PY
(cd paper/revised && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 && echo "main.pdf rebuilt") || echo "latexmk not available: rebuild the PDF yourself"
echo "Now: git add paper/revised/main.tex paper/revised/main.pdf CITATION.cff && git commit -m 'Add release tag, commit hash and DOI to the paper' && git push origin main"
