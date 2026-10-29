"""Static checks for the manuscript (no LaTeX installed here): environments, braces,
citation keys, cross-references, figure files and an approximate word count.

Usage (from the repo root): .venv/Scripts/python -I paper/lint.py
"""

import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["main.tex", "results.tex", "results_neuro.tex"]
COMMENT = re.compile(r"(?<!\\)%.*")


def main():
    texts = {name: COMMENT.sub("", (HERE / name).read_text(encoding="utf-8")) for name in FILES}
    body = "\n".join(texts.values())
    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", body))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", body))
    mismatched = {k: (begins[k], ends[k]) for k in set(begins) | set(ends) if begins[k] != ends[k]}
    print("environment mismatches:", mismatched or "none")
    for name, text in texts.items():
        plain = text.replace("\\{", "").replace("\\}", "")
        print(f"{name}: braces {plain.count('{')} open / {plain.count('}')} close")
    bibkeys = set(re.findall(r"^@\w+\{(\w+),", (HERE / "refs.bib").read_text(encoding="utf-8"), re.M))
    cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", body) for k in group.split(",")}
    print("cited but missing from refs.bib:", sorted(cited - bibkeys) or "none")
    print("in refs.bib but never cited:", sorted(bibkeys - cited) or "none")
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
    print("undefined references:", sorted(refs - labels) or "none")
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
        print(f"figure {fig}: {'ok' if (HERE / fig).exists() else 'MISSING'}")
    words = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    print("approximate words:", len(re.findall(r"[A-Za-z][A-Za-z'-]+", words)))


if __name__ == "__main__":
    main()
