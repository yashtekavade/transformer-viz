"""
inject_trace.py

Swaps a trace JSON file into index.html's embedded <script id="trace-data">
block, so you don't have to hand-edit a 250KB JSON blob every time you
regenerate data.

Usage:
    python inject_trace.py trace.json          # overwrites index.html in place
    python inject_trace.py trace.json --out index_real.html   # writes a copy instead
"""

import argparse
import re

TAG_RE = re.compile(
    r'(<script id="trace-data" type="application/json">)(.*?)(</script>)',
    re.S,
)


def inject(html_path, json_path, out_path):
    with open(json_path) as f:
        new_data = f.read().strip()
    with open(html_path) as f:
        html = f.read()

    match = TAG_RE.search(html)
    if not match:
        raise SystemExit(f"Couldn't find the trace-data <script> tag in {html_path}")

    new_html = html[: match.start(2)] + new_data + html[match.end(2):]
    with open(out_path, "w") as f:
        f.write(new_html)
    print(f"Injected {json_path} into {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="e.g. trace.json from extract_traces.py")
    parser.add_argument("--html", default="index.html", help="template html (default: index.html)")
    parser.add_argument("--out", default=None, help="output path (default: overwrite --html in place)")
    args = parser.parse_args()
    inject(args.html, args.json_file, args.out or args.html)
