"""
check_links.py
Quick diagnostic: shows which cached papers have a DOI, a landing
page URL, and an open-access flag, so we can see exactly what the
References section has to work with. Read-only - changes nothing.
"""

import os
import json
import time

CACHE_FILE = os.path.join("cache", "retriever", "ethical_hacking.json")

if not os.path.exists(CACHE_FILE):
    print(f"File not found: {CACHE_FILE}")
    raise SystemExit

modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(CACHE_FILE)))
print(f"File: {CACHE_FILE}")
print(f"Last modified: {modified}")

with open(CACHE_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Total papers: {len(papers)}\n")

no_link = 0
for i, p in enumerate(papers, start=1):
    doi = p.get("doi")
    landing = p.get("landing_page_url")
    has_link = bool(landing or doi)
    if not has_link:
        no_link += 1
    flag = "OK       " if has_link else "NO LINK  "
    print(f"{i:2}. {flag} doi={'yes' if doi else 'no ':3} landing={'yes' if landing else 'no ':3} "
          f"is_oa={p.get('is_oa', 'missing')}  {p.get('title', '')[:55]}")

print(f"\nPapers with NO link at all: {no_link}")