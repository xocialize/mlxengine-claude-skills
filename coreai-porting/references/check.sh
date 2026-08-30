#!/bin/sh
# Fail if any two reference files share a first-line heading (the clobber signature).
cd "$(dirname "$0")" || exit 1
dupes=$(for f in *.md; do printf '%s\t%s\n' "$(head -1 "$f")" "$f"; done | sort | awk -F'\t' '
  {if ($1==prev) {print "  CLOBBER: " prevf " and " $2 " share heading: " $1; bad=1} prev=$1; prevf=$2}
  END {exit bad+0}')
if [ -n "$dupes" ]; then echo "$dupes"; echo "FAIL"; exit 1; fi
echo "OK — $(ls *.md | wc -l | tr -d ' ') reference files, all headings distinct"
