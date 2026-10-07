---
title: duplicate records
repo: catalog-tools
kind: reference
---
## MatchCandidates

MatchCandidates pairs records whose title and first author are close after
normalising case and punctuation. Pairs are ranked so the likeliest duplicates
come first.

## MergeRecords

MergeRecords keeps the richer record, moves copies from the other record onto
it, and leaves a redirect so old links keep working.

## Review queue

Only a cataloguer can confirm a merge; the tool never merges on its own.