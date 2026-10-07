---
title: isbn validation
repo: catalog-tools
kind: reference
---
## IsbnChecker

IsbnChecker verifies the check digit of both ten and thirteen digit ISBNs and
strips hyphens and spaces before comparing. A record with an invalid ISBN is
kept but flagged for review.

## Conversion

Ten digit ISBNs are converted to thirteen digits on import so lookups use a
single form.

## Duplicate warning

Two records with the same normalised ISBN raise a warning, not an error.