---
title: inventory audit
repo: catalog-tools
kind: reference
---
## AuditScan

AuditScan walks a shelf range with a handheld reader and compares the scanned
barcodes with the catalog. Items seen but not expected, and expected but not
seen, are listed separately.

## MissingItems

An item missing from two audits in a row is marked lost and its replacement
price is added to the report.

## Scheduling

Each range is audited once a year; reference works are audited twice.