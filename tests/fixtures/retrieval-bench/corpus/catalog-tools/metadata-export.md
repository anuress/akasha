---
title: metadata export
repo: catalog-tools
kind: reference
---
## ExportJob

ExportJob writes the catalog as MARC or CSV, chosen per run. The CSV form has
one row per copy; the MARC form has one record per title.

## Field mapping

The field mapping file lists which catalog fields become which output columns.
A field with no mapping is left out and counted in the summary.

## Incremental runs

An export can be limited to records changed since a given date.