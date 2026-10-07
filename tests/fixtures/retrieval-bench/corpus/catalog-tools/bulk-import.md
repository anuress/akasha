---
title: bulk import
repo: catalog-tools
kind: reference
---
## ImportBatch

ImportBatch reads a spreadsheet of records in groups of five hundred. Each
group is applied in one transaction, so a bad row rolls back its whole group
and nothing is half imported.

## Row report

Every rejected row is written to a report with its line number and the reason,
including rows skipped as duplicate records.

## Dry run

A dry run validates the file and prints the counts without writing anything.