---
title: search facets
repo: lending-api
kind: reference
---
## FacetIndex

FacetIndex counts matching records per author, subject and publication year so
the results page can show filters with their counts.

## FacetSelection

Selected facets combine with AND across groups and OR within a group. Choosing
a second subject widens the results; choosing a year narrows them.

## Stale counts

Counts are rebuilt after each catalog import, so they can lag a new record by
a few minutes.