---
title: overdue fines
repo: lending-api
kind: reference
---
## FineCalculator

FineCalculator charges a flat daily rate for every day an item is late, up to a
cap equal to the item's replacement price. Grace days at the start of a loan
are never charged.

## FineWaiver

FineWaiver lets a librarian forgive a fine once per member per year. The waiver
carries a reason code so the monthly report can separate waived balances from
unpaid ones.

## Settling a balance

A member with an unpaid balance cannot borrow again until it is settled; the
checkout desk shows the amount owed.