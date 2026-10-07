---
title: hold queue
repo: lending-api
kind: reference
---
## HoldQueue

HoldQueue orders waiting members by the moment they placed the hold. When a
copy comes back, the first member in line is notified and the copy is set aside
on the pickup shelf, labelled with a barcode for the desk.

## PickupWindow

PickupWindow gives a notified member three days to collect an item. After that
the hold passes to the next member and the copy is released.

## Queue position

Position is recalculated whenever a member ahead cancels, and the member sees
the new position on their next visit.