---
title: renewal limits
repo: lending-api
kind: reference
---
## RenewalPolicy

RenewalPolicy allows an item to be renewed twice. A renewal is refused when
another member has a hold on the item, and the refusal names the due date that
still applies.

## RenewalWindow

A renewal is only offered in the last three days before the due date, so a
member cannot extend a loan weeks in advance.

## Late items

An overdue item cannot be renewed until its fine is settled.