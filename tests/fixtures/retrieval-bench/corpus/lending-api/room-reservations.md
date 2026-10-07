---
title: room reservations
repo: lending-api
kind: reference
---
## RoomCalendar

RoomCalendar stores bookings in fifteen-minute slots. A booking may not overlap
another booking in the same room.

## BookingLimit

A member may hold two upcoming reservations at once. A reservation is released
automatically when nobody checks in within fifteen minutes of the start.

## Group study rooms

Group rooms need at least three names on the booking; solo rooms need one.