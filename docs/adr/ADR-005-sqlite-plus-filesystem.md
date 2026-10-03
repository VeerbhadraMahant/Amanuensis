# ADR-005: SQLite plus filesystem storage

Status: accepted

## Context
Single user, inspectable data, modest scale.

## Options considered
Postgres; SQLite + files; flat files only.

## Decision
Audio as 16 kHz mono WAV on disk; metadata, transcripts, lineage in SQLite.

## Consequences
Simple and portable; not built for concurrent multi-user access.
