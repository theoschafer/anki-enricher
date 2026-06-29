"""
clean_articles.py
=================
Strip leading "A " / "An " from the English field of all Korean notes.

For every note of the `theo-korean-advanced` model whose English field starts
with "a " or "an " (case-insensitive), this script removes the article and
capitalises the first remaining letter. If the result would collide with an
existing English value (in the same model), " DUP" is appended so the user
can clean those up manually later.

By default the script triggers an AnkiWeb sync before and after applying
changes (equivalent to clicking the sync button in the Anki toolbar). Use
--no-sync to skip that.

Usage:
    python clean_articles.py --dry-run   # preview only, no changes
    python clean_articles.py             # preview + ask for confirmation
    python clean_articles.py --yes       # apply without prompting
    python clean_articles.py --no-sync   # skip the auto-sync calls

Requires Anki to be running with the AnkiConnect addon (code 2055492159)
and (for sync) a valid AnkiWeb login configured in the Anki desktop app.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

ANKICONNECT_URL = "http://localhost:8765"
ANKI_MODEL_NAME = "theo-korean-advanced"
ENGLISH_FIELD = "English"

# Match a leading "a" or "an" followed by at least one whitespace character.
# Case-insensitive. Anything else (e.g. "Apple", "Antenna") is left alone.
ARTICLE_RE = re.compile(r"^\s*(an?)\s+", re.IGNORECASE)


def ankiconnect_request(action: str, **params):
    payload = json.dumps(
        {"action": action, "version": 6, "params": params}
    ).encode("utf-8")
    req = urllib.request.Request(
        ANKICONNECT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach AnkiConnect at {ANKICONNECT_URL}. "
            f"Make sure Anki is running and the AnkiConnect addon is installed.\n"
            f"Original error: {exc}"
        )
    if data.get("error"):
        raise RuntimeError(f"AnkiConnect error on {action}: {data['error']}")
    return data["result"]


def strip_leading_article(text: str) -> str | None:
    """Return article-stripped, capitalised text, or None if no leading
    article is present."""
    m = ARTICLE_RE.match(text)
    if not m:
        return None
    rest = text[m.end():].lstrip()
    if not rest:
        return None
    return rest[0].upper() + rest[1:]


def sync_anki(label: str) -> None:
    """Trigger AnkiWeb sync (same as clicking the sync button)."""
    print(f"[ANKI] Syncing with AnkiWeb ({label})...")
    try:
        ankiconnect_request("sync")
        print(f"[ANKI] Sync ({label}) complete.")
    except RuntimeError as exc:
        print(f"[ANKI] Sync ({label}) failed: {exc}", file=sys.stderr)


def fetch_all_notes() -> list[dict]:
    note_ids = ankiconnect_request(
        "findNotes", query=f'note:"{ANKI_MODEL_NAME}"'
    )
    print(f"[ANKI] Found {len(note_ids)} notes for model '{ANKI_MODEL_NAME}'")
    if not note_ids:
        return []

    notes: list[dict] = []
    BATCH = 500
    for i in range(0, len(note_ids), BATCH):
        chunk = note_ids[i : i + BATCH]
        notes.extend(ankiconnect_request("notesInfo", notes=chunk))
    return notes


def plan_updates(notes: list[dict]) -> list[tuple[int, str, str]]:
    """Return a list of (note_id, old_english, new_english)."""

    # All current English values (case-folded) for duplicate detection.
    existing_values: set[str] = set()
    for n in notes:
        v = n["fields"].get(ENGLISH_FIELD, {}).get("value", "")
        existing_values.add(v.strip().casefold())

    updates: list[tuple[int, str, str]] = []
    for n in notes:
        old = n["fields"].get(ENGLISH_FIELD, {}).get("value", "")
        new = strip_leading_article(old)
        if new is None:
            continue
        # If the stripped value already exists, mark as DUP.
        if new.casefold() in existing_values:
            new = new + " DUP"
        # Reserve the new value so two distinct notes that collapse to the
        # same target also get caught (e.g. "A pen" + "An pen" → both DUP).
        existing_values.add(new.casefold())
        updates.append((n["noteId"], old, new))
    return updates


def apply_updates(updates: list[tuple[int, str, str]]) -> None:
    failed = 0
    for note_id, _old, new in updates:
        try:
            ankiconnect_request(
                "updateNoteFields",
                note={"id": note_id, "fields": {ENGLISH_FIELD: new}},
            )
        except RuntimeError as exc:
            failed += 1
            print(f"  FAIL note {note_id}: {exc}", file=sys.stderr)
    print(
        f"[ANKI] Done. Updated {len(updates) - failed}, failed {failed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip leading 'A'/'An' from the English field of Korean notes."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print proposed changes and exit (no modifications).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply changes without asking for confirmation.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Don't auto-sync with AnkiWeb before/after running.",
    )
    args = parser.parse_args()

    if not args.no_sync:
        sync_anki("before")

    notes = fetch_all_notes()
    if not notes:
        return

    updates = plan_updates(notes)

    print(f"\n[PLAN] {len(updates)} note(s) to update\n")
    dup_count = 0
    for note_id, old, new in updates:
        is_dup = new.endswith(" DUP")
        if is_dup:
            dup_count += 1
        flag = "  [DUP]" if is_dup else ""
        print(f"  {note_id}: {old!r:40s} -> {new!r}{flag}")
    if dup_count:
        print(f"\n  ({dup_count} of these would collide with an existing entry)")

    if not updates:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n[DRY-RUN] No changes applied.")
        return

    if not args.yes:
        try:
            answer = input(
                f"\nApply these {len(updates)} updates? [y/N] "
            ).strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    apply_updates(updates)

    if not args.no_sync:
        sync_anki("after")
        print(
            "\nDon't forget to sync your phone's Anki app to pull the changes."
        )


if __name__ == "__main__":
    main()
