"""
enrich.py — Enrich existing Anki notes with TTS audio and AI illustrations.

Usage:
  python enrich.py                  # process all notes tagged 'pending-enrichment'
  python enrich.py --limit 3        # only process the first 3 (good for first try)
  python enrich.py --dry-run        # show what would be processed, no API calls
  python enrich.py --keep-tag       # don't remove the tag after enrichment

How to mark notes for enrichment:
  In Anki, browse → select notes → right-click → "Add Tags" → 'pending-enrichment'
  Once enriched, the tag is automatically removed (unless --keep-tag is set).

This script searches your ENTIRE Anki collection (all decks) for notes that:
  - have note type 'theo-korean-advanced'
  - have the 'pending-enrichment' tag
For each, it generates TTS for the Korean field and an illustration for the
English field, then fills the KoreanPronunciation and NormalImage fields
in-place via AnkiConnect (note IDs and scheduling are preserved).
"""

import argparse
import base64
import html
import re
import sys
import time
from pathlib import Path

from pipeline import (
    ANKI_MODEL_NAME,
    AUDIO_DIR,
    GCP_PROJECT,
    IMAGE_DIR,
    ankiconnect_request,
    craft_image_prompt,
    generate_image,
    generate_tts,
    sync_anki,
)

ENRICH_TAG = "pending-enrichment"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_field(value: str) -> str:
    """Strip HTML tags + decode entities so a field is safe to feed to APIs."""
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_pending_notes() -> list[dict]:
    query = f'"note:{ANKI_MODEL_NAME}" tag:{ENRICH_TAG}'
    note_ids = ankiconnect_request("findNotes", query=query)
    if not note_ids:
        return []
    return ankiconnect_request("notesInfo", notes=note_ids)


def upload_media(path: Path) -> None:
    data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ankiconnect_request("storeMediaFile", filename=path.name, data=data_b64)


# ---------------------------------------------------------------------------
# Per-note enrichment
# ---------------------------------------------------------------------------

def enrich_note(note: dict) -> bool:
    """Returns True if any API call was made (caller should sleep)."""
    nid = note["noteId"]
    fields = note["fields"]

    korean_raw  = fields.get("Korean", {}).get("value", "")
    english_raw = fields.get("English", {}).get("value", "")
    korean  = clean_field(korean_raw)
    english = clean_field(english_raw)

    if not korean or not english:
        print(f"  [SKIP] note {nid}: missing Korean or English (Korean={korean!r}, English={english!r})")
        return False

    has_audio = bool(clean_field(fields.get("KoreanPronunciation", {}).get("value", "")))
    has_image = bool(clean_field(fields.get("NormalImage", {}).get("value", "")))

    audio_path = AUDIO_DIR / f"note_{nid}.mp3"
    image_path = IMAGE_DIR / f"note_{nid}.jpg"

    api_called = False
    fields_to_update: dict[str, str] = {}

    if has_audio:
        print(f"  [TTS]  note {nid}: KoreanPronunciation already filled, skipping")
    else:
        if not audio_path.exists():
            generate_tts(korean, audio_path)
            api_called = True
            print(f"  [TTS]  note {nid} '{korean}' → {audio_path}")
        else:
            print(f"  [TTS]  note {nid}: file exists, reusing {audio_path}")
        upload_media(audio_path)
        fields_to_update["KoreanPronunciation"] = f"[sound:{audio_path.name}]"

    if has_image:
        print(f"  [IMG]  note {nid}: NormalImage already filled, skipping")
    else:
        if not image_path.exists():
            prompt = craft_image_prompt(english)
            print(f"  [GEMINI] note {nid} prompt → {prompt[:80]}…")
            generate_image(prompt, image_path)
            api_called = True
            print(f"  [IMG]  note {nid} → {image_path}")
        else:
            print(f"  [IMG]  note {nid}: file exists, reusing {image_path}")
        upload_media(image_path)
        fields_to_update["NormalImage"] = f'<img src="{image_path.name}">'

    if fields_to_update:
        ankiconnect_request("updateNoteFields", note={
            "id": nid,
            "fields": fields_to_update,
        })
        print(f"  [ANKI] note {nid} updated: {list(fields_to_update.keys())}")
    else:
        print(f"  [ANKI] note {nid}: nothing to update")

    return api_called


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich Anki notes tagged 'pending-enrichment'")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N matching notes (good for testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="List the notes that would be processed, then exit")
    p.add_argument("--keep-tag", action="store_true",
                   help="Don't remove the 'pending-enrichment' tag after success")
    p.add_argument("--no-sync", action="store_true",
                   help="Don't auto-sync with AnkiWeb before/after running.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if GCP_PROJECT == "your-project-id":
        print("ERROR: Set GCP_PROJECT in pipeline.py before running.", file=sys.stderr)
        sys.exit(1)

    AUDIO_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    if not args.no_sync and not args.dry_run:
        sync_anki("before")

    notes = find_pending_notes()
    print(f"Found {len(notes)} note(s) tagged '{ENRICH_TAG}' "
          f"with note type '{ANKI_MODEL_NAME}' (across all decks)")

    if args.limit is not None:
        notes = notes[: args.limit]
        print(f"  → Limited to first {len(notes)}")

    if not notes:
        return

    if args.dry_run:
        print("\nDry run — these notes would be processed:")
        for n in notes:
            nid = n["noteId"]
            kor = clean_field(n["fields"].get("Korean", {}).get("value", ""))
            eng = clean_field(n["fields"].get("English", {}).get("value", ""))
            print(f"  - note {nid}: {kor}  ({eng})")
        return

    enriched = 0
    failed = 0
    for i, note in enumerate(notes):
        nid = note["noteId"]
        print(f"\n[{i+1}/{len(notes)}] note {nid}")
        try:
            api_called = enrich_note(note)

            if not args.keep_tag:
                remove_tag_with_verification(nid, ENRICH_TAG)

            enriched += 1

            # Throttle only when we hit the Imagen quota (~5/min on new projects)
            if api_called and i < len(notes) - 1:
                time.sleep(8)
        except Exception as exc:
            failed += 1
            print(f"  ERROR on note {nid}: {exc}", file=sys.stderr)

    print(f"\nDone. Enriched: {enriched}, Failed: {failed}")

    if not args.no_sync and enriched > 0:
        sync_anki("after")
        print("\nDon't forget to sync your phone's Anki app to pull the changes.")


def remove_tag_with_verification(nid: int, tag: str, attempts: int = 3) -> None:
    """Remove a tag and verify the change persisted.

    AnkiConnect occasionally drops a removeTags call if it follows an
    updateNoteFields call too closely, so we retry a few times with a small
    delay until verification confirms the tag is gone.
    """
    for attempt in range(1, attempts + 1):
        # Tiny wait so updateNoteFields finishes committing
        time.sleep(0.5)
        ankiconnect_request("removeTags", notes=[nid], tags=tag)

        info = ankiconnect_request("notesInfo", notes=[nid])
        current_tags = info[0].get("tags", []) if info else []
        if tag not in current_tags:
            print(f"  [TAG]  removed '{tag}' from note {nid}")
            return
        print(f"  [TAG]  removeTags didn't stick on attempt {attempt}, retrying…")

    print(f"  [TAG]  WARNING: could not remove '{tag}' from note {nid} after {attempts} attempts",
          file=sys.stderr)


if __name__ == "__main__":
    main()
