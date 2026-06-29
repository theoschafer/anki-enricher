"""
correct.py — LLM-powered correction of Korean/English Anki notes.

Scans every note with note type 'theo-korean-advanced', asks Gemini 2.5 Flash
whether the Korean has a spelling typo or whether the English mismatches the
Korean's meaning, and writes proposed fixes to corrections.csv for review.
A second pass (--apply) commits those fixes back to Anki via AnkiConnect.

Typical workflow:
  python correct.py --limit 20      # smoke test the LLM on 20 notes
  # review corrections.csv
  python correct.py                 # full scan of remaining notes
  # review corrections.csv again
  python correct.py --apply         # commit to Anki + tag 'corrected'
  python enrich.py                  # regenerate audio for Korean-fixed notes

Design:
  Pass 1 (default): read-only on Anki, calls Gemini per note, appends rows to
    corrections.csv. Resumable — skips notes already scored. Skips notes already
    tagged 'corrected' (override with --include-corrected).
  Pass 2 (--apply): reads corrections.csv, updates fields via AnkiConnect, adds
    the 'corrected' tag. If Korean was fixed, also clears KoreanPronunciation
    and re-adds the 'pending-enrichment' tag so enrich.py regenerates the audio
    on its next run.

Budget:
  At ~200 input + ~100 output tokens per note over ~833 notes the total cost is
  roughly $0.30 on Gemini 2.5 Flash. A hard guardrail aborts at projected $4.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

import config as _config

from pipeline import (
    ANKI_MODEL_NAME,
    AUDIO_FIELD,
    ENGLISH_FIELD,
    GCP_PROJECT,
    GCP_REGION,
    TARGET_FIELD,
    _call_with_timeout,
    ankiconnect_request,
    sync_anki,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORRECTIONS_CSV = Path("corrections.csv")
CORRECTED_TAG   = "corrected"
ENRICH_TAG      = "pending-enrichment"

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_TIMEOUT_S  = 45

# Approx pricing for gemini-2.5-flash (USD per 1M tokens), used only as a
# coarse cost guardrail — exact rate may drift but is well within our margin.
PRICE_INPUT_PER_M  = 0.30
PRICE_OUTPUT_PER_M = 2.50
COST_CAP_USD       = 4.0   # abort if running cost projected to exceed this

CSV_FIELDS = [
    "noteId",
    "original_target",
    "original_english",
    "target_fixed",
    "english_fixed",
    "target_changed",
    "english_changed",
    "reason",
]

CORRECTION_PROMPT = _config.CORRECTION_PROMPT


def _build_prompt(target: str, english: str) -> str:
    """Substitute __TARGET__ and __ENGLISH__ placeholders in the correction prompt."""
    if not CORRECTION_PROMPT:
        raise RuntimeError(
            "CORRECTION_PROMPT is None in config.py — "
            "set it to a prompt string or skip correct.py."
        )
    return CORRECTION_PROMPT.replace("__TARGET__", target).replace("__ENGLISH__", english)


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


def load_existing_corrections() -> dict[int, dict]:
    """Return {noteId: row} of any previously-scored notes in corrections.csv."""
    if not CORRECTIONS_CSV.exists():
        return {}
    out: dict[int, dict] = {}
    with open(CORRECTIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[int(row["noteId"])] = row
            except (KeyError, ValueError):
                continue
    return out


def append_correction_row(row: dict) -> None:
    """Append one row to corrections.csv, writing the header if the file is new."""
    is_new = not CORRECTIONS_CSV.exists()
    with open(CORRECTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

_GEMINI_MODEL: GenerativeModel | None = None


def _get_model() -> GenerativeModel:
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
        _GEMINI_MODEL = GenerativeModel(GEMINI_MODEL_NAME)
    return _GEMINI_MODEL


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences the model sometimes adds despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        # drop the opening fence (with optional language tag) and trailing fence
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def score_note(target: str, english: str, max_retries: int = 3) -> tuple[dict, int, int]:
    """Ask Gemini whether the note has a typo / mismatch.

    Returns (parsed_json, input_tokens, output_tokens). Token counts are taken
    from the response usage metadata when available, else 0.
    """
    model = _get_model()
    prompt = _build_prompt(target, english)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _call_with_timeout(
                model.generate_content, prompt, timeout=GEMINI_TIMEOUT_S,
            )
            raw = _strip_code_fences(response.text)
            parsed = json.loads(raw)

            in_tok = 0
            out_tok = 0
            usage = getattr(response, "usage_metadata", None)
            if usage is not None:
                in_tok  = getattr(usage, "prompt_token_count", 0) or 0
                out_tok = getattr(usage, "candidates_token_count", 0) or 0

            return parsed, in_tok, out_tok
        except TimeoutError as exc:
            last_err = exc
            print(f"    [GEMINI] timeout — retry {attempt}/{max_retries}")
        except json.JSONDecodeError as exc:
            last_err = exc
            print(f"    [GEMINI] bad JSON — retry {attempt}/{max_retries}: {exc}")
        except Exception as exc:  # noqa: BLE001
            # Transient API errors: brief backoff and retry
            last_err = exc
            print(f"    [GEMINI] error '{exc}' — retry {attempt}/{max_retries}")
            time.sleep(2 * attempt)

    raise last_err if last_err else RuntimeError("score_note failed")


def estimated_cost(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1_000_000) * PRICE_INPUT_PER_M + (out_tok / 1_000_000) * PRICE_OUTPUT_PER_M


# ---------------------------------------------------------------------------
# Pass 1 — scan and propose
# ---------------------------------------------------------------------------

def find_candidate_notes(include_corrected: bool) -> list[dict]:
    parts = [f'"note:{ANKI_MODEL_NAME}"']
    if not include_corrected:
        parts.append(f"-tag:{CORRECTED_TAG}")
    query = " ".join(parts)
    note_ids = ankiconnect_request("findNotes", query=query)
    if not note_ids:
        return []
    return ankiconnect_request("notesInfo", notes=note_ids)


def interpret_result(parsed: dict, target: str, english: str) -> tuple[str, str, bool, bool]:
    """Normalize the LLM response into (target_fixed, english_fixed, target_changed, english_changed)."""
    target_ok  = bool(parsed.get("target_ok", True))
    english_ok = bool(parsed.get("english_ok", True))

    target_fixed_raw  = parsed.get("target_fixed")
    english_fixed_raw = parsed.get("english_fixed")

    target_fixed  = (target_fixed_raw  or "").strip()
    english_fixed = (english_fixed_raw or "").strip()

    # A change only counts if the LLM said NOT ok AND produced a different value.
    target_changed  = (not target_ok)  and bool(target_fixed)  and target_fixed  != target
    english_changed = (not english_ok) and bool(english_fixed) and english_fixed != english

    return target_fixed, english_fixed, target_changed, english_changed


def run_scan(args: argparse.Namespace) -> None:
    if not args.no_sync and not args.dry_run:
        sync_anki("before")

    notes = find_candidate_notes(include_corrected=args.include_corrected)
    print(f"Found {len(notes)} note(s) of type '{ANKI_MODEL_NAME}'"
          + (" (including already-corrected)" if args.include_corrected else " (untagged)"))

    existing = {} if args.rescore else load_existing_corrections()
    if existing and not args.rescore:
        print(f"  → {len(existing)} note(s) already in {CORRECTIONS_CSV.name}, will skip "
              f"(use --rescore to re-score them)")

    # Filter out notes already scored (unless --rescore)
    notes = [n for n in notes if n["noteId"] not in existing]

    if args.limit is not None:
        notes = notes[: args.limit]
        print(f"  → Limited to first {len(notes)}")

    if not notes:
        print("Nothing to do.")
        return

    total_in_tok = 0
    total_out_tok = 0
    scanned = 0
    proposed = 0
    target_fixes = 0
    english_fixes = 0
    failed = 0

    for i, note in enumerate(notes):
        nid = note["noteId"]
        fields = note["fields"]
        target  = clean_field(fields.get(TARGET_FIELD, {}).get("value", ""))
        english = clean_field(fields.get(ENGLISH_FIELD, {}).get("value", ""))

        if not target or not english:
            print(f"[{i+1}/{len(notes)}] note {nid}: SKIP — missing {TARGET_FIELD} or {ENGLISH_FIELD}")
            continue

        try:
            parsed, in_tok, out_tok = score_note(target, english)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i+1}/{len(notes)}] note {nid}: ERROR {exc}", file=sys.stderr)
            time.sleep(0.5)
            continue

        total_in_tok  += in_tok
        total_out_tok += out_tok
        scanned += 1

        t_fixed, e_fixed, t_changed, e_changed = interpret_result(parsed, target, english)
        reason = str(parsed.get("reason", "")).strip()

        tag = []
        if t_changed:
            tag.append("TGT")
            target_fixes += 1
        if e_changed:
            tag.append("ENG")
            english_fixes += 1
        tag_str = "+".join(tag) if tag else "ok"

        prefix = f"[{i+1}/{len(notes)}] note {nid} [{tag_str}]"
        if t_changed or e_changed:
            proposed += 1
            change_bits = []
            if t_changed:
                change_bits.append(f"TGT '{target}' → '{t_fixed}'")
            if e_changed:
                change_bits.append(f"ENG '{english}' → '{e_fixed}'")
            print(f"{prefix}  {'; '.join(change_bits)}  ({reason})")
            if not args.dry_run:
                append_correction_row({
                    "noteId":           nid,
                    "original_target":  target,
                    "original_english": english,
                    "target_fixed":     t_fixed if t_changed else "",
                    "english_fixed":    e_fixed if e_changed else "",
                    "target_changed":   "1" if t_changed else "0",
                    "english_changed":  "1" if e_changed else "0",
                    "reason":           reason,
                })
        else:
            # Quietly note "ok" results so the run output is scannable
            print(f"{prefix}  ok")

        # Cost guardrail every 25 notes
        if scanned % 25 == 0:
            cost_so_far = estimated_cost(total_in_tok, total_out_tok)
            projected   = cost_so_far / scanned * len(notes) if scanned else 0
            print(f"  [COST] scanned {scanned}, est. cost ${cost_so_far:.3f}, "
                  f"projected total ${projected:.3f}")
            if projected > COST_CAP_USD:
                print(f"  [COST] projected cost ${projected:.2f} exceeds cap "
                      f"${COST_CAP_USD:.2f} — aborting", file=sys.stderr)
                break

        # Gentle pacing — Gemini Flash RPM is high but be polite
        time.sleep(0.1)

    cost = estimated_cost(total_in_tok, total_out_tok)
    print(
        f"\nScan complete. Scanned {scanned}, proposed {proposed} correction(s) "
        f"({target_fixes} target, {english_fixes} English), failed {failed}. "
        f"Tokens: {total_in_tok} in / {total_out_tok} out  ≈ ${cost:.3f}."
    )
    if not args.dry_run and proposed > 0:
        print(f"Review {CORRECTIONS_CSV} then run: python correct.py --apply")


# ---------------------------------------------------------------------------
# Pass 2 — apply CSV to Anki
# ---------------------------------------------------------------------------

def add_tag_with_verification(nid: int, tag: str, attempts: int = 3) -> bool:
    """Add a tag and verify it stuck. Mirrors enrich.py's removeTags helper."""
    for attempt in range(1, attempts + 1):
        time.sleep(0.3)
        ankiconnect_request("addTags", notes=[nid], tags=tag)

        info = ankiconnect_request("notesInfo", notes=[nid])
        current_tags = info[0].get("tags", []) if info else []
        if tag in current_tags:
            return True
        print(f"  [TAG]  addTags '{tag}' didn't stick on attempt {attempt}, retrying…")

    print(f"  [TAG]  WARNING: could not add '{tag}' to note {nid} after {attempts} attempts",
          file=sys.stderr)
    return False


def run_apply(args: argparse.Namespace) -> None:
    if not CORRECTIONS_CSV.exists():
        print(f"ERROR: {CORRECTIONS_CSV} not found. Run pass 1 first.", file=sys.stderr)
        sys.exit(1)

    with open(CORRECTIONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Keep only rows that actually have a change
    rows = [r for r in rows if r.get("korean_changed") == "1" or r.get("english_changed") == "1"]

    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("No correction rows to apply.")
        return

    print(f"Applying {len(rows)} correction(s) from {CORRECTIONS_CSV.name}")

    if not args.no_sync:
        sync_anki("before")

    updated = 0
    korean_updated = 0
    english_updated = 0
    failed = 0

    for i, row in enumerate(rows):
        try:
            nid = int(row["noteId"])
        except (KeyError, ValueError) as exc:
            failed += 1
            print(f"[{i+1}/{len(rows)}] bad row, skipping: {exc}", file=sys.stderr)
            continue

        k_changed = row.get("target_changed") == "1"
        e_changed = row.get("english_changed") == "1"

        fields_to_update: dict[str, str] = {}
        if k_changed:
            fields_to_update[TARGET_FIELD] = row["target_fixed"]
            # Audio now stale — clear it so enrich.py regenerates
            fields_to_update[AUDIO_FIELD] = ""
        if e_changed:
            fields_to_update[ENGLISH_FIELD] = row["english_fixed"]

        print(f"[{i+1}/{len(rows)}] note {nid}: updating "
              f"{list(fields_to_update.keys())}")

        try:
            ankiconnect_request("updateNoteFields", note={
                "id": nid,
                "fields": fields_to_update,
            })
            add_tag_with_verification(nid, CORRECTED_TAG)
            if k_changed:
                add_tag_with_verification(nid, ENRICH_TAG)
                korean_updated += 1
            if e_changed:
                english_updated += 1
            updated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR on note {nid}: {exc}", file=sys.stderr)

    print(
        f"\nApply complete. Updated {updated}/{len(rows)} note(s) "
        f"(Korean: {korean_updated}, English: {english_updated}), failed {failed}."
    )

    if not args.no_sync and updated > 0:
        sync_anki("after")
        print("\nDon't forget to sync your phone's Anki app to pull the changes.")

    if korean_updated > 0:
        print(f"\n{korean_updated} note(s) had their {TARGET_FIELD} fixed — their "
              f"{AUDIO_FIELD} was cleared and the '{ENRICH_TAG}' tag was "
              f"re-added.\nRun  python enrich.py  to regenerate the audio.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-powered correction pass for Anki notes")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N matching notes (good for a smoke test)")
    p.add_argument("--apply", action="store_true",
                   help="Pass 2: apply corrections.csv to Anki via AnkiConnect")
    p.add_argument("--rescore", action="store_true",
                   help="Pass 1: re-score notes already present in corrections.csv")
    p.add_argument("--include-corrected", action="store_true",
                   help="Pass 1: also scan notes already tagged 'corrected'")
    p.add_argument("--dry-run", action="store_true",
                   help="Pass 1: don't write to corrections.csv, just print the LLM's verdict")
    p.add_argument("--no-sync", action="store_true",
                   help="Don't auto-sync with AnkiWeb before/after.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not CORRECTION_PROMPT:
        print("ERROR: CORRECTION_PROMPT is None in config.py — set a prompt or skip correct.py.",
              file=sys.stderr)
        sys.exit(1)

    if args.apply:
        run_apply(args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
