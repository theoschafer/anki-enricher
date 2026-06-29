"""
Anki Note Generation Pipeline
==============================
Reads <language>_vocab_cleaned.csv and for each word:
  1. Generates TTS audio (Google Cloud Text-to-Speech)
  2. Asks Gemini to craft an image prompt, then generates an image (Imagen 3)
  3. Exports inspection.csv
  4. Builds <language>_anki.apkg ready to import into Anki

Usage:
    python pipeline.py                  # full run (sample pause after word 5)
    python pipeline.py --sample-only    # generate first 5 words only, no .apkg
    python pipeline.py --apkg-only      # skip generation, rebuild .apkg from existing files
    python pipeline.py --limit 20       # process only the first N words

Language / Anki settings live in config.py — copy config.example.py to get started.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import genanki
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

# ---------------------------------------------------------------------------
# Load language config
# ---------------------------------------------------------------------------
try:
    import config
except ImportError:
    sys.exit(
        "ERROR: config.py not found.\n"
        "Copy config.example.py to config.py and fill in your settings."
    )

# Re-export so enrich.py / correct.py can `from pipeline import …` as before.
GCP_PROJECT     = config.GCP_PROJECT
GCP_REGION      = config.GCP_REGION
TTS_LANG        = config.TTS_LANG
TTS_VOICE       = config.TTS_VOICE
ANKI_MODEL_NAME = config.ANKI_MODEL_NAME
ANKI_MODEL_ID   = config.ANKI_MODEL_ID
TARGET_FIELD    = config.TARGET_FIELD
AUDIO_FIELD     = config.AUDIO_FIELD
ENGLISH_FIELD   = config.ENGLISH_FIELD
IMAGE_FIELD     = config.IMAGE_FIELD
TARGET_COLUMN   = config.TARGET_COLUMN

# ---------------------------------------------------------------------------
# Paths (derived from language name so they're easy to find)
# ---------------------------------------------------------------------------
INPUT_CSV   = Path(f"{config.LANGUAGE}_vocab_cleaned.csv")
AUDIO_DIR   = Path("audio")
IMAGE_DIR   = Path("images")
OUTPUT_CSV  = Path("inspection.csv")
OUTPUT_APKG = Path(f"{config.LANGUAGE}_anki.apkg")

SAMPLE_SIZE = 5   # words to generate before the interactive review pause

# AnkiConnect endpoint
ANKICONNECT_URL = "http://localhost:8765"

# ---------------------------------------------------------------------------
# Gemini image-prompt template (language-agnostic — works from English gloss)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    'Your job is to create a prompt, that will be given to an image generation model. '
    'Given the vocabulary word or expression "{word}", come up with a vivid, concrete visual scene '
    'that illustrates this concept. When in doubt about the meaning of the word or expression, '
    'use the most common meaning. Output only the image generation prompt, max 60 words. '
    'Avoid mentioning children. If children are essential to illustrate the concept, '
    'add an instruction to use cartoon style. '
    'Example — given "How many times": '
    '"A close-up of a wall covered in tally marks being counted by a hand, '
    'representing repeated occurrences, minimal background, clear and symbolic." '
    'Example — given "hesitate": '
    '"A realistic image of a person frozen mid-step at a crosswalk while the world '
    'around them moves in motion blur, representing hesitation, anxious expression."'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def word_id(idx: int) -> str:
    return f"word_{idx:04d}"


def load_vocab(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[ENGLISH_FIELD]  = df[ENGLISH_FIELD].str.strip()
    df[TARGET_COLUMN]  = df[TARGET_COLUMN].str.strip()
    df["deck"]         = df["deck"].str.strip()
    df["id"]           = [word_id(i + 1) for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def generate_tts(text: str, out_path: Path) -> None:
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=TTS_LANG,
        name=TTS_VOICE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    out_path.write_bytes(response.audio_content)


# ---------------------------------------------------------------------------
# Timeout helper — wraps a blocking call so it can't hang forever.
#
# Uses a daemon thread (NOT a ThreadPoolExecutor): when the call times out
# we don't try to join the worker thread, we just abandon it. Daemon threads
# die when the main process exits, so we never block on shutdown. The hung
# gRPC call leaks a small amount of memory; harmless at our scale.
# ---------------------------------------------------------------------------

def _call_with_timeout(func, *args, timeout: float, **kwargs):
    result_box: list = []
    error_box: list = []

    def _runner():
        try:
            result_box.append(func(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # Worker still running — abandon it. It's a daemon, so it'll die
        # when the process exits. Don't try to join.
        raise TimeoutError(f"call timed out after {timeout}s")

    if error_box:
        raise error_box[0]
    return result_box[0]


# ---------------------------------------------------------------------------
# Gemini prompt crafting (via Vertex AI — uses same credentials as Imagen)
# ---------------------------------------------------------------------------

GEMINI_TIMEOUT_S = 60


def craft_image_prompt(english_word: str, max_retries: int = 3) -> str:
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
    model = GenerativeModel("gemini-2.5-flash")
    user_prompt = PROMPT_TEMPLATE.format(word=english_word)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _call_with_timeout(
                model.generate_content, user_prompt, timeout=GEMINI_TIMEOUT_S
            )
            text = response.text.strip()
            # Strip markdown bold/italic and any leading/trailing whitespace
            text = text.replace("**", "").replace("*", "").replace("`", "")
            return text
        except TimeoutError as exc:
            last_err = exc
            print(f"    [GEMINI] timeout — retry {attempt}/{max_retries}")
        except Exception as exc:
            raise exc
    raise last_err if last_err else RuntimeError("craft_image_prompt failed")


# ---------------------------------------------------------------------------
# Imagen 3 image generation
# ---------------------------------------------------------------------------

IMAGEN_TIMEOUT_S = 90


def generate_image(prompt: str, out_path: Path, max_retries: int = 5) -> None:
    from vertexai.preview.vision_models import ImageGenerationModel
    from google.api_core.exceptions import ResourceExhausted

    vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    def _gen(p: str):
        return _call_with_timeout(
            model.generate_images,
            prompt=p,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_few",
            person_generation="allow_adult",
            timeout=IMAGEN_TIMEOUT_S,
        )

    delay = 15  # seconds, doubles on each retry
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            images = _gen(prompt)
            if not images:
                # Safety filter blocked — retry once with a softened prompt
                if attempt == 0:
                    soft_prompt = (
                        "A simple, friendly illustration in flat cartoon style of: "
                        + prompt
                    )
                    images = _gen(soft_prompt)
                if not images:
                    raise RuntimeError("Imagen returned 0 images (safety filter)")
            images[0].save(location=str(out_path), include_generation_parameters=False)
            return
        except ResourceExhausted as exc:
            last_err = exc
            print(f"    [IMG]   429 quota — retry {attempt+1}/{max_retries} in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
        except TimeoutError as exc:
            last_err = exc
            print(f"    [IMG]   timeout — retry {attempt+1}/{max_retries}")
        except Exception as exc:
            raise exc
    raise last_err if last_err else RuntimeError("generate_image failed")


# ---------------------------------------------------------------------------
# Process a single word
# ---------------------------------------------------------------------------

def process_word(row: pd.Series, audio_dir: Path, image_dir: Path) -> dict:
    wid     = row["id"]
    target  = row[TARGET_COLUMN]
    english = row[ENGLISH_FIELD]
    deck    = row["deck"]
    tags    = row.get("tags", "")

    audio_path = audio_dir / f"{wid}.mp3"
    image_path = image_dir / f"{wid}.jpg"

    # TTS
    audio_ok = False
    audio_generated = False
    if audio_path.exists():
        print(f"    [TTS]   {wid} already exists, skipping")
        audio_ok = True
    else:
        try:
            generate_tts(target, audio_path)
            audio_ok = True
            audio_generated = True
            print(f"    [TTS]   {wid} → {audio_path}")
        except Exception as exc:
            print(f"    [TTS]   ERROR for {wid}: {exc}", file=sys.stderr)

    # Image prompt + generation
    image_prompt = ""
    image_ok = False
    image_generated = False
    if image_path.exists():
        print(f"    [IMG]   {wid} already exists, skipping")
        image_ok = True
        # Try to recover the prompt from existing inspection.csv
        if OUTPUT_CSV.exists():
            try:
                existing = pd.read_csv(OUTPUT_CSV)
                match = existing.loc[existing["id"] == wid, "image_prompt"]
                if not match.empty:
                    image_prompt = match.iloc[0]
            except Exception:
                pass
    else:
        try:
            image_prompt = craft_image_prompt(english)
            print(f"    [GEMINI] prompt → {image_prompt[:80]}…")
            generate_image(image_prompt, image_path)
            image_ok = True
            image_generated = True
            print(f"    [IMG]   {wid} → {image_path}")
        except Exception as exc:
            print(f"    [IMG]   ERROR for {wid}: {exc}", file=sys.stderr)

    return {
        "id":           wid,
        "deck":         deck,
        ENGLISH_FIELD:  english,
        TARGET_COLUMN:  target,
        "tags":         tags,
        "audio_file":   str(audio_path) if audio_ok else "",
        "image_file":   str(image_path) if image_ok else "",
        "image_prompt": image_prompt,
        "_generated":   audio_generated or image_generated,
    }


# ---------------------------------------------------------------------------
# Inspection CSV
# ---------------------------------------------------------------------------

def write_inspection_csv(records: list[dict], path: Path) -> None:
    fieldnames = ["id", "deck", ENGLISH_FIELD, TARGET_COLUMN, "tags",
                  "audio_file", "image_file", "image_prompt"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[CSV] Inspection file written → {path}  ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Anki .apkg generation
# ---------------------------------------------------------------------------

ANKI_DECK_ID_BASE = 1_234_567_890   # deck IDs are derived from this + deck name hash


def deck_id_from_name(name: str) -> int:
    """Derive a stable integer deck ID from the deck name string."""
    return ANKI_DECK_ID_BASE + hash(name) % (1 << 28)


def _anki_ref(field: str) -> str:
    """Anki template reference: {{FieldName}}"""
    return "{{" + field + "}}"


def _anki_cond(field: str, content: str) -> str:
    """Anki conditional block: {{#Field}}content{{/Field}}"""
    return "{{#" + field + "}}" + content + "{{/" + field + "}}"


def build_apkg(records: list[dict], out_path: Path) -> None:
    model = genanki.Model(
        ANKI_MODEL_ID,
        ANKI_MODEL_NAME,
        fields=[
            {"name": ENGLISH_FIELD},
            {"name": TARGET_FIELD},
            {"name": AUDIO_FIELD},
            {"name": IMAGE_FIELD},
        ],
        templates=[
            {
                "name": f"{TARGET_FIELD} → {ENGLISH_FIELD}",
                "qfmt": (
                    _anki_ref(TARGET_FIELD) + "<br>"
                    + _anki_cond(AUDIO_FIELD, f"[sound:{_anki_ref(AUDIO_FIELD)}]")
                ),
                "afmt": (
                    "{{FrontSide}}<hr id=answer>"
                    + _anki_ref(ENGLISH_FIELD) + "<br>"
                    + _anki_cond(IMAGE_FIELD, f"<img src='{_anki_ref(IMAGE_FIELD)}'>")
                ),
            },
            {
                "name": f"{ENGLISH_FIELD} → {TARGET_FIELD}",
                "qfmt": (
                    _anki_ref(ENGLISH_FIELD) + "<br>"
                    + _anki_cond(IMAGE_FIELD, f"<img src='{_anki_ref(IMAGE_FIELD)}'>")
                ),
                "afmt": (
                    "{{FrontSide}}<hr id=answer>"
                    + _anki_ref(TARGET_FIELD) + "<br>"
                    + _anki_cond(AUDIO_FIELD, f"[sound:{_anki_ref(AUDIO_FIELD)}]")
                ),
            },
        ],
    )

    # One genanki Deck per unique deck name
    decks: dict[str, genanki.Deck] = {}
    media_files: list[str] = []

    for rec in records:
        deck_name = rec["deck"]
        if deck_name not in decks:
            decks[deck_name] = genanki.Deck(
                deck_id_from_name(deck_name),
                deck_name,
            )

        # Resolve media filenames (just the basename — Anki finds them by name)
        audio_ref = ""
        if rec["audio_file"] and Path(rec["audio_file"]).exists():
            audio_ref = Path(rec["audio_file"]).name
            media_files.append(rec["audio_file"])

        image_ref = ""
        if rec["image_file"] and Path(rec["image_file"]).exists():
            image_ref = Path(rec["image_file"]).name
            media_files.append(rec["image_file"])

        # Wrap with Anki special markup so the existing templates render correctly.
        sound_field = f"[sound:{audio_ref}]" if audio_ref else ""
        image_field = f'<img src="{image_ref}">' if image_ref else ""

        note = genanki.Note(
            model=model,
            fields=[
                rec[ENGLISH_FIELD],
                rec[TARGET_COLUMN],
                sound_field,
                image_field,
            ],
            tags=rec["tags"].split() if rec.get("tags") else [],
            guid=genanki.guid_for(rec["id"]),  # stable GUID based on our unique ID
        )
        decks[deck_name].add_note(note)

    package = genanki.Package(list(decks.values()))
    package.media_files = list(set(media_files))  # deduplicate
    package.write_to_file(str(out_path))
    print(f"[APKG] Package written → {out_path}  ({len(records)} notes, {len(decks)} decks)")


# ---------------------------------------------------------------------------
# AnkiConnect push (uses existing note type by name, no duplication)
# ---------------------------------------------------------------------------

def ankiconnect_request(action: str, **params):
    """Send a request to AnkiConnect and return the result, raising on error."""
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode("utf-8")
    req = urllib.request.Request(ANKICONNECT_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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


def sync_anki(label: str) -> None:
    """Trigger AnkiWeb sync via AnkiConnect (same as clicking the sync button)."""
    print(f"[ANKI] Syncing with AnkiWeb ({label})...")
    try:
        ankiconnect_request("sync")
        print(f"[ANKI] Sync ({label}) complete.")
    except RuntimeError as exc:
        print(f"[ANKI] Sync ({label}) failed: {exc}", file=sys.stderr)


def push_via_ankiconnect(records: list[dict]) -> None:
    """Push notes and media into the running Anki app via AnkiConnect."""

    # 1. Sanity-check the note type exists and has the expected fields
    models = ankiconnect_request("modelNamesAndIds")
    if ANKI_MODEL_NAME not in models:
        raise RuntimeError(
            f"Note type '{ANKI_MODEL_NAME}' not found in Anki. "
            f"Available models: {list(models.keys())}"
        )
    fields = ankiconnect_request("modelFieldNames", modelName=ANKI_MODEL_NAME)
    for expected_field in (ENGLISH_FIELD, TARGET_FIELD, AUDIO_FIELD, IMAGE_FIELD):
        if expected_field not in fields:
            raise RuntimeError(
                f"Note type '{ANKI_MODEL_NAME}' is missing field '{expected_field}'. "
                f"Found fields: {fields}"
            )

    # 2. Ensure all decks exist
    deck_names = sorted({rec["deck"] for rec in records})
    existing_decks = set(ankiconnect_request("deckNames"))
    for d in deck_names:
        if d not in existing_decks:
            ankiconnect_request("createDeck", deck=d)
            print(f"[ANKI] Created deck: {d}")

    # 3. Upload media files (audio + images)
    uploaded = 0
    for rec in records:
        for path_str in (rec.get("audio_file"), rec.get("image_file")):
            if not path_str:
                continue
            p = Path(path_str)
            if not p.exists():
                continue
            data_b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            ankiconnect_request("storeMediaFile", filename=p.name, data=data_b64)
            uploaded += 1
    print(f"[ANKI] Uploaded {uploaded} media files")

    # 4. Add notes
    added, skipped, failed = 0, 0, 0
    for rec in records:
        audio_ref = Path(rec["audio_file"]).name if rec.get("audio_file") else ""
        image_ref = Path(rec["image_file"]).name if rec.get("image_file") else ""
        sound_field = f"[sound:{audio_ref}]" if audio_ref else ""
        image_field = f'<img src="{image_ref}">' if image_ref else ""

        note_payload = {
            "deckName": rec["deck"],
            "modelName": ANKI_MODEL_NAME,
            "fields": {
                ENGLISH_FIELD: rec[ENGLISH_FIELD],
                TARGET_FIELD:  rec[TARGET_COLUMN],
                AUDIO_FIELD:   sound_field,
                IMAGE_FIELD:   image_field,
            },
            "tags": rec["tags"].split() if rec.get("tags") else [],
            "options": {
                "allowDuplicate": False,
                "duplicateScope": "deck",
            },
        }
        try:
            ankiconnect_request("addNote", note=note_payload)
            added += 1
        except RuntimeError as exc:
            err = str(exc)
            if "duplicate" in err.lower():
                skipped += 1
            else:
                failed += 1
                print(f"  [ANKI] FAIL {rec['id']} ({rec[ENGLISH_FIELD]}): {err}")

    print(f"[ANKI] Added {added}, skipped {skipped} (duplicates), failed {failed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anki enrichment pipeline")
    p.add_argument("--sample", type=int, nargs="?", const=SAMPLE_SIZE, default=None,
                   metavar="N",
                   help="Generate/push the first N words only (no interactive pause). "
                        f"Use --sample with no value for {SAMPLE_SIZE} words.")
    p.add_argument("--sample-only", dest="sample", action="store_const",
                   const=SAMPLE_SIZE,
                   help=f"Alias for --sample {SAMPLE_SIZE}")
    p.add_argument("--apkg-only", action="store_true",
                   help="Skip generation, rebuild .apkg from existing files")
    p.add_argument("--push-only", action="store_true",
                   help="Skip generation, push existing files to Anki via AnkiConnect")
    p.add_argument("--push", action="store_true",
                   help="After generation, push notes to Anki via AnkiConnect "
                        "(uses your existing note type, no .apkg). "
                        "Requires Anki running with AnkiConnect addon.")
    p.add_argument("--no-sync", action="store_true",
                   help="When pushing to Anki, don't auto-sync with AnkiWeb "
                        "before/after.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    AUDIO_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    df = load_vocab(INPUT_CSV)
    print(f"Loaded {len(df)} words from {INPUT_CSV}")

    # --apkg-only / --push-only: skip generation, use existing inspection.csv
    if args.apkg_only or args.push_only:
        if not OUTPUT_CSV.exists():
            print(f"ERROR: {OUTPUT_CSV} not found. Run a generation pass first.", file=sys.stderr)
            sys.exit(1)
        records = pd.read_csv(OUTPUT_CSV).fillna("").to_dict("records")
        if args.push_only:
            if not args.no_sync:
                sync_anki("before")
            push_via_ankiconnect(records)
            if not args.no_sync:
                sync_anki("after")
                print("\nDon't forget to sync your phone's Anki app to pull the changes.")
        else:
            build_apkg(records, OUTPUT_APKG)
        return

    # --sample N or full run
    sample_mode = args.sample is not None
    if sample_mode:
        df = df.head(args.sample)
        print(f"  → Sample mode: processing first {args.sample} word(s)")
    total = len(df)
    records: list[dict] = []

    print(f"\n{'='*60}")
    if sample_mode:
        print(f"PHASE 1: Generating {args.sample} word(s)")
    else:
        print(f"PHASE 1: Generating first {SAMPLE_SIZE} words (sample for review)")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(df.iterrows()):
        # Interactive pause only on full runs, after the first SAMPLE_SIZE words
        pausing = (i == SAMPLE_SIZE and not sample_mode)

        if pausing:
            # Write intermediate CSV before pausing
            write_inspection_csv(records, OUTPUT_CSV)
            print(f"\n{'='*60}")
            print("SAMPLE COMPLETE — please review the first {SAMPLE_SIZE} files:".format(SAMPLE_SIZE=SAMPLE_SIZE))
            for r in records:
                if r["audio_file"]:
                    print(f"  Audio: {r['audio_file']}")
                if r["image_file"]:
                    print(f"  Image: {r['image_file']}")
            print(f"\nOpen the files above to check audio quality and image quality.")
            input("Press Enter to continue with all remaining words (Ctrl-C to abort)…\n")
            print(f"{'='*60}")
            print(f"PHASE 2: Processing all {total} words")
            print(f"{'='*60}")

        print(f"[{i+1}/{total}] {row['id']}  {row[TARGET_COLUMN]}  ({row[ENGLISH_FIELD]})")
        rec = process_word(row, AUDIO_DIR, IMAGE_DIR)
        records.append(rec)

        # Only sleep when an actual API call was made — skipped (cached) words
        # cost nothing and don't need throttling. Imagen quota is ~5/min on
        # new projects, so 8s between real generations stays under the limit.
        if rec.get("_generated"):
            time.sleep(8)

    # Write final inspection CSV
    write_inspection_csv(records, OUTPUT_CSV)

    # Final phase: push to Anki (preferred) or build .apkg
    phase = "2" if sample_mode else "3"
    print(f"\n{'='*60}")
    if args.push:
        print(f"PHASE {phase}: Pushing to Anki via AnkiConnect")
        print(f"{'='*60}")
        if not args.no_sync:
            sync_anki("before")
        push_via_ankiconnect(records)
        if not args.no_sync:
            sync_anki("after")
        if sample_mode:
            print(f"\nSample run complete ({args.sample} word(s)). Open Anki to verify.")
            print(f"  When happy, re-run without --sample to process all words.")
        else:
            print(f"\nDone! Notes are in Anki, browse to verify.")
        if not args.no_sync:
            print("Don't forget to sync your phone's Anki app to pull the changes.")
    else:
        print(f"PHASE {phase}: Building .apkg")
        print(f"{'='*60}")
        apkg_path = OUTPUT_APKG.with_name(f"{config.LANGUAGE}_anki_sample.apkg") if sample_mode else OUTPUT_APKG
        build_apkg(records, apkg_path)
        if sample_mode:
            print(f"\nSample run complete ({args.sample} word(s)). Test import: open Anki and import {apkg_path}")
            print(f"  When happy, re-run without --sample to process all words.")
        else:
            print(f"\nDone! Import {apkg_path} into Anki via File → Import.")


if __name__ == "__main__":
    main()
