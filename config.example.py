# config.example.py — template for configuring the Anki enrichment pipeline.
#
# SETUP: Copy this file to config.py and fill in your values.
#        config.py is gitignored so your personal settings stay local.
#
# The pipeline generates Anki notes with TTS audio + AI illustrations for any
# language pair. The only language-specific settings are below.

LANGUAGE = "korean"    # used as a label in output filenames; e.g. "japanese", "french"

# ── Google Cloud ─────────────────────────────────────────────────────────────
GCP_PROJECT = "your-project-id"   # ← your Google Cloud project ID
GCP_REGION  = "us-central1"       # Imagen 3 is available in us-central1

# ── Text-to-Speech ───────────────────────────────────────────────────────────
# Full voice name list: https://cloud.google.com/text-to-speech/docs/voices
# Examples:
#   Korean:   TTS_LANG="ko-KR"  TTS_VOICE="ko-KR-Chirp3-HD-Achernar"
#   Japanese: TTS_LANG="ja-JP"  TTS_VOICE="ja-JP-Chirp3-HD-Aoede"
#   French:   TTS_LANG="fr-FR"  TTS_VOICE="fr-FR-Chirp3-HD-Aoede"
#   Spanish:  TTS_LANG="es-ES"  TTS_VOICE="es-ES-Chirp3-HD-Aoede"
#   Mandarin: TTS_LANG="cmn-CN" TTS_VOICE="cmn-CN-Chirp3-HD-Aoede"
TTS_LANG  = "ko-KR"
TTS_VOICE = "ko-KR-Chirp3-HD-Achernar"

# ── Anki note type ───────────────────────────────────────────────────────────
# Must already exist in your Anki collection. The pipeline looks it up by name.
# You can create a note type in Anki: Tools → Manage Note Types → Add.
ANKI_MODEL_NAME = "my-language-deck"
ANKI_MODEL_ID   = 1_234_567_890_001   # pick any unique integer; used for .apkg generation

# ── Field names (must match your Anki note type exactly) ─────────────────────
TARGET_FIELD  = "Korean"        # the target-language word/phrase (TTS input)
AUDIO_FIELD   = "Pronunciation" # filled with [sound:…]  by pipeline / enrich.py
ENGLISH_FIELD = "English"       # native-language gloss (used for image prompts)
IMAGE_FIELD   = "Image"         # filled with <img …> by pipeline / enrich.py

# ── Input CSV ────────────────────────────────────────────────────────────────
# The column in your vocab CSV that holds the target-language words.
# Other required columns: "English", "deck" (and optionally "tags").
TARGET_COLUMN = "Korean"

# ── LLM correction prompt (correct.py) ───────────────────────────────────────
# Gemini will use this prompt to check for spelling errors and meaning mismatches.
# Use __TARGET__ and __ENGLISH__ as placeholders — they are substituted at runtime.
# The model must return JSON with keys: target_ok, target_fixed, english_ok,
# english_fixed, reason.
# Set to None to skip the correction pass entirely.
CORRECTION_PROMPT = (
    'You are a language expert reviewing flashcards. Given a word or expression '
    'in the target language and its English meaning, decide:\n'
    '  1. Is the target-language word spelled correctly?\n'
    '  2. Does the English match the word\'s meaning?\n'
    '\n'
    'Return STRICT JSON, no markdown, no commentary:\n'
    '  {"target_ok": bool, "target_fixed": string|null, '
    '"english_ok": bool, "english_fixed": string|null, '
    '"reason": string}\n'
    '\n'
    'Rules:\n'
    '  - target_fixed: corrected spelling, or null if target_ok=true.\n'
    '  - english_fixed: corrected translation, or null if english_ok=true.\n'
    '  - reason: one short sentence. Say "ok" if both are fine.\n'
    '\n'
    'Input:\n'
    '  Target:  __TARGET__\n'
    '  English: __ENGLISH__\n'
)
