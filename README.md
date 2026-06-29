# Korean Anki Note Generation Pipeline

Generates Korean vocabulary Anki notes with:
- **Korean TTS audio** via Google Cloud Text-to-Speech (`ko-KR-Chirp3-HD-Achernar`)
- **Word illustration images** via Gemini (prompt crafting) + Imagen 3 (image generation)
- **Inspection CSV** for reviewing results before import
- **`.apkg` file** ready to import into Anki

Plus maintenance / automation scripts:
- **`enrich.py`** — fill missing audio + image on existing notes tagged `pending-enrichment` (great for cards added on the phone)
- **`clean_articles.py`** — strip leading "A " / "An " from English fields in bulk
- **AnkiWeb auto-sync** — every script that touches the live collection syncs before and after via AnkiConnect
- **Scheduled runs** — `enrich.py` can run autonomously at a set time of day via `launchd`

---

## Prerequisites

- Python 3.10+
- A Google Cloud account with a billing account enabled
- Anki desktop (for the final import step)

---

## Step 1 — Google Cloud Project Setup

### 1.1 Create a project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown (top left) → **New Project**
3. Name it (e.g. `korean-anki`) and click **Create**
4. Make sure this project is selected in the top bar

### 1.2 Enable billing

Imagen 3 and Cloud TTS require a billing account.

1. In the left sidebar go to **Billing**
2. Link a billing account to the project (you will not be charged unless you exceed free-tier limits; TTS and Gemini have generous free tiers)

### 1.3 Enable the required APIs

Run the following in your terminal (requires `gcloud` CLI — see step 1.4 if not installed):

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable texttospeech.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

Or enable them manually in the console:
- [Cloud Text-to-Speech API](https://console.cloud.google.com/apis/library/texttospeech.googleapis.com)
- [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) (for Imagen 3)
- [Generative Language API](https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com) (for Gemini)

### 1.4 Install the gcloud CLI (if needed)

```bash
# macOS via Homebrew
brew install --cask google-cloud-sdk

# Then authenticate
gcloud auth login
gcloud auth application-default login
```

The second command sets up **Application Default Credentials (ADC)**, which is the recommended way to authenticate for local development. With ADC set up you do **not** need a service account JSON key file.

### 1.5 Verify Imagen 3 access

Imagen 3 (`imagen-3.0-generate-002`) is generally available on Vertex AI with billing enabled. If you hit a permission error when running the pipeline, go to:

[Vertex AI → Model Garden → Imagen 3](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration)

And click **Enable** / **Request access** if prompted.

---

## Step 2 — Python Environment Setup

The pipeline runs in an isolated virtual environment so it does not affect other Python packages on your machine.

```bash
cd /path/to/anki-enricher

# Create virtual environment
python3 -m venv .venv

# Activate it (you must do this each time you open a new terminal)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

To deactivate the environment when done:

```bash
deactivate
```

---

## Step 3 — Configure the Pipeline

Open `pipeline.py` and set your Google Cloud project ID at the top:

```python
GCP_PROJECT = "your-project-id"   # ← change this
GCP_REGION  = "us-central1"       # Imagen 3 is available here
```

Make sure your ADC credentials are active:

```bash
gcloud auth application-default login
```

---

## Step 4 — Run the Pipeline

```bash
source .venv/bin/activate   # if not already active

python pipeline.py
```

### What happens

1. Loads `korean_vocab_cleaned.csv` (~777 words)
2. Assigns each word a unique ID (`word_0001`, `word_0002`, …)
3. For the **first 5 words only**:
   - Generates TTS audio → `audio/word_000N.mp3`
   - Asks Gemini to craft an image prompt, then generates image → `images/word_000N.jpg`
4. **Pauses** and prints file paths — open the files to verify quality
5. Press **Enter** to continue with all remaining words (or `Ctrl-C` to abort)
6. Exports `inspection.csv` with every field including the image prompt used
7. Builds `korean_anki.apkg` — the final import file

### Flags

```bash
# Sample run (5 words) — generates audio + images + inspection.csv + sample .apkg
python pipeline.py --sample

# Sample run with a custom size (e.g. 20 words)
python pipeline.py --sample 20

# Sample run, but push directly to Anki via AnkiConnect (recommended — uses
# your existing note type, no duplicate model created)
python pipeline.py --sample 20 --push

# Push existing media to Anki without regenerating
python pipeline.py --push-only

# Skip generation, only rebuild .apkg from existing files
python pipeline.py --apkg-only
```

> **Recommended workflow:** use `--push` instead of `.apkg`. It looks up the
> `theo-korean-advanced` note type by name in your running Anki collection, so
> it always uses your existing card templates rather than creating a duplicate
> model. Requires the AnkiConnect addon (code `2055492159`).

---

## Step 5 — Import into Anki

1. Open Anki desktop
2. Make sure your custom note type `theo-advanced-korean` exists (it should already if you've used it before)
3. Go to **File → Import** and select `korean_anki.apkg`
4. Notes will be placed in their original decks (e.g. `02-06`, `duolingo`, `DL 3-1`)

> **Note:** The `.apkg` embeds all audio and image files. No manual media copying needed.

---

## Step 6 — Maintenance Scripts

Two helper scripts operate directly on your live Anki collection via AnkiConnect (no `.apkg` round-trip). Both **auto-sync with AnkiWeb before and after** they run, so you don't need to click the sync button manually on the Mac.

### 6.1 `enrich.py` — fill audio + images for new notes

Designed for the workflow "add a card on the phone, fill in audio + image automatically on the Mac later".

**On the phone:** add a new card and either:
- tag it with `pending-enrichment`, or
- leave the `KoreanPronunciation` and/or `NormalImage` fields empty and tag it later from the desktop browser.

**On the Mac:**

```bash
source .venv/bin/activate
python enrich.py
```

What it does:

1. Syncs Anki with AnkiWeb (pulls down anything new from your phone).
2. Finds all `theo-korean-advanced` notes tagged `pending-enrichment`, across every deck.
3. For each one, generates TTS for the Korean field (only if `KoreanPronunciation` is empty) and an illustration for the English field (only if `NormalImage` is empty), uploads the media, and writes the fields back in place — **note IDs and review history are preserved**.
4. Removes the `pending-enrichment` tag.
5. Syncs back up to AnkiWeb so you can pull the changes on the phone.

Flags:

```bash
python enrich.py --limit 3      # only process the first 3 (good first try)
python enrich.py --dry-run      # list candidates, no API calls
python enrich.py --keep-tag     # don't remove the tag after success
python enrich.py --no-sync      # skip the auto-sync calls
```

### 6.2 `clean_articles.py` — strip leading "A" / "An" from English fields

Removes a leading `A ` or `An ` from the English field (case-insensitive, must be followed by whitespace so words like "Apple" / "Antenna" are left alone), and capitalises the first remaining letter. If the result would collide with an existing English value (case-insensitive), ` DUP` is appended so you can manually clean those up later.

```bash
python clean_articles.py --dry-run   # preview only, no changes
python clean_articles.py             # preview + ask for confirmation
python clean_articles.py --yes       # apply without prompting
python clean_articles.py --no-sync   # skip the auto-sync calls
```

Examples:
- `"An airport"` → `"Airport"`
- `"a pen"` → `"Pen"` (or `"Pen DUP"` if you already had a card called `"Pen"`)

### 6.3 AnkiWeb auto-sync (how it works)

`enrich.py`, `clean_articles.py`, and `pipeline.py --push` / `--push-only` all call AnkiConnect's `sync` action — the same operation as clicking the sync button in the Anki toolbar — once before they read from the collection and once after they're done writing. Anki desktop must be running and signed into AnkiWeb. Use `--no-sync` on any of them to opt out.

After running, you still need to **sync the phone** to pull the changes — the scripts print a reminder.

---

## Step 7 — Scheduled Autonomous Runs (macOS `launchd`)

`enrich.py` can run on a schedule, so the loop becomes "tag a card on the phone → wait → it's enriched". Scheduling is set up via a `launchd` LaunchAgent (Apple's recommended scheduler — more reliable than cron on macOS).

Two files in `scheduling/`:

| File | Purpose |
|---|---|
| `scheduling/run_enrich.sh` | Wrapper that opens Anki if it's not running, runs `enrich.py` via the project's venv Python, and appends timestamped output to `logs/enrich.log`. |
| `scheduling/com.theo.korean-enrich.plist` | LaunchAgent definition. Default: every day at 11:00 local time. |

### 7.1 Install (one-time)

```bash
# Symlink so future edits to the plist take effect after reload
ln -sf /path/to/anki-enricher/scheduling/com.theo.korean-enrich.plist \
       ~/Library/LaunchAgents/com.theo.korean-enrich.plist

# Register with launchd
launchctl load ~/Library/LaunchAgents/com.theo.korean-enrich.plist
```

That's it. From then on, every day at 11:00 your Mac's local time, the agent runs.

### 7.2 Test it

```bash
# Trigger it now via launchd (uses the same env launchd will use at 11am)
launchctl start com.theo.korean-enrich

# Or run the wrapper directly (bypasses launchd)
/path/to/anki-enricher/scheduling/run_enrich.sh

# Watch the log
tail -f /path/to/anki-enricher/logs/enrich.log
```

### 7.3 Manage it

```bash
# Disable
launchctl unload ~/Library/LaunchAgents/com.theo.korean-enrich.plist

# Re-enable after editing the plist
launchctl unload ~/Library/LaunchAgents/com.theo.korean-enrich.plist
launchctl load   ~/Library/LaunchAgents/com.theo.korean-enrich.plist

# Check it's registered
launchctl list | grep korean-enrich
```

To change the schedule, edit `Hour` / `Minute` in the plist and reload. For multiple times per day, replace the `StartCalendarInterval` dict with an array of dicts — `launchd` accepts both.

### 7.4 Caveats

- **The Mac has to be awake at the scheduled time.** If it's asleep, `launchd` will run the job the next time the Mac wakes (catch-up behaviour). To wake the Mac on a schedule, see `man pmset` (`pmset repeat wakeorpoweron MTWRFSU 10:55:00`).
- **Timezone**: `StartCalendarInterval` uses the Mac's system timezone. If your Mac is on Pacific time, `Hour=11` is 11am PST in winter / 11am PDT in summer (auto-adjusts for DST).
- **Anki must launch successfully.** The wrapper sleeps 25s after `open -a Anki` to give Anki + AnkiConnect time to come online. Increase that value in `run_enrich.sh` on slower machines.
- **Google credentials**: ADC at `~/.config/gcloud/application_default_credentials.json` is picked up automatically — no extra config needed under `launchd`.
- **First-run permission prompt**: macOS may prompt the first time the agent tries to control Anki. Approve it once. If you hit a wall, check System Settings → Privacy & Security → Automation / Full Disk Access.

---

## Output Files Reference

| File/Folder | Description |
|---|---|
| `audio/word_NNNN.mp3` | Korean TTS pronunciation (pipeline-generated) |
| `audio/note_<id>.mp3` | Korean TTS pronunciation (`enrich.py`-generated, keyed by Anki note ID) |
| `images/word_NNNN.jpg` | AI-generated illustration (pipeline-generated) |
| `images/note_<id>.jpg` | AI-generated illustration (`enrich.py`-generated) |
| `inspection.csv` | Full data table for review |
| `korean_anki.apkg` | Anki import package |
| `logs/enrich.log` | Output of scheduled `enrich.py` runs (timestamped) |
| `logs/launchd.{out,err}.log` | `launchd`'s own stdout/stderr for the agent |

---

## Cost Estimates (approximate)

| Service | Free tier | Cost beyond |
|---|---|---|
| Google Cloud TTS (Chirp3 HD) | 1M chars/month | ~$16/1M chars |
| Gemini 2.0 Flash | 1M tokens/day free | Very cheap |
| Imagen 3 | None | ~$0.04/image |

For 777 words: TTS is well within free tier. Imagen 3 ≈ **$31** total. Consider running `--sample-only` first to verify quality.
