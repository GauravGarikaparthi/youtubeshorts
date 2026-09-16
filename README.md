# Auto YouTube Uploader

Fully automated pipeline: finds a trending topic → writes a script → generates
a voiceover → pulls stock footage → assembles a video → uploads to YouTube.
Runs every four hours via GitHub Actions.

## How it works

```
trend_fetch.py       -> picks a trending topic (Google Trends, with fallback list)
generate_script.py    -> Groq (gpt-oss-120b) writes title/description/tags/narration/keywords
generate_voiceover.py -> Kokoro (local TTS; Piper fallback) turns narration into a WAV
fetch_visuals.py      -> Pexels downloads matching stock clips
assemble_video.py     -> ffmpeg stitches clips + voiceover + title card
generate_thumbnail.py -> Pillow makes a 1280x720 thumbnail from a video frame
upload_youtube.py     -> YouTube Data API v3 uploads the final video
main.py               -> runs all of the above in order
```

## One-time setup (about 30-45 minutes)

### 1. Groq API key (script writing, free)
Sign up free at https://console.groq.com/ → API Keys → Create API Key. No
credit card required.

### 2. Voiceover: nothing to sign up for
Voiceover uses [Kokoro](https://github.com/thewh1teagle/kokoro-onnx), a local,
open-source TTS engine. The GitHub workflow downloads and caches its English
model automatically. Languages without a configured Kokoro voice use
[Piper](https://github.com/OHF-Voice/piper1-gpl) as a local fallback; its
voice model downloads on first use. Neither path requires an account or has a
usage quota.

For local English runs, place `kokoro-v1.0.onnx` and `voices-v1.0.bin` in
`models/`; the workflow's download step shows their source URLs.

### 3. Pexels (stock footage)
Sign up free at https://www.pexels.com/api/ and copy your API key.

### 4. YouTube Data API v3 (upload)
This is the fiddly one:

1. Go to https://console.cloud.google.com/ and create a project
2. In "APIs & Services" → "Library", enable **YouTube Data API v3**
3. In "APIs & Services" → "OAuth consent screen": set it up as **External**,
   add your own Google account as a **Test user** (this avoids Google's
   verification review, but limits the token to your account only — that's
   fine here since you're uploading to your own channel)
4. In "APIs & Services" → "Credentials", create an **OAuth client ID** of
   type **Desktop app**. Download the JSON.
5. Save that file as `client_secret.json` in this project folder
6. Run locally (not in CI): `pip install -r requirements.txt` then
   `python get_refresh_token.py`
7. A browser opens — log into the Google account that owns your YouTube
   channel and approve access
8. The script prints `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, and
   `YT_REFRESH_TOKEN` — save all three

### 5. Add everything as GitHub Secrets
In your repo: Settings → Secrets and variables → Actions → New repository secret.
Add each of:

- `GROQ_API_KEY`
- `PEXELS_API_KEY`
- `YT_CLIENT_ID`
- `YT_CLIENT_SECRET`
- `YT_REFRESH_TOKEN`

### 6. Push this repo to GitHub
The workflow in `.github/workflows/daily_video.yml` will then run automatically
every four hours. You can also trigger it manually from the Actions tab
("Run workflow") to test before waiting for the schedule.

## Testing locally before relying on the schedule

```bash
pip install -r requirements.txt
# also install ffmpeg locally (brew install ffmpeg / apt install ffmpeg)
export GROQ_API_KEY=...
export PEXELS_API_KEY=...
export YT_CLIENT_ID=...
export YT_CLIENT_SECRET=...
export YT_REFRESH_TOKEN=...
python main.py
```

## Things worth knowing

- **YouTube upload quota**: the default quota is 10,000 units/day, and one
  upload costs 1,600 units — so you can comfortably upload several times a
  day if you ever want to.
- **Content policy risk**: YouTube can flag heavily templated/reused content
  as "repetitious" or low-effort, which affects monetization. Varying the
  script style, voice, and visuals over time reduces this risk.
- **First privacy status**: the workflow uploads as `public` by default.
  Change `YT_PRIVACY_STATUS` to `private` or `unlisted` in the workflow file
  if you'd rather review each video before it goes live.
- **Costs**: everything in this pipeline is free with no monthly quota --
  Groq and Pexels have generous free-tier rate limits (not hard monthly
  caps at this volume), and voiceover runs locally via Piper with no
  metering at all. The only limit that can bite is YouTube's own API
  quota (see above).
- **Groq free-tier limits**: rate limits are generous but not unlimited —
  if you hit a 429 on a run, it's a temporary rate limit, not a billing
  issue; just re-run later.
- **Voice quality**: Kokoro is the default English voice; Piper is the
  fallback for the other listed languages. Switching TTS providers is
  isolated to `generate_voiceover.py`.
