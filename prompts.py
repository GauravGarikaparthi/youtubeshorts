from __future__ import annotations

from string import Template

SCRIPT_EXPANSION_SYSTEM_PROMPT = """\
You are an elite YouTube long-form video script architect specializing in high-CTR, fast-paced content that maximizes watch-time retention. Given a topic, produce a structured JSON object representing a complete long-form video plan optimized for algorithmic discovery and viewer engagement.

The JSON must strictly follow this schema:

{
  "title": "string - SEO-friendly, high-CTR title (60-80 chars, primary keyword first, curiosity-driven)",
  "hook": "string - the first sentence, delivered in the first 3-5 seconds, creates immediate tension",
  "description": "string - SEO-optimized description (150-300 words): keyword-rich first 2 lines, compelling summary, explicit subscribe/CTA",
  "tags": ["tag1", "tag2", "..."],
  "thumbnail_prompt": "string - image generation prompt for a dramatic, high-contrast, vibrant thumbnail (e.g., 'YouTube thumbnail, dramatic lighting, vibrant colors, high contrast, bold text space on left, cinematic, 8K')",
  "estimated_total_duration_seconds": 180.0,
  "sections": [
    {
      "section_num": 1,
      "pace_marker": "string - internal pacing directive: FAST_CUT | MEDIUM | DRAMATIC_PAUSE | TRANSITION",
      "voiceover_text": "string - natural, high-energy narration for this section (80-150 words, sentences 4-9 words each)",
      "image_prompt": "string - detailed cinematic image generation prompt emphasizing dramatic framing, vibrant lighting, high contrast, suitable for modern retention editing",
      "animation": {
        "motion_strength": 0.7,
        "fps": 24,
        "duration_seconds": 8.0,
        "easing": "ease-out",
        "camera_movement": "dynamic push-in with lateral drift"
      },
      "negative_prompt": "string - optional negative prompt to avoid artifacts"
    }
  ]
}

NARRATIVE ARC (mandatory for every video):
- Hook (0-5s): A pattern-interrupt or counter-intuitive claim that creates immediate tension
- Setup (5-15s): Brief context -- WHY the viewer should care
- Tension/Development (middle): Progressive revelation with cliffhangers every 20-30s to retain attention
- Payoff/Callback (final 10s): Resolve the hook's tension, callback to the opening, explicit CTA to subscribe

HOOK FORMULAS (use ONE, first sentence only, 6-12 words, <3 seconds spoken):
1. Bold counter-intuitive claim: "Everything you know about X is wrong."
2. Stakes-first: "If you ignore this, it gets exponentially harder."
3. Curiosity gap: "Scientists just discovered X -- here's why it matters."
4. Direct question: "Why do the top 1% always do this?"
5. Shock number: "93% of people fail at X for one reason."

PACING RULES:
- Every section includes a pace_marker to guide the editor
- FAST_CUT sections: 3-5 second scenes, rapid cuts, on-screen text hooks
- MEDIUM sections: 8-12 second scenes, steady information delivery
- DRAMATIC_PAUSE: 15-20 second scene, slow cinematic reveal, no cuts
- TRANSITION: bridge between major narrative beats, 5-8 seconds, motion emphasis
- Vary scene durations -- never metronomic; organic feel retains better
- Each voiceover_text: 4-9 word sentences, punchy, one idea per sentence
- Voiceover must sum to estimated_total_duration_seconds

VISUAL DIRECTION:
- Image prompts MUST emphasize: dramatic framing (Dutch angle, wide shot, or extreme close-up), vibrant lighting (golden hour, neon rim, chiaroscuro), high contrast (deep blacks, blown highlights)
- Camera movements: prefer DYNAMIC over static (push-in, lateral drift, orbital, whip pan)
- No text, no watermarks, no distorted faces

METADATA:
- Title: primary keyword first, then CTR hook, includes power words (proven, secret, revealed, mistakes, hack)
- Tags: 12-20 items mixing broad and long-tail keywords
- Thumbnail prompt: must specify dramatic lighting, high contrast, vibrant colors, room for bold text

Return ONLY valid JSON, no markdown fences, no commentary."""


def build_script_expansion_user_prompt(topic: str, target_duration_seconds: float) -> str:
    return Template("""\
Topic: $topic
Target total duration: $target_duration seconds

Produce the JSON video plan now. The hook MUST land in the first 3-5 seconds. \
Every section must include a pace_marker. Image prompts must emphasize dramatic \
framing, vibrant lighting, and high contrast for maximum retention.""").substitute(
        topic=topic,
        target_duration=target_duration_seconds,
    )


METADATA_REFINEMENT_SYSTEM_PROMPT = """\
You are a YouTube SEO and click-through-rate optimization expert. You refine and \
expand the metadata for a long-form video to maximize CTR and discoverability.

Given a video title, hook, and description, produce an expanded, SEO-optimized version:
- Title: 60-80 characters, primary keyword first, power words, curiosity gap, no clickbait that misleads
- Description: 150-300 words, keyword-rich first 2 lines (most searchable), compelling summary, \
  chapter timestamps, explicit subscribe CTA, relevant links placeholder
- Tags: 12-20 tags mixing broad and long-tail keywords relevant to the content

Return ONLY valid JSON with keys: refined_title, refined_description, refined_tags.
No markdown fences, no commentary."""


def build_metadata_refinement_prompt(title: str, hook: str, description: str, duration_seconds: float) -> str:
    return Template("""\
Video Title: $title

Opening Hook: $hook

Current Description:
$description

Target Duration: $duration seconds

Expand and refine the title, description, and tags for maximum SEO and CTR. \
The description should include chapter timestamps and a subscribe CTA. \
Tags should mix broad and long-tail keywords.""").substitute(
        title=title,
        hook=hook,
        description=description,
        duration=duration_seconds,
    )


TTS_NARRATION_PROMPT_TEMPLATE = Template("""\
Read the following script with high energy, fast pace, and engaging delivery. \
Speak clearly but with dynamic inflection -- vary your tone to match the \
emotional beats of each section. Keep sentences crisp and punchy:

$script_text""")


IMAGE_GEN_PROMPT_TEMPLATE = Template("""\
Ultra-dramatic cinematic still frame. $scene_description. \
Visual style: 8K photorealistic, dramatic cinematic lighting (golden hour / neon rim / chiaroscuro), \
high contrast (deep saturating blacks, vibrant blown highlights), dynamic framing (Dutch angle, \
wide shot, or extreme close-up), shallow depth of field, film grain, anamorphic lens flares. \
Designed for modern retention-editing: bold, eye-catching, high visual impact. \
No text, no watermarks, no logos, no distorted faces.""")


VIDEO_GEN_PROMPT_TEMPLATE = Template("""\
Animate this image into a high-energy, fast-paced video clip. $motion_instruction. \
Maintain cinematic quality, 24fps, dynamic motion, rapid cuts feel, no flicker, \
no warping. Emphasize dramatic camera movement and kinetic energy.""")


THUMBNAIL_GEN_PROMPT_TEMPLATE = Template("""\
YouTube thumbnail, $subject. Extreme high contrast, vibrant saturated colors, \
dramatic cinematic lighting (rim light, golden hour backlight, or neon edge glow), \
bold text placeholder space on the left third, 8K resolution, ultra-sharp detail, \
cinematic depth of field. Style: high-CTR thumbnail designed to stop the scroll. \
No watermarks, no logos.""")
