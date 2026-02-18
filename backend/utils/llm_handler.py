import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Initialize OpenAI client ───────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Main function called by app.py ─────────────────────────────────────────────
def process_braille_text(raw_text):
    """
    Takes raw detected braille letters (e.g. "hlowrld") and uses
    GPT-4o-mini to reconstruct it into proper readable English.

    Args:
        raw_text (str): raw string of detected letters from detector.py

    Returns:
        result (dict): {
            'original': raw input,
            'corrected': cleaned up English sentence,
            'success': True/False
        }
    """
    if not raw_text or raw_text.strip() == "":
        return {
            'original': raw_text,
            'corrected': '',
            'success': False,
            'error': 'Empty input'
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Braille text reconstruction assistant. "
                        "You will receive raw letters detected from a Braille image. "
                        "The detection may have errors, missing letters, or Braille Grade 2 contractions. "
                        "Your job is to reconstruct the most likely intended English sentence. "
                        "Return ONLY the reconstructed sentence, nothing else. "
                        "Do not explain, do not add punctuation unless obvious, just return the clean sentence."
                    )
                },
                {
                    "role": "user",
                    "content": f"Raw braille detected text: {raw_text}"
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        corrected = response.choices[0].message.content.strip()

        return {
            'original': raw_text,
            'corrected': corrected,
            'success': True
        }

    except Exception as e:
        # If LLM fails, return the raw text so the app still works
        return {
            'original': raw_text,
            'corrected': raw_text,
            'success': False,
            'error': str(e)
        }