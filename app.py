import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import spacy
import openai

# ✅ Set the API key first
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set or loaded")

# Optional: Confirm it's set (for debugging)
print("API key loaded:", openai.api_key[:8] + "..." if openai.api_key else "Not found")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Get user input
text = input("Enter a sentence to analyze: ")
doc = nlp(text)

# Header
print(
    "\n{:<12} → {:<10} → {:<10} → {:<30} → {:<50}".format(
        "Word", "POS", "Lemma", "Meaning", "Example"
    )
)
print("-" * 120)

# Analyze
for token in doc:
    if token.is_alpha:
        lemma = token.lemma_
        pos = token.pos_
        word = token.text

        try:
            prompt = f"Give a short definition and a sentence using the word '{lemma}' as a {pos.lower()}."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=80,
            )
            reply = response.choices[0].message.content.strip()

            # Try parsing
            if "Definition:" in reply and "Example:" in reply:
                meaning = reply.split("Definition:")[1].split("Example:")[0].strip()
                example = reply.split("Example:")[1].strip()
            else:
                parts = reply.split(". ")
                meaning = parts[0].strip()
                example = ". ".join(parts[1:]).strip()

        except Exception as e:
            meaning = "Error"
            example = str(e)

        print(
            "{:<12} → {:<10} → {:<10} → {:<30} → {:<50}".format(
                word, pos, lemma, meaning[:30], example[:50]
            )
        )
