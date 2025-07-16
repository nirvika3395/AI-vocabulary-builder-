import os
from dotenv import load_dotenv
import spacy
from huggingface_hub import InferenceClient

# Load .env and get Hugging Face API key
load_dotenv()
hf_key = os.getenv("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY is not set in .env")

# Set up the Hugging Face client
client = InferenceClient(token=hf_key)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Get input text
text = input("Enter a sentence to analyze: ")
doc = nlp(text)

# Header
print("\n{:<12} → {:<10} → {:<10} → {:<30} → {:<50}".format("Word", "POS", "Lemma", "Meaning", "Example"))
print("-" * 120)

# Analyze each token
for token in doc:
    if token.is_alpha:
        word = token.text
        lemma = token.lemma_
        pos = token.pos_

        try:
            # Ask Hugging Face model
            prompt = f"Define the word '{lemma}' as a {pos.lower()} and use it in a sentence."
            response = client.text_generation(
                prompt,
                model="google/flan-t5-base",  # ✅ RECOMMENDED MODEL
                max_new_tokens=100,
                temperature=0.5,
            )
            reply = response.strip()


            # Parse response
            if "Definition:" in reply and "Example:" in reply:
                meaning = reply.split("Definition:")[1].split("Example:")[0].strip()
                example = reply.split("Example:")[1].strip()
            else:
                parts = reply.split(". ")
                meaning = parts[0].strip() if parts else "N/A"
                example = ". ".join(parts[1:]).strip() if len(parts) > 1 else "N/A"

        except Exception as e:
            meaning = "Error"
            example = str(e)

        # Print result
        print("{:<12} → {:<10} → {:<10} → {:<30} → {:<50}".format(
            word, pos, lemma, meaning[:30], example[:50]
        ))
