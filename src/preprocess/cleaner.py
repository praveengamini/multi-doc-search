import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text
