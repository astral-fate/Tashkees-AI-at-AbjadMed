import re

def clean_text(text):
    """
    Cleans Arabic medical text by removing:
    1. Structural artifacts (headers, dashes)
    2. Common social/religious greetings
    3. Formatting characters (underscores, newlines)
    4. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove Structural Artifacts (Headers & Dashes)
    # Removes: "السؤال", "الجواب", and repeating dashes "-------"
    text = re.sub(r'السؤال\s*\n\s*-+', ' ', text)
    text = re.sub(r'الجواب\s*\n\s*-+', ' ', text)

    # 2. Remove Common Social/Religious Greetings (Noise)
    greetings = [
        r'السلام عليكم ورحمة الله وبركاته',
        r'السلام عليكم',
        r'بسم الله الرحمن الرحيم',
        r'صباح الخير',
        r'مساء الخير',
        r'تحية طيبة',
        r'يا دكتور',
        r'دكتور',
        r'شكرا لك',
        r'سلامتك'
    ]
    text = re.sub(r'|'.join(greetings), ' ', text)

    # 3. Remove Formatting
    text = re.sub(r'_+', ' ', text) # Remove underscores like "____"
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    # text = re.sub(r'[^\w\s]', '', text) # Optional: Remove punctuation (commented out as per original)

    # 4. Collapse Whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
