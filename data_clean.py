import html
import re
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────
INPUT_FILE = "dataset/raw/merged_raw.csv"
OUTPUT_FILE = "dataset/cleaned/cleaned.csv"
MIN_WORD_COUNT = 2       # Drop texts shorter than this many words
MIN_CHAR_LENGTH = 5      # Drop texts shorter than this many characters

# ── Emoji pattern ──────────────────────────────────────────────────────────
# Matches emojis, emoticons, and other non-text symbols commonly found in
# social media posts (flags, hearts, faces, gestures, etc.)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002300-\U000023FF"  # miscellaneous technical
    "\U000025A0-\U000027BF"  # dingbats / geometric shapes
    "\U00002934-\U00002935"  # arrows
    "\U00002B05-\U00002B07"  # arrows
    "\U00002B1B-\U00002B1C"  # squares
    "\U00002B50"             # star
    "\U0000200D"             # zero-width joiner
    "\U0000FE0F"             # variation selector-16
    "]+",
    flags=re.UNICODE,
)

# ── Load raw merged data ───────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
total_before = len(df)
print(f"[LOAD] {total_before} rows from {INPUT_FILE}")

# ── Text cleaning function ─────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Apply a comprehensive cleaning pipeline for Indonesian social media
    text destined for SVM sentiment analysis.

    Steps:
    1. Decode HTML entities (&amp; → &)
    2. Remove URLs
    3. Remove mentions (@username)
    4. Strip hashtag symbol # but keep keyword
    5. Normalise smart/curly quotes to straight quotes
    6. Remove emojis and other Unicode symbol garbage
    7. Remove standalone punctuation artifacts (e.g. leftover dots, hyphens)
    8. Collapse whitespace
    9. Strip wrapping quotes that survived CSV parsing
    10. Trim and validate
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities: &amp; → &, &lt; → <, etc.
    text = html.unescape(text)

    # Remove URLs (http, https, t.co, etc.)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove Twitter/X mentions (@username)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol '#' but keep the word after it
    text = re.sub(r"#", "", text)

    # Normalise smart / curly quotes and other MS-word–style punctuation
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # left/right double
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # left/right single
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    text = text.replace("\u00a0", " ")                          # non-breaking space
    text = text.replace("\u2026", "...")                        # ellipsis

    # Remove emojis and other decorative Unicode symbols
    text = EMOJI_PATTERN.sub("", text)

    # Strip leading/trailing double-quotes left over from CSV quoting
    text = text.strip('"').strip("'").strip()

    # Remove leftover standalone punctuation artefacts like "Mantap.. Ni.."
    # (repeated full-stops, isolated commas/dashes between words)
    text = re.sub(r"\s*\.{2,}\s*", " ", text)    # ".." or "..."
    text = re.sub(r"\s+-\s+", " ", text)          # " - " dash between words

    # Remove newline / carriage-return characters
    text = text.replace("\n", " ").replace("\r", " ")

    # Collapse multiple spaces into a single space
    text = re.sub(r"\s+", " ", text)

    # Final trim
    text = text.strip().strip('"').strip("'").strip()

    return text

# ── Apply cleaning ─────────────────────────────────────────────────────────
df["text"] = df["text"].astype(str).apply(clean_text)

# ── Drop rows with empty or whitespace-only text ───────────────────────────
df = df[df["text"].str.strip() != ""]

# ── Drop texts that are too short to be meaningful ─────────────────────────
word_counts = df["text"].str.split().str.len()
char_lengths = df["text"].str.len()
mask_short = (word_counts >= MIN_WORD_COUNT) & (char_lengths >= MIN_CHAR_LENGTH)
df = df[mask_short]

# ── Drop duplicate texts (keep first occurrence) ───────────────────────────
df = df.drop_duplicates(subset=["text"])

# ── Keep only the columns needed for sentiment analysis ────────────────────
df = df[["text", "date", "source"]]

# ── Save cleaned result ────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

# ── Summary ────────────────────────────────────────────────────────────────
total_after = len(df)
removed = total_before - total_after
print(f"[DONE] {OUTPUT_FILE} → {total_after} rows")
print(f"       Removed: {removed} rows (empty / duplicates / too short)")
print(f"       Sources: {dict(df['source'].value_counts())}")
