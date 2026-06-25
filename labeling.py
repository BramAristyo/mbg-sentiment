"""
Automatic sentiment labelling for MBG dataset using DeepSeek API.

Reads cleaned texts, sends them in batches to DeepSeek, and stores
"positif" / "negatif" / "netral" labels. Supports resume on failure.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────
load_dotenv()

INPUT_FILE = "dataset/cleaned/cleaned.csv"
OUTPUT_FILE = "dataset/labeled/labeled.csv"
PROGRESS_FILE = "dataset/labeled/.labeling_progress.json"

TEST_MODE = False  # ← Set False to label ALL 7k+ rows
TEST_LIMIT = 35  # ← Number of rows to label in TEST_MODE

BATCH_SIZE = 35  # texts per API call
DELAY_BETWEEN_BATCHES = 1.0  # seconds to wait between batches
MAX_RETRIES = 3  # retries on API failure
BACKOFF_BASE = 5  # initial backoff seconds (exponential)

# ── Prompt templates (Bahasa Indonesia) ────────────────────────────────────
SYSTEM_PROMPT = (
    "Kamu adalah annotator sentimen untuk teks media sosial berbahasa Indonesia. "
    "Tugasmu mengklasifikasikan setiap teks terkait program "
    '"Makan Bergizi Gratis (MBG)" menjadi salah satu dari tiga label:\n\n'
    '- "positif" — teks mendukung, memuji, optimis, atau menyetujui MBG.\n'
    '- "negatif" — teks mengkritik, menolak, mengecam, atau kecewa terhadap MBG.\n'
    '- "netral"  — teks informatif, faktual, tidak berpihak, atau tidak mengandung sentimen jelas.\n\n'
    "Perhatikan:\n"
    "- Jika teks tidak terkait MBG sama sekali, tetap beri label 'netral'.\n"
    "- Jika ada sarkasme, pilih sentimen yang SESUNGGUHNYA dimaksud (bukan literal).\n"
    "- Jangan tambahkan teks apapun selain JSON."
)

USER_PROMPT_TEMPLATE = (
    "Klasifikasikan {count} teks di bawah ini.\n\n"
    "{texts}\n\n"
    "Balas HANYA dengan JSON array persis seperti format ini, "
    "tanpa teks pembuka atau penutup:\n"
    '[{{"index": 0, "label": "positif"}}, {{"index": 1, "label": "negatif"}}, ...]'
)

# ── Client ─────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-dummy-change-me"),
    base_url="https://api.deepseek.com",
)


# ── Helpers ────────────────────────────────────────────────────────────────
def build_user_prompt(texts: list[str]) -> str:
    """Format a list of texts into the batch prompt."""
    lines = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    return USER_PROMPT_TEMPLATE.format(count=len(texts), texts=lines)


def parse_response(raw: str, expected_count: int) -> list[str]:
    """Extract labels from the model's JSON response. Returns list of labels
    in index order. Fills 'netral' for missing/unparseable entries."""
    # Try to find a JSON array in the response (may be wrapped in ```json ... ```)
    raw = raw.strip()
    if raw.startswith("```"):
        # Strip code fences
        parts = raw.split("\n")
        parts = [p for p in parts if not p.startswith("```")]
        raw = "\n".join(parts).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON array with a regex
        import re

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                print(f"  [WARN] Could not parse JSON from: {raw[:200]}")
                return ["netral"] * expected_count
        else:
            print(f"  [WARN] No JSON array found in: {raw[:200]}")
            return ["netral"] * expected_count

    labels = ["netral"] * expected_count
    for item in parsed:
        idx = item.get("index")
        lbl = item.get("label", "netral").strip().lower()
        if lbl not in ("positif", "negatif", "netral"):
            lbl = "netral"
        if isinstance(idx, int) and 0 <= idx < expected_count:
            labels[idx] = lbl

    return labels


def call_api(texts: list[str]) -> list[str]:
    """Send a batch to DeepSeek with retry logic. Returns label list."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(texts)},
                ],
                temperature=0.0,
                max_tokens=2048,
                stream=False,
            )
            raw = response.choices[0].message.content
            return parse_response(raw, len(texts))

        except Exception as e:
            wait = BACKOFF_BASE**attempt
            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {e} — waiting {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries")


def load_progress() -> set[int]:
    """Load already-processed row indices from the progress file."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        data = json.load(f)
    return set(data.get("done", []))


def save_progress(done_indices: set[int]):
    """Persist the set of processed row indices."""
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": sorted(done_indices)}, f)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MBG Sentiment Labelling via DeepSeek API")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    if TEST_MODE:
        df = df.head(TEST_LIMIT)
        print(f"[TEST MODE] Only first {len(df)} rows will be labelled")
        print("            Set TEST_MODE=False in labeling.py to label all rows")

    total_rows = len(df)
    print(f"[LOAD] {total_rows} rows from {INPUT_FILE}")

    done = load_progress()
    remaining = total_rows - len(done)
    print(f"[PROGRESS] {len(done)} already labelled, {remaining} remaining")

    if remaining == 0:
        print("[DONE] All rows already labelled. Nothing to do.")
        return

    total_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[BATCH] {BATCH_SIZE} texts/batch → ~{total_batches} batches")
    print(f"[COST] ~$0.30–0.80 estimated for {remaining} rows")
    print()

    # Initialise or load the label column
    if "label" not in df.columns:
        df["label"] = None

    # Restore previously saved labels from progress
    for idx in done:
        if idx < len(df):
            # We don't store labels in the progress file — re-read from output
            pass

    # We'll work with an in-memory dict for speed: {row_index: label}
    labels_dict: dict[int, str] = {}

    # If there is an existing output file, load its labels
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        try:
            existing = pd.read_csv(OUTPUT_FILE)
        except Exception:
            existing = None
        if (
            existing is not None
            and "label" in existing.columns
            and len(existing) == len(df)
        ):
            for i, lbl in enumerate(existing["label"]):
                if pd.notna(lbl):
                    labels_dict[i] = lbl

    batch_count = 0
    for start in range(0, total_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_rows)

        # Figure out which rows in this batch still need labelling
        batch_indices = list(range(start, end))
        pending_indices = [i for i in batch_indices if i not in done]

        if not pending_indices:
            continue  # whole batch already done

        # Prepare texts (use the cleaned text from DataFrame)
        batch_texts = [df.at[i, "text"] for i in pending_indices]

        batch_count += 1
        print(
            f"[{batch_count}/{total_batches}] "
            f"rows {pending_indices[0]}–{pending_indices[-1]} "
            f"({len(pending_indices)} texts)...",
            end=" ",
            flush=True,
        )

        try:
            labels = call_api(batch_texts)
        except RuntimeError as e:
            print(f"\n[FATAL] {e}")
            print("Progress saved. Re-run to resume.")
            return

        for idx, label in zip(pending_indices, labels):
            labels_dict[idx] = label
            done.add(idx)

        # Distribution of this batch
        from collections import Counter

        dist = Counter(labels)
        print(f"✓  {dict(dist)}")

        # Save progress (JSON) + partial output (CSV) after every batch
        save_progress(done)

        # Write the full labelled DataFrame to CSV so far
        df["label"] = df.index.map(lambda i: labels_dict.get(i, "netral"))
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        if batch_count < total_batches:
            time.sleep(DELAY_BETWEEN_BATCHES)

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("[DONE] Labelling complete!")

    df["label"] = df.index.map(lambda i: labels_dict.get(i, "netral"))
    dist = df["label"].value_counts()
    print(f"       Total: {len(df)} rows")
    for lbl in ["positif", "negatif", "netral"]:
        print(f"       {lbl:10s}: {dist.get(lbl, 0)}")
    print(f"       Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()

    # ============================================================
    # [DONE] Labelling complete!
    #        Total: 7172 rows
    #        positif   : 2334
    #        negatif   : 2999
    #        netral    : 1839
    #        Output: dataset/labeled/labeled.csv
    # ============================================================
