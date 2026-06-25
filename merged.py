import pandas as pd
import os

RAW_DIR = "dataset/raw"
OUTPUT = os.path.join(RAW_DIR, "merged_raw.csv")

sources = {
    os.path.join(RAW_DIR, "x_raw.csv"): {
        "text_col": "text",
        "date_col": "date",
        "label": "x",
    },
    os.path.join(RAW_DIR, "youtube_raw.csv"): {
        "text_col": "Comment",
        "date_col": "PublishedTime",
        "label": "youtube",
    },
    os.path.join(RAW_DIR, "instagram_raw.csv"): {
        "text_col": "Comment",
        "date_col": "Date",
        "label": "instagram",
    },
    os.path.join(RAW_DIR, "kaggle_raw.csv"): {
        "text_col": "full_text",
        "date_col": "created_at",
        "label": "kaggle",
    },
}

dfs = []
for path, cfg in sources.items():
    if not os.path.exists(path):
        print(f"[SKIP] not found: {path}")
        continue
    df = pd.read_csv(path)
    df_out = pd.DataFrame()
    df_out["text"] = df[cfg["text_col"]]
    df_out["date"] = pd.to_datetime(df[cfg["date_col"]], errors="coerce", utc=True)
    df_out["source"] = cfg["label"]
    dfs.append(df_out)
    print(f"[OK] {cfg['label']}: {len(df_out)} rows")

merged = pd.concat(dfs, ignore_index=True)
merged.dropna(subset=["text"], inplace=True)
merged["text"] = merged["text"].astype(str).str.strip()
merged.drop_duplicates(subset=["text"], inplace=True)
merged.sort_values("date", inplace=True, na_position="first")
merged.to_csv(OUTPUT, index=False)

print(f"\n[DONE] {OUTPUT} -> {len(merged)} rows")
print(merged["source"].value_counts().to_string())
