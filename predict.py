"""
Interactive CLI for MBG sentiment prediction using the trained SVM model.

Usage:
    python predict.py              # interactive mode
    python predict.py --demo       # run built-in examples
"""
import sys
import joblib

MODEL_PATH = "ml/models/svm_sentiment.pkl"
TFIDF_PATH = "ml/models/tfidf_vectorizer.pkl"

# ── Load model once ────────────────────────────────────────────────────────
print("[LOAD] Loading model & vectorizer...", end=" ")
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
print("OK\n")


def predict(text: str) -> str:
    """Return sentiment label for a single Indonesian text."""
    vec = tfidf.transform([text])
    return model.predict(vec)[0]


# ── Demo examples ──────────────────────────────────────────────────────────
examples = [
    "Program makan bergizi gratis ini sangat membantu anak-anak kurang mampu, semoga terus berlanjut!",
    "Makan bergizi gratis katanya, tapi kenyataannya menunya tidak layak dan porsinya sangat kurang",
    "Pemerintah resmi meluncurkan program makan bergizi gratis untuk siswa SD di seluruh Indonesia",
    "Daftar jadi mitra MBG gampang banget, langsung cuan deh! Makasih pak Prabowo",
    "Sudah seminggu program MBG berjalan tapi sekolah kami belum dapat sama sekali, kemana anggarannya?",
]

if "--demo" in sys.argv:
    print("=" * 60)
    print("Demo — Sentiment Prediction")
    print("=" * 60)
    for i, text in enumerate(examples, 1):
        label = predict(text)
        emoji = {"positif": "+", "negatif": "-", "netral": "~"}.get(label, "?")
        print(f"\n[{i}] {text}")
        print(f"    Sentiment: {emoji} {label}")
    print()
    sys.exit(0)

# ── Interactive mode ───────────────────────────────────────────────────────
print("=" * 60)
print("MBG Sentiment Predictor (SVM)")
print("Type 'exit' to quit")
print("=" * 60)

while True:
    try:
        text = input("\nText: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if text.lower() in ("exit", "quit", "keluar"):
        print("Bye!")
        break

    if not text:
        continue

    label = predict(text)
    emoji = {"positif": "+", "negatif": "-", "netral": "~"}.get(label, "?")
    print(f"  Sentiment: {emoji} {label}")
