# MBG Sentiment Analysis

Klasifikasi sentimen program **Makan Bergizi Gratis (MBG)** dari data media sosial berbahasa Indonesia menggunakan SVM.

> Dokumentasi lengkap pipeline dari scraping sampai evaluasi: **[ml/training/readme.MD](ml/training/readme.MD)**

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Prediksi teks
python predict.py --demo          
python predict.py      
```

## Struktur Project

```
mbg-sentiment/
├── predict.py                    # CLI prediksi sentimen
├── dataset/
│   ├── raw/                      # Data mentah (X, YouTube, Instagram, Kaggle)
│   ├── cleaned/                  # cleaned.csv
│   ├── labeled/                  # labeled.csv, labeled_v2.csv, manual_labeled.csv
│   └── scraping_scripts/         # Scraper X/Twitter (twikit)
├── ml/
│   ├── training/                 # Script + notebook training
│   │   ├── data_clean.py         # Pembersihan teks
│   │   ├── llm_labelling.py      # Labelling via DeepSeek API
│   │   ├── main.ipynb            # Notebook modelling SVM
│   │   └── readme.MD             # Dokumentasi super detail
│   └── models/                   # Model hasil training
│       ├── svm_sentiment.pkl
│       └── tfidf_vectorizer.pkl
├── .env                          # API keys (tidak di-commit)
├── .env.example                  # Template .env
└── requirements.txt
```

## Teknologi

Python 3.12, scikit-learn, TF-IDF, SVM, DeepSeek API, pandas, matplotlib, seaborn, wordcloud, twikit

## Link

- **GitHub:** https://github.com/BramAristyo/mbg-sentiment
