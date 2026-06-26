# MBG Sentiment Analysis — Dokumentasi Project ML

> Klasifikasi sentimen program **Makan Bergizi Gratis (MBG)** dari data media sosial berbahasa Indonesia menggunakan Support Vector Machine (SVM).

---

## Daftar Isi

1. [Ikhtisar Project](#ikhtisar-project)
2. [Tech Stack](#tech-stack)
3. [Struktur Folder Lengkap](#struktur-folder-lengkap)
4. [Pipeline End-to-End](#pipeline-end-to-end)
5. [Setup & Instalasi](#setup--instalasi)
6. [Step-by-Step: Cara Menjalankan](#step-by-step-cara-menjalankan)
7. [Detail Tiap Script](#detail-tiap-script)
8. [Hasil Training & Evaluasi](#hasil-training--evaluasi)
9. [Prediksi (predict.py)](#prediksi-predictpy)
10. [Distribusi Label](#distribusi-label)
11. [Catatan Penting / Tips](#catatan-penting--tips)

---

## Ikhtisar Project

Project ini membangun pipeline lengkap **sentiment analysis** untuk program **Makan Bergizi Gratis (MBG)** — program pemerintah Indonesia yang menyediakan makanan bergizi gratis untuk siswa sekolah.

**Tujuan:** Mengklasifikasikan setiap teks/komen media sosial ke dalam 3 kelas sentimen:

| Label    | Arti                                                    |
|----------|---------------------------------------------------------|
| `positif` | Mendukung, memuji, optimis, atau menyetujui MBG         |
| `negatif` | Mengkritik, menolak, mengecam, atau kecewa terhadap MBG |
| `netral`  | Informatif, faktual, tidak berpihak, atau tidak jelas   |

**Sumber data:** X (Twitter), YouTube, Instagram, Kaggle

---

## Tech Stack

| Komponen         | Teknologi                                           |
|------------------|------------------------------------------------------|
| Bahasa           | Python 3.12                                          |
| Scraping         | `twikit` (X), export manual (YouTube/Instagram/Kaggle) |
| Data Processing  | `pandas`, `re`, `html`                               |
| Labelling        | **DeepSeek API** (via `openai` library, batch 35 teks) |
| Feature Extraction | **TF-IDF** (1-2 gram, 10.000 fitur, `sublinear_tf=True`) |
| Model            | **SVM** kernel Linear (`sklearn.svm.SVC`)             |
| Tuning           | `GridSearchCV` (5-fold, `f1_macro`)                   |
| Visualisasi      | `matplotlib`, `seaborn`, `wordcloud`                  |
| Deployment       | `joblib` (serialisasi model), `predict.py` (CLI interaktif) |

---

## Struktur Folder Lengkap

```
mbg-sentiment/
│
├── .env                          # API keys & config (tidak di-commit)
├── .env.example                  # Template .env
├── .gitignore
├── requirements.txt              # Dependencies lengkap, install via pip
├── predict.py                    # CLI interaktif untuk prediksi teks baru
├── README.MD                     # README root project
│
├── dataset/
│   ├── raw/                      # Data mentah dari berbagai sumber
│   │   ├── merged.py             # Script merge 4 sumber → merged_raw.csv
│   │   ├── merged_raw.csv        # Hasil merge (7.847 baris)
│   │   ├── x_raw.csv             # Scraping X/Twitter
│   │   ├── youtube_raw.csv       # Scraping YouTube (manual)
│   │   ├── instagram_raw.csv     # Scraping Instagram (manual)
│   │   ├── kaggle_raw.csv        # Dataset Kaggle
│   │   ├── instagram/            # File export mentah Instagram
│   │   │   ├── export_20260622-120133.csv
│   │   │   └── export_20260622-120433.csv
│   │   └── youtube/              # File export mentah YouTube (per video)
│   │       ├── apa-yang-sebenarnya-terjadi-di-balik-dapur-mbg-helmy-yahya-bicara.csv
│   │       ├── dpr-tolak-mbg-saat-libur-sekolah-muncul-kekhawatiran-program-dipaksakan-serap-an.csv
│   │       ├── full-evaluasi-mbg-moratorium-dapur-hingga-insentif-disetop-makan-bergizi-gratis-.csv
│   │       ├── problematika-dan-masalah-makan-bergizi-gratis.csv
│   │       └── soal-stunting-mbg-watch-perbedaan-sppg-papua-pegunungan-dengan-wilayah-lain.csv
│   │
│   ├── cleaned/
│   │   └── cleaned.csv           # Hasil cleaning (7.172 baris)
│   │
│   ├── labeled/
│   │   ├── labeled.csv           # Hasil labelling DeepSeek (full)
│   │   ├── manual_labeled.csv    # 300 sampel untuk review manual
│   │   ├── labeled_v2.csv        # Merge manual prio + DeepSeek (final training data)
│   │   └── .labeling_progress.json  # State resume labelling (auto-generated)
│   │
│   └── scraping_scripts/         # Script pengumpulan data
│       ├── x_scraper.py          # Scraper X/Twitter (via twikit)
│       ├── cookie.json.example   # Template cookie auth untuk X
│       └── record_counter.py     # Utility counter record
│
├── ml/
│   ├── training/                 # Semua script & notebook training
│   │   ├── data_clean.py         # Script pembersihan teks
│   │   ├── llm_labelling.py      # Script labelling via DeepSeek API
│   │   ├── main.ipynb            # Notebook modelling SVM (TF-IDF → GridSearch → Evaluasi → Simpan)
│   │   └── readme.MD             # ← File ini
│   │
│   └── models/                   # Model hasil training (siap production)
│       ├── svm_sentiment.pkl     # SVM terlatih
│       └── tfidf_vectorizer.pkl  # TF-IDF vectorizer terlatih
│
└── .venv/                        # Virtual environment Python 3.12
```

---

## Pipeline End-to-End

Berikut alur project dari awal sampai siap produksi:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. SCRAPING (Pengumpulan Data)                                      │
│    • X/Twitter   → x_scraper.py (twikit)                            │
│    • YouTube     → export manual dari website open-source           │
│    • Instagram   → export manual dari website open-source           │
│    • Kaggle      → download manual, sudah berlabel                  │
├─────────────────────────────────────────────────────────────────────┤
│ 2. MERGE (Penggabungan)                    dataset/raw/merged.py    │
│    • Gabungkan 4 sumber → merged_raw.csv     7.847 baris            │
│    • Format seragam: [text, date, source]                           │
├─────────────────────────────────────────────────────────────────────┤
│ 3. CLEANING (Pembersihan)                ml/training/data_clean.py  │
│    • Decode HTML entities ( &amp; → & )                             │
│    • Hapus URL, @mention, #hashtag (simpan kata setelah #)          │
│    • Hapus emoji & simbol Unicode dekoratif                         │
│    • Normalisasi smart quotes, en/em dash, non-breaking space       │
│    • Hapus baris dengan teks kosong / terlalu pendek (<2 kata, <5 char)│
│    • Hapus duplikat teks                                             │
│    → cleaned.csv                          7.172 baris               │
├─────────────────────────────────────────────────────────────────────┤
│ 4. LABELLING (Pelabelan Otomatis)     ml/training/llm_labelling.py  │
│    • Batch 35 teks per API call ke DeepSeek API                     │
│    • Prompt Bahasa Indonesia dengan panduan sentimen jelas          │
│    • Resume support → bisa lanjut jika gagal di tengah jalan        │
│    • Retry otomatis maks 3x dengan exponential backoff              │
│    → labeled.csv                          7.172 baris berlabel      │
├─────────────────────────────────────────────────────────────────────┤
│ 5. SAMPLE MANUAL (Review Manusia)      sample_manual.py (notebook)  │
│    • Ambil 300 baris random dari labeled.csv                        │
│    → manual_labeled.csv                  300 baris (perlu dicek)    │
├─────────────────────────────────────────────────────────────────────┤
│ 6. MERGE LABEL (Prioritas Manual)       sample_manual.py (notebook) │
│    • Gabungkan manual (300) + otomatis (6.872)                      │
│    • Jika duplikat → manual yang dipakai (prioritas)                │
│    → labeled_v2.csv                       7.172 baris               │
├─────────────────────────────────────────────────────────────────────┤
│ 7. MODELLING (Training SVM)            ml/training/main.ipynb       │
│    • Baca labeled_v2.csv                                            │
│    • TF-IDF vectorization (1-2 gram, 10k fitur)                     │
│    • Train/test split 80:20 stratified                              │
│    • GridSearchCV tuning C = [0.1, 1, 10, 100]                      │
│    • SVM kernel Linear, class_weight='balanced'                     │
│    • Evaluasi: classification report + confusion matrix             │
│    • Simpan model → ml/models/svm_sentiment.pkl                     │
│    • Simpan vectorizer → ml/models/tfidf_vectorizer.pkl             │
├─────────────────────────────────────────────────────────────────────┤
│ 8. PREDIKSI (Production)                     predict.py             │
│    • CLI interaktif: ketik teks → dapat sentimen                    │
│    • Demo mode: python predict.py --demo                            │
│    • Load model + vectorizer sekali, prediksi real-time             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup & Instalasi

### 1. Clone Project & Buat Virtual Environment

```bash
git clone <repo-url>
cd mbg-sentiment

# Buat venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi API Key (untuk labelling)

```bash
# Salin template .env
cp .env.example .env

# Edit .env — isi dengan API key DeepSeek
# DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Register Jupyter Kernel (untuk notebook)

```bash
python -m ipykernel install --user --name=mbg-sentiment --display-name="MBG Sentiment"
```

### 4. Hanya untuk training ulang dari nol — jalankan step di bawah berurutan

---

## Step-by-Step: Cara Menjalankan

Semua command dijalankan dari **root project** (`mbg-sentiment/`), dengan virtual environment sudah aktif (`source .venv/bin/activate`).

### Step 1 — Merge Data Mentah

```bash
python dataset/raw/merged.py
```
> Menggabungkan 4 file CSV mentah (X, YouTube, Instagram, Kaggle) jadi satu `merged_raw.csv`.
> 
> **Output:** `dataset/raw/merged_raw.csv` (7.847 baris)

### Step 2 — Pembersihan Teks

```bash
python ml/training/data_clean.py
```
> Script ini HARUS dijalankan dari root project karena path relatifnya dihitung dari `ml/training/`.
> 
> **Input:** `dataset/raw/merged_raw.csv`
> **Output:** `dataset/cleaned/cleaned.csv` (7.172 baris)

### Step 3 — Labelling via DeepSeek API

```bash
# Test mode dulu (hanya 35 baris, untuk cek prompt)
# Buka llm_labelling.py, set TEST_MODE = True, lalu:
python ml/training/llm_labelling.py

# Full mode (labeling seluruh 7.172 baris)
# Buka llm_labelling.py, set TEST_MODE = False, lalu:
python ml/training/llm_labelling.py
```

> **Biaya estimasi:** ~$0.30–0.80 untuk 7.172 baris (via DeepSeek API).
> 
> **Resume:** Jika gagal di tengah jalan, jalankan ulang — script akan melanjutkan dari baris terakhir yang sudah selesai (progress disimpan di `.labeling_progress.json`).
> 
> **Output:** `dataset/labeled/labeled.csv` (7.172 baris berlabel)

### Step 4 — Training Model SVM (Jupyter Notebook)

```bash
cd ml/training
jupyter notebook main.ipynb
```

> Buka notebook `main.ipynb`, pastikan kernel `mbg-sentiment` terpilih, lalu **Run All Cells**.
> 
> Notebook hanya fokus pada **modelling** — tidak ada cleaning atau labelling di dalamnya.

### Step 5 — Prediksi Teks Baru

```bash
# Mode demo (5 contoh built-in)
python predict.py --demo

# Mode interaktif
python predict.py
# → ketik teks → dapat sentimen
```

---

## Detail Tiap Script

### `dataset/raw/merged.py`

Menggabungkan 4 sumber data mentah ke dalam 1 file CSV dengan format seragam.

**Kolom yang dihasilkan:**
| Kolom   | Deskripsi                                    |
|---------|----------------------------------------------|
| `text`  | Teks konten / komen (kolom bervariasi per sumber, disamakan) |
| `date`  | Tanggal posting (format datetime)            |
| `source`| Sumber data: `x`, `youtube`, `instagram`, `kaggle` |

**Jumlah akhir:** 7.847 baris (sebelum cleaning)

---

### `ml/training/data_clean.py`

Membersihkan teks mentah agar siap untuk labelling & training.

**Langkah cleaning (berurutan):**

| # | Langkah | Contoh |
|---|---------|--------|
| 1 | Decode HTML entities | `&amp;` → `&`, `&lt;` → `<` |
| 2 | Hapus URL | `https://t.co/xyz` → *(dihilangkan)* |
| 3 | Hapus mention | `@username` → *(dihilangkan)* |
| 4 | Hapus simbol `#` tapi simpan kata | `#MBG` → `MBG` |
| 5 | Normalisasi smart quotes & dash | `“ ” ‘ ’ – —` → `" " ' ' - -` |
| 6 | Hapus emoji (Unicode range) | 😊👍🔥 → *(dihilangkan)* |
| 7 | Hapus non-breaking space | `\u00a0` → spasi normal |
| 8 | Hapus artefak tanda baca berdiri sendiri | `Mantap..` → `Mantap` |
| 9 | Hapus newline & carriage return | `\n \r` → spasi |
| 10 | Collapse multi-spasi | `a   b` → `a b` |
| 11 | Strip wrapping quotes CSV | `"teks"` → `teks` |

**Filter pasca-cleaning:**
- Hapus baris dengan teks kosong
- Minimal **2 kata** dan **5 karakter** (di bawah itu dianggap tidak bermakna)
- Hapus **duplikat teks** (keep first occurrence)

**Output:** `cleaned.csv` — hanya kolom `text`, `date`, `source`

---

### `ml/training/llm_labelling.py`

Melakukan pelabelan sentimen otomatis menggunakan **DeepSeek API** (via OpenAI-compatible client).

**Konfigurasi utama:**

| Parameter              | Nilai    | Keterangan                                     |
|------------------------|----------|------------------------------------------------|
| `INPUT_FILE`           | `../../dataset/cleaned/cleaned.csv` | Data bersih                        |
| `OUTPUT_FILE`          | `../../dataset/labeled/labeled.csv` | Hasil labeling                     |
| `PROGRESS_FILE`        | `../../dataset/labeled/.labeling_progress.json` | State resume           |
| `BATCH_SIZE`           | 35       | Jumlah teks per 1 API call                     |
| `DELAY_BETWEEN_BATCHES`| 1.0      | Jeda antar batch (detik)                       |
| `MAX_RETRIES`          | 3        | Maks retry jika API gagal                      |
| `BACKOFF_BASE`         | 5        | Backoff awal (eksponensial: 5, 25, 125 detik)  |
| `TEST_MODE`            | `False`  | `True` = hanya 35 baris, `False` = semua baris |
| `TEST_LIMIT`           | 35       | Jumlah baris di test mode                      |

**Prompt System (Bahasa Indonesia):**
- Diminta klasifikasikan menjadi `positif`, `negatif`, atau `netral`
- Diinstruksikan menangani sarkasme (pilih sentimen sesungguhnya, bukan literal)
- Teks tidak terkait MBG tetap diberi label `netral`
- Response HARUS JSON array, tanpa teks lain

**Prompt User:** per batch — daftar teks bernomor `[0] ... [34]`, diminta output `[{"index": 0, "label": "positif"}, ...]`

**Resume mechanism:**
- Progress disimpan setiap selesai 1 batch di `.labeling_progress.json`
- Jika script dimatikan di tengah jalan, jalankan ulang → otomatis lanjut dari batch terakhir

**Hasil akhir (dari console output script):**
```
positif   : 2334
negatif   : 2999
netral    : 1839
```

---

### `ml/training/main.ipynb` — Notebook Modelling

Notebook ini hanya berisi **modelling** (tidak ada cleaning/labelling di dalamnya). Terdiri dari 9 cell:

| Cell | Judul                           | Isi                                                        |
|------|---------------------------------|------------------------------------------------------------|
| 1    | Import & Setup                  | Import library, path model & data                          |
| 2    | Baca Data Berlabel              | Load `labeled_v2.csv`, plot distribusi label               |
| 3    | TF‑IDF Vectorization            | `TfidfVectorizer` → 10k fitur, 1-2 gram, `sublinear_tf=True` |
| 4    | Train/Test Split                | Split 80:20 stratified (`random_state=42`)                 |
| 5    | Grid Search Tuning + Training   | 4 candidate × 5 folds = 20 fits, scoring `f1_macro`        |
| 6    | Evaluasi                        | Classification report + confusion matrix heatmap           |
| 7    | Simpan Model                    | `joblib.dump()` model & vectorizer ke `ml/models/`         |
| 8    | Demo Prediksi                   | 5 contoh teks Bahasa Indonesia                             |
| 9    | Wordcloud                       | Visualisasi kata paling dominan di dataset                 |

---

## Hasil Training & Evaluasi

### Parameter SVM Terbaik (GridSearchCV)

| Parameter          | Nilai        |
|--------------------|--------------|
| Kernel             | `linear`     |
| C                  | **1**        |
| `class_weight`     | `balanced`   |
| CV f1_macro        | **0.7460**   |

### Classification Report (Test Set — 1.435 baris)

| Kelas    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| `negatif` | 0.75      | 0.79   | 0.77     | 600     |
| `netral`  | 0.56      | 0.60   | 0.58     | 368     |
| `positif` | 0.88      | 0.77   | 0.83     | 467     |

| Metrik       | Nilai |
|--------------|-------|
| **Accuracy** | 0.74  |
| Macro Avg F1 | 0.72  |
| Weighted Avg F1 | 0.74 |

### Confusion Matrix

|            | Pred Negatif | Pred Netral | Pred Positif |
|------------|-------------|------------|-------------|
| **Negatif**  | 476         | 72         | 52          |
| **Netral**   | 79          | 220        | 69          |
| **Positif**  | 84          | 24         | 359         |

### Analisis Evaluasi

- **Kelas positif** memiliki performa terbaik (F1 = 0.83, precision 0.88) — model paling akurat membedakan teks pro-MBG.
- **Kelas negatif** juga baik (F1 = 0.77, recall 0.79) — model cukup andal mendeteksi kritik.
- **Kelas netral** adalah yang terlemah (F1 = 0.58) karena:
  - Jumlah data paling sedikit (1.839 baris dari 7.172)
  - Sering overlap dengan teks positif/negatif bernada ringan
  - Mudah tertukar dengan sentimen lemah di kedua arah

**Kesimpulan:** Model cukup solid untuk produksi, terutama untuk mendeteksi sentimen positif dan negatif yang kuat. Untuk netral perlu perbaikan — bisa dengan tambahan data netral atau threshold confidence.

---

## Prediksi (`predict.py`)

CLI tool sederhana untuk memprediksi sentimen teks baru.

### Cara Pakai

```bash
# Mode demo — 5 contoh built-in
python predict.py --demo

# Mode interaktif — ketik teks apa saja
python predict.py
```

### Output Demo

```
Text 1: Program makan bergizi gratis ini sangat membantu anak-anak kurang mampu...
  Sentimen: + positif

Text 2: Makan bergizi gratis katanya, tapi kenyataannya menunya tidak layak...
  Sentimen: - negatif

Text 3: Pemerintah resmi meluncurkan program makan bergizi gratis untuk siswa SD...
  Sentimen: ~ netral
```

### Cara Kerja

1. Load model SVM & TF-IDF vectorizer dari `ml/models/` (dengan `joblib`)
2. Transform teks input via vectorizer
3. Prediksi label via `model.predict()`
4. Return `positif` / `negatif` / `netral`

---

## Distribusi Label

### Setelah Labelling DeepSeek (`labeled.csv`)

| Label    | Jumlah | Persentase |
|----------|--------|------------|
| `positif` | 2.334  | 32.5%      |
| `negatif` | 2.999  | 41.8%      |
| `netral`  | 1.839  | 25.6%      |
| **Total** | **7.172** | **100%** |

### Setelah Merge Manual Priority (`labeled_v2.csv`) — data training final

Sama karena manual_labeled.csv adalah subset dari labeled.csv (belum ada koreksi label). Setelah koreksi manual selesai, distribusi bisa berubah.

---

## Catatan Penting / Tips

### 1. Path Relatif

Semua path di script `ml/training/*.py` menggunakan path relatif dari direktori `ml/training/`:

```python
# Di data_clean.py dan llm_labelling.py:
INPUT_FILE = "../../dataset/..."
OUTPUT_FILE = "../../dataset/..."
```

> Jalankan python dari root project agar path ini benar:
> ```bash
> python ml/training/data_clean.py       # ✅ benar
> cd ml/training && python data_clean.py  # ❌ salah (path rusak)
> ```

### 2. Test Mode di Labelling

Sebelum labelling 7k+ baris, **selalu uji dengan test mode dulu**:

```python
# Di llm_labelling.py
TEST_MODE = True   # Hanya 35 baris pertama
TEST_LIMIT = 35
```

Cek outputnya di console — pastikan prompt menghasilkan label yang masuk akal. Baru set `TEST_MODE = False` untuk full run.

### 3. Resume Labelling

Jika labelling gagal di tengah jalan (API down, internet putus, dsb):

- **Jangan hapus** `.labeling_progress.json`
- Jalankan ulang script — akan otomatis lanjut dari batch terakhir
- Tidak ada biaya ganda karena baris yang sudah selesai tidak dikirim ulang

### 4. API Key

DeepSeek API key disimpan di `.env` (root project):
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Script memakai `openai` library dengan `base_url="https://api.deepseek.com"` — kompatibel penuh dengan OpenAI SDK.

### 5. Tidak Ada Stemming

Pipeline ini **tidak** menggunakan Sastrawi atau stemming apapun. Hanya cleaning dasar (hapus URL, mention, emoji, HTML entities). TF-IDF bekerja langsung di kata asli Bahasa Indonesia.

### 6. Kenapa SVM?

- SVM linear cocok untuk data teks berdimensi tinggi (TF-IDF 10k fitur)
- Cepat training & prediksi
- Dengan `class_weight='balanced'` menangani imbalance kelas
- Tidak butuh GPU — bisa training di CPU biasa

### 7. Cara Mulai Ulang dari Nol

Jika ingin training ulang dari nol (misal ada data baru), hapus file intermediate:

```bash
rm dataset/cleaned/cleaned.csv
rm dataset/labeled/labeled.csv
rm dataset/labeled/labeled_v2.csv
rm dataset/labeled/manual_labeled.csv
rm dataset/labeled/.labeling_progress.json
rm ml/models/svm_sentiment.pkl
rm ml/models/tfidf_vectorizer.pkl
```

Lalu jalankan ulang step 1–4.

### 8. Scraping X/Twitter

Hanya X/Twitter yang punya scraper otomatis (`dataset/scraping_scripts/x_scraper.py`). Scraper ini butuh cookie autentikasi dan konfigurasi di `.env`:

```
START_DATE_TWEET=2025-01-01
END_DATE_TWEET=2025-06-30
```

**YouTube, Instagram, dan Kaggle** diambil manual — export dari website open-source, format disesuaikan lalu digabung di `merged.py`.

### 9. Next Steps (Rekomendasi Pengembangan)

- Review dan koreksi **300 baris manual_labeled.csv** — perbaiki label yang salah
- Setelah koreksi, re-merge dan training ulang — evaluasi apakah akurasi meningkat
- Tambahkan **threshold confidence** — prediksi netral jika probabilitas rendah
- Eksperimen dengan **SMOTE** (Synthetic Minority Over-sampling) untuk menambah data netral
- Coba **ensemble** SVM + model lain (Random Forest, XGBoost)
- Build **REST API** dengan FastAPI untuk prediksi via HTTP

---

## Ringkasan Angka

| Metrik         | Nilai                |
|----------------|----------------------|
| Data mentah    | 7.847 baris          |
| Data bersih    | 7.172 baris          |
| Data training  | 7.172 baris          |
| Train/test     | 5.737 / 1.435        |
| Label negatif  | 2.999                |
| Label positif  | 2.334                |
| Label netral   | 1.839                |
| Fitur TF-IDF   | 10.000               |
| SVM C terbaik  | 1                    |
| CV F1 (macro)  | 0.7460               |
| Test Accuracy  | 0.74                 |
| Test F1 macro  | 0.72                 |

---
