# Sentiment Project 'MBG' Using SVM

A Python project for scraping tweets about "Makan Bergizi Gratis" (Free Nutritious Meals) program and performing sentiment analysis using SVM Model.

## Tech Stack

- **Language:** Python 3.8+
- **Web Framework:** Django 5.2.9
- **Database:** MySQL
- **Data Processing:** Pandas 2.3.3, NumPy 2.2.6
- **Twitter Scraping:** Twikit 2.3.3, SNScrape 0.7.0
- **HTTP Client:** HTTPX 0.28.1, Requests 2.32.5
- **Environment Management:** python-dotenv 1.2.1

## Prerequisites

- Python 3.8 or higher
- MySQL Server 5.7+
- pip (Python package manager)
- MySQL Client (mysql-connector-python or PyMySQL)

## Installation

### 1. Clone or Navigate to Project
```bash
cd c:\Com-Things\PYTHON\sentiment_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup MySQL Database
```bash
mysql -u root -p
CREATE DATABASE sentiment_project CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 6. Setup Environment Variables
```bash
copy .env.example .env
```
Then edit `.env` file with your configuration:
- Twitter API credentials
- Database credentials (MySQL)
- Date ranges for scraping
- Delay and retry settings

**Example .env:**
```
# Django
DEBUG=True
SECRET_KEY=your_secret_key_here

# Database
DB_ENGINE=mysql
DB_NAME=sentiment_project
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=127.0.0.1
DB_PORT=3306

# Twitter Scraper
START_DATE_TWEET=2025-01-01
END_DATE_TWEET=2025-01-31
MAX_PER_QUERY=100
DELAY_BETWEEN_QUERIES=5
DELAY_BETWEEN_PAGES=3
```

### 7. Prepare Cookies
Place your `cookie.json` file in:
```
dataset/scraping_scripts/cookie.json
```

### 8. Run Django Migrations
```bash
python manage.py migrate
```

## Quick Start

### Run Twitter Scraper
```bash
python dataset/scraping_scripts/x_scraper.py
```

Output will be saved to:
```
dataset/raw/x_raw.csv
```

### Start Django Development Server
```bash
python manage.py runserver
```

Access at: `http://localhost:8000`

## Project Structure
```
sentiment_project/
├── dataset/
│   ├── raw/              # Raw scraped data (CSV)
│   ├── scraping_scripts/ # Scraper scripts
│   └── processed/        # Processed data
├── sentiment_app/        # Django app
├── manage.py             # Django management script
├── requirements.txt      # Dependencies
├── .env                  # Configuration (local, do not commit)
├── .env.example          # Configuration template
└── README.md             # This file
```

## Troubleshooting

**Rate Limit Issue:**
- Increase `DELAY_BETWEEN_QUERIES` in `.env`
- Check `MAX_RETRIES` and `DELAY_AFTER_RATE_LIMIT`

**Cookie Error:**
- Ensure `cookie.json` exists in `dataset/scraping_scripts/`
- Update credentials if expired

**MySQL Connection Error:**
- Verify MySQL is running: `mysql -u root -p`
- Check DB credentials in `.env`
- Ensure database exists

**Django Migration Error:**
```bash
python manage.py makemigrations
python manage.py migrate
```
