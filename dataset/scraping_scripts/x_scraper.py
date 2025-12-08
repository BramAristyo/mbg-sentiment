import asyncio
from twikit import Client
import csv
import os
from datetime import datetime, timedelta
import time
import random
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = "dataset/raw/x_raw.csv"
START_DATE = os.getenv("START_DATE_TWEET")
END_DATE = os.getenv("END_DATE_TWEET")

MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY"))
DELAY_BETWEEN_QUERIES = int(os.getenv("DELAY_BETWEEN_QUERIES"))
DELAY_BETWEEN_PAGES = int(os.getenv("DELAY_BETWEEN_PAGES"))
DELAY_AFTER_RATE_LIMIT = int(os.getenv("DELAY_AFTER_RATE_LIMIT"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES"))

INITIAL_BACKOFF = int(os.getenv("INITIAL_BACKOFF"))
MAX_BACKOFF = int(os.getenv("MAX_BACKOFF"))

COOKIES_FILE = "dataset/scraping_scripts/cookie.json"

KEYWORDS = [
    '"Makan Bergizi Gratis"',
    '"MBG"',
    '"Makan Bergizi"',
    '"Program MBG"',
    '"Makan Siang Gratis"',
    '"Program Makan Gratis"',
    '"MBG Prabowo"',
    '"MBG Gibran"',
    '"Stunting"',
    '"Gizi Anak Indonesia"',
    '"Menu Makan Bergizi"',
    '"Daun Kelor Susu"',
    '"Badan Gizi Nasional"',
    '"BGN"',
    '"SPPG"',
    '"Anggaran MBG"',
    '"Penipuan MBG"',
    '"Mitra MBG"',
    '"Dapur MBG TNI"'
]

# Load existing IDs
existing_ids = set()
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row["tweet_id"])

def append_to_csv(data):
    """Append single tweet to CSV"""
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tweet_id", "date", "keyword", "created_at", "username", "text", "likes"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def random_delay(base_delay):
    """Add random jitter to delays to appear more human-like"""
    jitter = random.uniform(0.5, 1.5)
    return base_delay * jitter

async def safe_request(func, *args, **kwargs):
    """Smart retry with exponential backoff for rate limits"""
    backoff = INITIAL_BACKOFF
    
    for attempt in range(MAX_RETRIES):
        try:
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            msg = str(e).lower()
            
            # Handle rate limit (429)
            if "429" in msg or "rate limit" in msg:
                if attempt < MAX_RETRIES - 1:
                    wait_time = min(backoff, MAX_BACKOFF)
                    print(f"[WARNING] Rate limit hit. Waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                    backoff *= 2  # Exponential backoff
                    continue
                else:
                    print(f"[ERROR] Rate limit persists after {MAX_RETRIES} attempts. Waiting {DELAY_AFTER_RATE_LIMIT}s")
                    await asyncio.sleep(DELAY_AFTER_RATE_LIMIT)
                    return None
            
            # Handle other errors
            print(f"[ERROR] {msg}")
            if attempt < MAX_RETRIES - 1:
                wait_time = random_delay(10)
                print(f"[INFO] Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                continue
            return None
    
    return None

async def fetch_query(client, keyword, day):
    """Fetch tweets for a specific keyword and date"""
    date_since = day.strftime("%Y-%m-%d")
    date_until = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    
    query = f'{keyword} since:{date_since} until:{date_until} lang:id -filter:retweets'
    
    count = 0
    cursor = None
    page = 1
    
    print(f"[SEARCH] Searching: {keyword} on {date_since}")
    
    while count < MAX_PER_QUERY:
        print(f"[PAGE] {page}...")
        
        tweets = await safe_request(
            client.search_tweet,
            query=query,
            product='Latest',
            count=20,
            cursor=cursor
        )
        
        if not tweets:
            print(f"[INFO] No more tweets or request failed")
            break
        
        new_in_page = 0
        for t in tweets:
            tweet_id = str(t.id)
            
            if tweet_id in existing_ids:
                continue
            
            data = {
                "tweet_id": tweet_id,
                "date": date_since,
                "keyword": keyword,
                "created_at": t.created_at,
                "username": t.user.screen_name,
                "text": t.full_text.replace('\n', ' ').replace('\r', ' '),
                "likes": t.favorite_count
            }
            
            append_to_csv(data)
            existing_ids.add(tweet_id)
            count += 1
            new_in_page += 1
            
            if count >= MAX_PER_QUERY:
                break
        
        print(f"[SUCCESS] Found {new_in_page} new tweets (total: {count})")
        
        # Check for next page
        cursor = getattr(tweets, "next_cursor", None)
        if not cursor:
            print(f"[INFO] No more pages available")
            break
        
        if count >= MAX_PER_QUERY:
            break
        
        # Delay between pages with random jitter
        delay = random_delay(DELAY_BETWEEN_PAGES)
        print(f"[WAIT] Waiting {delay:.1f}s before next page...")
        await asyncio.sleep(delay)
        page += 1
    
    print(f"[DONE] [{date_since}] {keyword} - {count} new tweets")
    return count

async def main():
    """Main execution"""
    print("[START] Twitter scraper (Rate-Limit Friendly)")
    print(f"[DATE] Range: {START_DATE} to {END_DATE}")
    print(f"[CONFIG] Keywords: {len(KEYWORDS)}")
    print(f"[OUTPUT] File: {CSV_FILE}")
    print(f"[CONFIG] Max per query: {MAX_PER_QUERY}")
    print("=" * 60)
    
    client = Client('en-US')
    
    try:
        client.load_cookies(COOKIES_FILE)
        print("[SUCCESS] Cookies loaded successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to load cookies: {e}")
        return
    
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    total_saved = 0
    total_queries = 0
    
    day = start
    while day <= end:
        print(f"\n{'='*60}")
        print(f"[DATE] Processing: {day.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        for idx, kw in enumerate(KEYWORDS, 1):
            print(f"[{idx}/{len(KEYWORDS)}]", end=" ")
            count = await fetch_query(client, kw, day)
            total_saved += count
            total_queries += 1
            
            # Delay between different keywords
            if idx < len(KEYWORDS):
                delay = random_delay(DELAY_BETWEEN_QUERIES)
                print(f"[WAIT] Waiting {delay:.1f}s before next keyword...\n")
                await asyncio.sleep(delay)
        
        day += timedelta(days=1)
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Scraping complete!")
    print(f"{'='*60}")
    print(f"[STATS] Total queries: {total_queries}")
    print(f"[STATS] Total new tweets saved: {total_saved}")
    print(f"[OUTPUT] File: {CSV_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())