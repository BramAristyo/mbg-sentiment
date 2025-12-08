import csv

CSV_FILE = "dataset/raw/x_raw.csv"

def count_records(csv_file):
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) 
            count = sum(1 for _ in reader)
            return count
    except FileNotFoundError:
        return 0

total = count_records(CSV_FILE)
print(f"Total record: {total}")
