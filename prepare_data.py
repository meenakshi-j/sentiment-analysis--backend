import os
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
DATA_FOLDER = "datasets"  # folder with CSV files
OUTPUT_FILE = "merged_reviews.csv"
MAX_ROWS_PER_FILE = 100000  # limit rows per CSV to save memory

# ---------- DEBUG: Check folder ----------
print("Current working directory:", os.getcwd())
print("Checking DATA_FOLDER:", os.path.abspath(DATA_FOLDER))

if not os.path.exists(DATA_FOLDER):
    print("❌ Folder does not exist! Please create 'datasets' folder here.")
    exit()

files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

if not files:
    print("❌ No CSV files found in", DATA_FOLDER)
    exit()

print("✅ Found CSV files:", files)

# ---------- HELPER FUNCTION ----------
def rating_to_sentiment(rating):
    if pd.isna(rating):
        return np.nan
    try:
        rating = float(rating)
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return np.nan

# ---------- MERGE DATA ----------
merged_data = pd.DataFrame(columns=["text", "sentiment"])
problem_files = []  # store files that fail

for file in files:
    path = os.path.join(DATA_FOLDER, file)
    print(f"\nProcessing file: {file}")
    
    # Try different encodings automatically
    df = None
    for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception as e:
            continue
    
    if df is None:
        print(f"❌ Could not read {file} with common encodings.")
        problem_files.append(file)
        continue

    # ---------- CLEAN HEADERS ----------
    df.columns = [col.strip().lower() for col in df.columns]  # trim spaces, lowercase
    
    # ---------- DETECT COLUMNS ----------
    text_col = None
    sentiment_col = None
    
    for col in df.columns:
        if any(keyword in col for keyword in ["review", "content", "text", "review_text", "review_header"]):
            text_col = col
        if "sentiment" in col:
            sentiment_col = col
        if any(keyword in col for keyword in ["rate", "rating", "score", "own_rating"]):
            if sentiment_col is None:
                df["sentiment"] = df[col].apply(rating_to_sentiment)
    
    if text_col is None:
        print(f"⚠ Skipping {file}, no review/text column found")
        problem_files.append(file)
        continue
    
    if sentiment_col is None and "sentiment" not in df.columns:
        df["sentiment"] = df.apply(lambda row: rating_to_sentiment(
            row.get("rate") or row.get("score") or row.get("rating") or row.get("own_rating")
        ), axis=1)
    
    temp = df[[text_col, "sentiment"]].rename(columns={text_col: "text"})
    temp = temp.dropna(subset=["text", "sentiment"])
    
    if len(temp) == 0:
        print(f"⚠ Skipping {file}, no valid rows after cleaning")
        problem_files.append(file)
        continue
    
    # Limit rows
    if len(temp) > MAX_ROWS_PER_FILE:
        temp = temp.sample(MAX_ROWS_PER_FILE, random_state=42)
    
    merged_data = pd.concat([merged_data, temp], ignore_index=True)
    print(f"✅ Added {len(temp)} rows from {file}")

# Shuffle final data
merged_data = merged_data.sample(frac=1, random_state=42)

# Save merged CSV
merged_data.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Merged dataset saved to {OUTPUT_FILE}")
print("Total rows:", len(merged_data))

# Show files that had problems
if problem_files:
    print("\n⚠ The following files could not be processed fully:")
    for pf in problem_files:
        print("-", pf)