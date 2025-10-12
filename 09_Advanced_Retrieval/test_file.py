import json
import pandas as pd

input_file = "data/arxiv_sample.json"
output_file = "data/arxiv_cleaned_sample.csv"

rows = []
with open(input_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            rows.append({
                "id": data.get("id"),
                "title": data.get("title"),
                "abstract": data.get("abstract"),
                "categories": data.get("categories"),
                "authors": data.get("authors"),
            })
        except json.JSONDecodeError:
            continue  # skip malformed lines

df = pd.DataFrame(rows)
df = df.dropna(subset=["title", "abstract"])

# Clean whitespace and build combined field
for col in ["title", "abstract", "categories", "authors"]:
    df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

df["combined_text"] = (
    "Title: " + df["title"] +
    ". Authors: " + df["authors"] +
    ". Categories: " + df["categories"] +
    ". Abstract: " + df["abstract"]
)

df.to_csv(output_file, index=False)
print(f"✅ Saved cleaned dataset: {len(df)} rows → {output_file}")
