import pandas as pd

# === 1. Load your CSV file ===
# Change this path to match where you saved the file
csv_path = "data/patch_logs.csv"
df = pd.read_csv(csv_path)

# === 2. Convert the relevant columns to float ===
columns_to_convert = ["skew", "kurt", "std", "pval"]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # safely convert, drop non-numeric

# === 3. Drop rows with any NaN values in these columns ===
df = df.dropna(subset=columns_to_convert)

# === 4. Compute and print the stats ===
summary = {
    "Skewness": (df["skew"].mean(), df["skew"].std()),
    "Kurtosis": (df["kurt"].mean(), df["kurt"].std()),
    "Noise Std": (df["std"].mean(), df["std"].std()),
    "P-Value": (df["pval"].mean(), df["pval"].std()),
}

print("\n📊 Noise Statistics from CSV:")
for metric, (mean, std) in summary.items():
    print(f"{metric:<12} →  mean = {mean:.6f}   std = {std:.6f}")