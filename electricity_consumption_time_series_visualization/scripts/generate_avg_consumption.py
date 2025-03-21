import pandas as pd
import json
import os

# Get the absolute path to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
datasets_dir = os.path.join(parent_dir, "datasets")
data_dir = os.path.join(parent_dir, "data")

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

# List of CSV file base names
file_base_names = [
    "3fd0482ebd211dd11741080835.csv",
    "9e9dca492a061e211740838882.csv",
    "c550bcace2429c281741504217.csv",
    "ed9f4fcf0bfb1afa1741424674.csv",
    "e19ce9f7cd6985d21742127622.csv",
    "eb18877694cc036a1742320408.csv"
]

# Create full paths
file_names = [os.path.join(datasets_dir, base_name) for base_name in file_base_names]

# Check if files exist
missing_files = [file for file in file_names if not os.path.exists(file)]
if missing_files:
    print("The following files were not found:")
    for file in missing_files:
        print(f"  - {file}")
    print(f"\nMake sure your dataset files are in: {datasets_dir}")
    
    # Continue with files that exist
    file_names = [file for file in file_names if os.path.exists(file)]
    if not file_names:
        print("No valid files found. Exiting.")
        import sys
        sys.exit(1)
    else:
        print(f"Proceeding with {len(file_names)} existing files.")

# Load and combine all CSV files, skipping the first 4 rows
dataframes = []
for file in file_names:
    try:
        # Read the CSV file
        df = pd.read_csv(file, sep=";", skiprows=4)
        
        # Ensure we only take the first two columns (Periood and Energy)
        if len(df.columns) < 2:
            print(f"Error: {file} has fewer than 2 columns.")
            continue
            
        df = df.iloc[:, :2]  # Take only the first two columns
        df.columns = ["Periood", "Energy"]  # Rename to standard column names
        
        # Extract just the file name (without path and extension) as the dataset label
        dataset_name = os.path.basename(file).split(".")[0]
        df["Dataset"] = dataset_name
        dataframes.append(df)
    except Exception as e:
        print(f"Error processing {file}: {e}")

if not dataframes:
    print("No data could be processed. Exiting.")
    import sys
    sys.exit(1)

# Concatenate all DataFrames
all_data = pd.concat(dataframes, ignore_index=True)

# Parse datetime
try:
    all_data["datetime"] = pd.to_datetime(all_data["Periood"], format="%d.%m.%Y %H:%M")
except ValueError as e:
    print(f"Error parsing datetime: {e}")
    print("First few rows of 'Periood' column:")
    print(all_data["Periood"].head())
    raise

# Extract day of week and hour
all_data["day_of_week"] = (all_data["datetime"].dt.dayofweek + 1) % 7  # 0=Sun, 1=Mon, ...
all_data["hour"] = all_data["datetime"].dt.hour

# Convert Energy to float
all_data["Energy"] = all_data["Energy"].str.replace(",", ".").astype(float)

# Compute average consumption by day of week, hour, and dataset
avg_consumption = all_data.groupby(["Dataset", "day_of_week", "hour"])["Energy"].mean().reset_index()
day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
avg_consumption["day_name"] = avg_consumption["day_of_week"].map(lambda x: day_names[x])

# Structure data for D3.js: list of objects, grouped by dataset
datasets = avg_consumption["Dataset"].unique()
data_for_json = []
for dataset in datasets:
    dataset_data = avg_consumption[avg_consumption["Dataset"] == dataset]
    days_data = []
    for day in range(7):
        day_data = dataset_data[dataset_data["day_of_week"] == day]
        hourly_values = day_data.sort_values("hour")[["hour", "Energy"]].to_dict("records")
        days_data.append({
            "day": day_names[day],
            "values": hourly_values
        })
    data_for_json.append({
        "dataset": dataset,
        "data": days_data
    })

# Save to JSON file
output_file = os.path.join(data_dir, "avg_consumption.json")
with open(output_file, "w") as f:
    json.dump(data_for_json, f, indent=2)

print(f"Data saved to ../data/avg_consumption.json")