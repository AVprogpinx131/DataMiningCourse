import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- 1. Read & Preprocess the CSV -----
# Skip first 4 rows (anonymization info) and use ';' as delimiter
df = pd.read_csv('your_time_series.csv', sep=';', skiprows=4, header=0)
df.columns = ['Periood', 'consumption']
df['Periood'] = pd.to_datetime(df['Periood'], dayfirst=True)
df['consumption'] = df['consumption'].str.replace(',', '.').astype(float)

# Create date and hour columns
df['date'] = df['Periood'].dt.date
df['hour'] = df['Periood'].dt.hour

# ----- 2. Pivot Data: one row per day with 24 hourly values -----
pt = df.pivot_table(index='date', columns='hour', values='consumption', aggfunc='mean').sort_index()
pt = pt.dropna()  # Keep only complete days
dates = list(pt.index)
print("Total complete days available:", len(dates))

# ----- 3. Define helper functions -----
def compute_rmse(y_pred, y_true):
    """Calculate Root Mean Squared Error between predicted and actual values"""
    # Handle potential length mismatches
    min_len = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:min_len], y_true[:min_len]
    return np.sqrt(np.mean((y_pred - y_true)**2))

def compute_similarity(day1, day2):
    """Calculate the Euclidean distance between two days' hourly patterns"""
    # Handle potential length mismatches
    min_len = min(len(day1), len(day2))
    return np.sqrt(np.sum((day1[:min_len] - day2[:min_len]) ** 2))

def find_similar_days(reference_day, historical_data, n=3):
    """
    Find the n most similar days to the reference day in the historical data.
    
    Args:
        reference_day: The day pattern to compare against
        historical_data: DataFrame with historical daily patterns
        n: Number of similar days to return
    
    Returns:
        List of dates of the most similar days
    """
    similarities = []
    
    for date in historical_data.index:
        day_pattern = historical_data.loc[date].values
        similarity = compute_similarity(reference_day, day_pattern)
        similarities.append((date, similarity))
    
    # Sort by similarity (lowest distance = most similar) and get top n
    similarities.sort(key=lambda x: x[1])
    return [date for date, _ in similarities[:n]]

# ----- 4. Find and plot the 3 most similar days -----
def plot_similar_days(reference_date=None):
    """
    Find and plot the 3 most similar days to the reference date.
    
    Args:
        reference_date: Date to use as reference (if None, use latest date)
    """
    if reference_date is None:
        reference_date = dates[-1]  # Use most recent date
    
    # Get the consumption pattern for the reference day
    reference_pattern = pt.loc[reference_date].values
    
    # Create a copy of the pivot table excluding the reference date
    historical_data = pt.drop(reference_date, errors='ignore')
    
    # Find the 3 most similar days
    similar_days = find_similar_days(reference_pattern, historical_data, n=3)
    
    # Plot the reference day and the 3 most similar days
    plt.figure(figsize=(12, 6))
    
    # Plot reference day
    hours = range(len(reference_pattern))
    plt.plot(hours, reference_pattern, 'b-', linewidth=2, 
             label=f'Reference Day: {reference_date}')
    
    # Plot similar days with different colors and line styles
    colors = ['r', 'g', 'c']
    for i, day in enumerate(similar_days):
        day_values = pt.loc[day].values
        plt.plot(hours[:len(day_values)], day_values[:len(hours)], 
                 f'{colors[i]}-', linewidth=1.5, 
                 label=f'Similar Day {i+1}: {day}')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'Reference Day and 3 Most Similar Days')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('similar_days.png')
    plt.show()
    
    return similar_days

# ----- 5. Main execution -----
def main():
    print("\n----- Finding Similar Days -----")
    
    # Let the user specify a reference date or use the latest date
    use_latest = input("Use latest date as reference? (y/n): ").lower() == 'y'
    
    if use_latest:
        reference_date = dates[-1]
        print(f"Using latest date as reference: {reference_date}")
    else:
        date_str = input("Enter reference date (YYYY-MM-DD): ")
        try:
            reference_date = pd.Timestamp(date_str).date()
            if reference_date not in pt.index:
                print(f"Date {reference_date} not found in data. Using latest date instead.")
                reference_date = dates[-1]
        except:
            print("Invalid date format. Using latest date instead.")
            reference_date = dates[-1]
    
    # Find and plot similar days
    similar_days = plot_similar_days(reference_date)
    
    print("\nThe 3 most similar days to", reference_date, "are:")
    for i, day in enumerate(similar_days):
        print(f"{i+1}. {day}")
    
    # Calculate and show the average pattern
    print("\nCalculating average consumption pattern from similar days...")
    
    # Get the patterns for the similar days
    patterns = [pt.loc[day].values for day in similar_days]
    
    # Calculate the average pattern
    avg_pattern = np.mean(patterns, axis=0)
    
    # Plot the average pattern alongside the reference day
    plt.figure(figsize=(12, 6))
    hours = range(len(reference_pattern))
    
    plt.plot(hours, pt.loc[reference_date].values, 'b-', linewidth=2,
             label=f'Reference Day: {reference_date}')
    plt.plot(hours[:len(avg_pattern)], avg_pattern[:len(hours)], 'r--', linewidth=2,
             label=f'Average of 3 Similar Days')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(f'Reference Day vs. Similar Days Average')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('similar_days_average.png')
    plt.show()
    
    print("Plots saved as 'similar_days.png' and 'similar_days_average.png'")

if __name__ == "__main__":
    main()