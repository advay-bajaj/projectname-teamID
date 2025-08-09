
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION VARIABLES
# ============================================================================
DATA_DIRECTORY = r"/Users/advaybajaj/Desktop/internship"
TARGET_DATE = '2025-05-16'
SYNTH_FUTURE_RANGE = 0.03  # 3%

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_process_data():
    """Load and process the options data"""
    start_time = time.time()
    
    # Load the main options data
    file_path = f"{DATA_DIRECTORY}/combined_bhavcopy.csv"
    df = pd.read_csv(file_path)
    load_time = time.time() - start_time
    
    # Convert dates to datetime
    date_convert_start = time.time()
    df['TradDt'] = pd.to_datetime(df['TradDt'])
    df['FininstrmActlXpryDt'] = pd.to_datetime(df['FininstrmActlXpryDt'])
    date_convert_time = time.time() - date_convert_start
    
    # Filter for the target date
    filter_start = time.time()
    target_date = pd.to_datetime(TARGET_DATE)
    df = df[df['TradDt'] == target_date].copy()
    filter_time = time.time() - filter_start
    
    total_time = time.time() - start_time
    
    print(f"Data loading and processing completed:")
    print(f"  - File loading: {load_time:.3f} seconds")
    print(f"  - Date conversion: {date_convert_time:.3f} seconds")
    print(f"  - Date filtering: {filter_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - Data filtered for {target_date.strftime('%d.%m.%Y')}: {len(df)} rows")
    
    return df

def load_working_days():
    """Load working days data"""
    start_time = time.time()
    
    working_days_path = f"{DATA_DIRECTORY}/workingdays.csv"
    working_days_df = pd.read_csv(working_days_path, header=None, names=['Date'])
    load_time = time.time() - start_time
    
    # Convert to datetime
    date_convert_start = time.time()
    working_days_df['Date'] = pd.to_datetime(working_days_df['Date'], format='%d/%m/%Y')
    working_days_df = working_days_df.sort_values('Date').reset_index(drop=True)
    date_convert_time = time.time() - date_convert_start
    
    total_time = time.time() - start_time
    
    print(f"Working days loading completed:")
    print(f"  - File loading: {load_time:.3f} seconds")
    print(f"  - Date conversion & sorting: {date_convert_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - Working days loaded: {len(working_days_df)} days")
    
    return working_days_df

def calculate_strike_range(df, synth_future_range=SYNTH_FUTURE_RANGE):
    """Calculate strike price range for each symbol-expiry combination"""
    start_time = time.time()
    
    # Get unique combinations of TckrSymb and FininstrmActlXpryDt with their underlying prices
    unique_combinations = df.groupby(['TckrSymb', 'FininstrmActlXpryDt'])['UndrlygPric'].first().reset_index()
    
    # Calculate bounds
    unique_combinations['lower_bound'] = unique_combinations['UndrlygPric'] * (1 - synth_future_range)
    unique_combinations['upper_bound'] = unique_combinations['UndrlygPric'] * (1 + synth_future_range)
    
    total_time = time.time() - start_time
    
    print(f"Strike range calculation completed:")
    print(f"  - Time taken: {total_time:.3f} seconds")
    print(f"  - Strike price ranges calculated for {len(unique_combinations)} symbol-expiry combinations")
    
    return unique_combinations

def filter_data_by_strike_range(df, strike_ranges):
    """Filter data based on strike price ranges"""
    start_time = time.time()
    
    # Merge with strike ranges
    merge_start = time.time()
    df_merged = pd.merge(df, strike_ranges, on=['TckrSymb', 'FininstrmActlXpryDt'], suffixes=('', '_range'))
    merge_time = time.time() - merge_start
    
    # Filter based on strike price range
    filter_start = time.time()
    df_filtered = df_merged[
        (df_merged['StrkPric'] >= df_merged['lower_bound']) & 
        (df_merged['StrkPric'] <= df_merged['upper_bound'])
    ].copy()
    filter_time = time.time() - filter_start
    
    total_time = time.time() - start_time
    
    print(f"Strike range filtering completed:")
    print(f"  - Merge time: {merge_time:.3f} seconds")
    print(f"  - Filter time: {filter_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - Data filtered by strike range: {len(df_filtered)} rows")
    
    return df_filtered

def calculate_synthetic_future_prices(df_filtered):
    """Calculate synthetic future prices using put-call parity"""
    start_time = time.time()
    
    # Remove rows with null option types
    clean_start = time.time()
    df_clean = df_filtered.dropna(subset=['OptnTp']).copy()
    clean_time = time.time() - clean_start
    
    # Pivot to get CE and PE prices for each strike
    pivot_start = time.time()
    pivot_df = df_clean.pivot_table(
        index=['TckrSymb', 'FininstrmActlXpryDt', 'StrkPric', 'UndrlygPric'],
        columns='OptnTp',
        values='LastPric',
        aggfunc='first'
    ).reset_index()
    pivot_time = time.time() - pivot_start
    
    # Calculate synthetic future price: Strike + Call - Put
    calc_start = time.time()
    pivot_df['synthetic_future'] = pivot_df['StrkPric'] + pivot_df['CE'] - pivot_df['PE']
    
    # Remove rows where we don't have both CE and PE prices
    pivot_df = pivot_df.dropna(subset=['CE', 'PE']).copy()
    
    # Calculate average synthetic future price by symbol-expiry combination
    avg_synthetic_future = pivot_df.groupby(['TckrSymb', 'FininstrmActlXpryDt'])['synthetic_future'].mean().reset_index()
    avg_synthetic_future.columns = ['TckrSymb', 'FininstrmActlXpryDt', 'avg_synthetic_future']
    calc_time = time.time() - calc_start
    
    total_time = time.time() - start_time
    
    print(f"Synthetic future price calculation completed:")
    print(f"  - Data cleaning: {clean_time:.3f} seconds")
    print(f"  - Pivot operation: {pivot_time:.3f} seconds")
    print(f"  - Calculation & aggregation: {calc_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - Synthetic future prices calculated for {len(avg_synthetic_future)} symbol-expiry combinations")
    
    return avg_synthetic_future

def calculate_days_to_expiry(df, working_days_df, trade_date=TARGET_DATE):
    """Calculate days to expiry using working days"""
    start_time = time.time()
    
    trade_date = pd.to_datetime(trade_date)
    
    # Get unique expiry dates from the data
    unique_expiry_dates = df['FininstrmActlXpryDt'].unique()
    
    # Find the position of trade date in working days
    lookup_start = time.time()
    trade_date_idx = working_days_df[working_days_df['Date'] == trade_date].index
    lookup_time = time.time() - lookup_start
    
    calc_start = time.time()
    if len(trade_date_idx) == 0:
        print(f"Warning: Trade date {trade_date} not found in working days. Using approximate calculation.")
        # Fallback to calendar days calculation
        dte_dict = {}
        for expiry_date in unique_expiry_dates:
            days_diff = (pd.to_datetime(expiry_date) - trade_date).days
            dte_dict[expiry_date] = max(1, days_diff)  # Ensure at least 1 day
        calc_time = time.time() - calc_start
        total_time = time.time() - start_time
        
        print(f"DTE calculation completed (fallback method):")
        print(f"  - Date lookup: {lookup_time:.3f} seconds")
        print(f"  - DTE calculation: {calc_time:.3f} seconds")
        print(f"  - Total time: {total_time:.3f} seconds")
        print(f"  - Days to expiry calculated for {len(dte_dict)} expiry dates")
        
        return dte_dict
    
    trade_date_position = trade_date_idx[0]
    
    # Calculate DTE for each expiry date
    dte_dict = {}
    for expiry_date in unique_expiry_dates:
        expiry_date_idx = working_days_df[working_days_df['Date'] == pd.to_datetime(expiry_date)].index
        
        if len(expiry_date_idx) == 0:
            # If expiry date not found in working days, use approximate calculation
            days_diff = (pd.to_datetime(expiry_date) - trade_date).days
            dte_dict[expiry_date] = max(1, days_diff)
        else:
            expiry_date_position = expiry_date_idx[0]
            dte = max(1, expiry_date_position - trade_date_position)  # Ensure at least 1 day
            dte_dict[expiry_date] = dte
    
    calc_time = time.time() - calc_start
    total_time = time.time() - start_time
    
    print(f"DTE calculation completed:")
    print(f"  - Date lookup: {lookup_time:.3f} seconds")
    print(f"  - DTE calculation: {calc_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - Days to expiry calculated for {len(dte_dict)} expiry dates")
    
    return dte_dict

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_delta(S, K, T, r, sigma, option_type):
    """Calculate option delta"""
    if T <= 0:
        if option_type == 'CE':
            return 1.0 if S > K else 0.0
        else:  # PE
            return -1.0 if S < K else 0.0
    
    if sigma <= 0:
        if option_type == 'CE':
            return 1.0 if S > K else 0.0
        else:  # PE
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'CE':
        delta = norm.cdf(d1)
    else:  # PE
        delta = norm.cdf(d1) - 1
    
    return delta

def calculate_implied_volatility(market_price, S, K, T, r, option_type):
    """Calculate implied volatility using Brent's method"""
    if T <= 0:
        return 0.0
    
    if market_price <= 0:
        return 0.0
    
    # Intrinsic value
    if option_type == 'CE':
        intrinsic = max(0, S - K)
    else:  # PE
        intrinsic = max(0, K - S)
    
    if market_price <= intrinsic:
        return 0.0
    
    def objective(sigma):
        if option_type == 'CE':
            theoretical_price = black_scholes_call(S, K, T, r, sigma)
        else:  # PE
            theoretical_price = black_scholes_put(S, K, T, r, sigma)
        return abs(theoretical_price - market_price)
    
    try:
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        return result.x if result.success else 0.0
    except:
        return 0.0

def calculate_iv_and_delta(df, synthetic_future_prices, dte_dict):
    """Calculate IV and Delta for all options"""
    start_time = time.time()
    
    # Merge synthetic future prices
    merge_start = time.time()
    df_with_synth = pd.merge(df, synthetic_future_prices, on=['TckrSymb', 'FininstrmActlXpryDt'], how='left')
    merge_time = time.time() - merge_start
    
    # Add DTE
    prep_start = time.time()
    df_with_synth['DTE'] = df_with_synth['FininstrmActlXpryDt'].map(dte_dict)
    
    # Parameters
    r = 0.0  # Risk-free rate
    dividend_rate = 0.0  # Dividend rate
    
    # Calculate T (time to expiry in years)
    df_with_synth['T'] = df_with_synth['DTE'] / 365.0
    
    # Use synthetic future price as underlying price, fallback to original underlying price
    df_with_synth['S'] = df_with_synth['avg_synthetic_future'].fillna(df_with_synth['UndrlygPric'])
    
    # Remove rows with missing essential data
    df_clean = df_with_synth.dropna(subset=['LastPric', 'StrkPric', 'OptnTp', 'S', 'T']).copy()
    df_clean = df_clean[df_clean['LastPric'] > 0].copy()
    prep_time = time.time() - prep_start
    
    # Calculate IV and Delta
    calc_start = time.time()
    print(f"    Calculating IV and Delta for {len(df_clean)} options...")
    
    iv_values = []
    delta_values = []
    
    # Process in batches for better performance tracking
    batch_size = 1000
    total_batches = (len(df_clean) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start_time = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df_clean))
        
        batch_iv = []
        batch_delta = []
        
        for idx in range(start_idx, end_idx):
            row = df_clean.iloc[idx]
            iv = calculate_implied_volatility(
                row['LastPric'], row['S'], row['StrkPric'], row['T'], r, row['OptnTp']
            )
            delta = calculate_delta(
                row['S'], row['StrkPric'], row['T'], r, iv, row['OptnTp']
            )
            
            batch_iv.append(iv)
            batch_delta.append(delta)
        
        iv_values.extend(batch_iv)
        delta_values.extend(batch_delta)
        
        batch_time = time.time() - batch_start_time
        if batch_idx % 5 == 0 or batch_idx == total_batches - 1:  # Print every 5 batches
            print(f"    Batch {batch_idx + 1}/{total_batches} completed in {batch_time:.3f} seconds")
    
    df_clean['IV'] = iv_values
    df_clean['Delta'] = delta_values
    df_clean['Delta_OI'] = df_clean['OpnIntrst'] * df_clean['Delta']
    
    calc_time = time.time() - calc_start
    total_time = time.time() - start_time
    
    print(f"IV and Delta calculation completed:")
    print(f"  - Data merge: {merge_time:.3f} seconds")
    print(f"  - Data preparation: {prep_time:.3f} seconds")
    print(f"  - IV/Delta calculation: {calc_time:.3f} seconds")
    print(f"  - Total time: {total_time:.3f} seconds")
    print(f"  - IV and Delta calculated for {len(df_clean)} options")
    
    return df_clean

def create_summary_dataframe(df_with_greeks):
    """Create summary dataframe with average IV and total Delta_OI"""
    start_time = time.time()
    
    summary_df = df_with_greeks.groupby(['TckrSymb', 'FininstrmActlXpryDt']).agg({
        'IV': 'mean',
        'Delta_OI': 'sum',
        'UndrlygPric': 'first',
        'avg_synthetic_future': 'first',
        'DTE': 'first'
    }).reset_index()
    
    summary_df.columns = ['TckrSymb', 'FininstrmActlXpryDt', 'Avg_IV', 'Total_Delta_OI', 
                         'UndrlygPric', 'Avg_Synthetic_Future', 'DTE']
    
    total_time = time.time() - start_time
    
    print(f"Summary dataframe creation completed:")
    print(f"  - Time taken: {total_time:.3f} seconds")
    print(f"  - Summary created for {len(summary_df)} symbol-expiry combinations")
    
    return summary_df

def main():
    """Main function to execute the complete analysis"""
    
    overall_start_time = time.time()
    
    print("=" * 80)
    print("STARTING OPTIONS IV AND DELTA CALCULATION")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Load and process data
    print("\n1. Loading and processing data...")
    step_start = time.time()
    df = load_and_process_data()
    step1_time = time.time() - step_start
    print(f"   Step 1 completed in {step1_time:.3f} seconds")
    
    # Step 2: Load working days
    print("\n2. Loading working days...")
    step_start = time.time()
    working_days_df = load_working_days()
    step2_time = time.time() - step_start
    print(f"   Step 2 completed in {step2_time:.3f} seconds")
    
    # Step 3: Calculate strike ranges
    print("\n3. Calculating strike price ranges...")
    step_start = time.time()
    strike_ranges = calculate_strike_range(df, SYNTH_FUTURE_RANGE)
    step3_time = time.time() - step_start
    print(f"   Step 3 completed in {step3_time:.3f} seconds")
    
    # Step 4: Filter data by strike range
    print("\n4. Filtering data by strike range...")
    step_start = time.time()
    df_filtered = filter_data_by_strike_range(df, strike_ranges)
    step4_time = time.time() - step_start
    print(f"   Step 4 completed in {step4_time:.3f} seconds")
    
    # Step 5: Calculate synthetic future prices
    print("\n5. Calculating synthetic future prices...")
    step_start = time.time()
    synthetic_future_prices = calculate_synthetic_future_prices(df_filtered)
    step5_time = time.time() - step_start
    print(f"   Step 5 completed in {step5_time:.3f} seconds")
    
    # Step 6: Calculate days to expiry
    print("\n6. Calculating days to expiry...")
    step_start = time.time()
    dte_dict = calculate_days_to_expiry(df, working_days_df)
    step6_time = time.time() - step_start
    print(f"   Step 6 completed in {step6_time:.3f} seconds")
    
    # Step 7: Calculate IV and Delta
    print("\n7. Calculating IV and Delta...")
    step_start = time.time()
    df_with_greeks = calculate_iv_and_delta(df, synthetic_future_prices, dte_dict)
    step7_time = time.time() - step_start
    print(f"   Step 7 completed in {step7_time:.3f} seconds")
    
    # Step 8: Create summary
    print("\n8. Creating summary dataframe...")
    step_start = time.time()
    summary_df = create_summary_dataframe(df_with_greeks)
    step8_time = time.time() - step_start
    print(f"   Step 8 completed in {step8_time:.3f} seconds")
    
    # Step 9: Save results
    print("\n9. Saving results...")
    save_start = time.time()
    
    # Save main dataframe
    main_output_path = f"{DATA_DIRECTORY}/options_with_iv_delta.csv"
    df_with_greeks.to_csv(main_output_path, index=False)
    main_save_time = time.time() - save_start
    
    # Save summary dataframe
    summary_start = time.time()
    summary_output_path = f"{DATA_DIRECTORY}/options_summary.csv"
    summary_df.to_csv(summary_output_path, index=False)
    summary_save_time = time.time() - summary_start
    
    step9_time = time.time() - save_start
    print(f"   Step 9 completed in {step9_time:.3f} seconds")
    
    print(f"File saving completed:")
    print(f"  - Main dataframe save: {main_save_time:.3f} seconds")
    print(f"  - Summary dataframe save: {summary_save_time:.3f} seconds")
    print(f"  - Total save time: {step9_time:.3f} seconds")
    print(f"  - Main dataframe saved to: {main_output_path}")
    print(f"  - Summary dataframe saved to: {summary_output_path}")
    
    # Calculate and display overall timing
    overall_time = time.time() - overall_start_time
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Display step-by-step timing summary
    print(f"\nSTEP-BY-STEP TIMING SUMMARY:")
    print(f"  Step 1 (Data Loading): {step1_time:.3f} seconds")
    print(f"  Step 2 (Working Days): {step2_time:.3f} seconds")
    print(f"  Step 3 (Strike Ranges): {step3_time:.3f} seconds")
    print(f"  Step 4 (Filter by Range): {step4_time:.3f} seconds")
    print(f"  Step 5 (Synthetic Future): {step5_time:.3f} seconds")
    print(f"  Step 6 (Days to Expiry): {step6_time:.3f} seconds")
    print(f"  Step 7 (IV and Delta): {step7_time:.3f} seconds")
    print(f"  Step 8 (Summary): {step8_time:.3f} seconds")
    print(f"  Step 9 (Save Results): {step9_time:.3f} seconds")
    print(f"  TOTAL PROGRAM TIME: {overall_time:.3f} seconds ({overall_time/60:.2f} minutes)")
    
    print(f"\nOVERALL PERFORMANCE SUMMARY:")
    print(f"  - Main DataFrame Shape: {df_with_greeks.shape}")
    print(f"  - Summary DataFrame Shape: {summary_df.shape}")
    print(f"  - Average time per option calculation: {(overall_time/len(df_with_greeks)*1000):.2f} milliseconds")
    
    # Performance breakdown
    print(f"\nPERFORMANCE INSIGHTS:")
    if len(df_with_greeks) > 0:
        options_per_second = len(df_with_greeks) / overall_time
        print(f"  - Options processed per second: {options_per_second:.1f}")
        
        if overall_time > 60:
            print(f"  - Consider optimizing IV calculation for better performance")
        else:
            print(f"  - Performance is good for dataset size")
    
    # Display sample results
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS")
    print("=" * 80)
    
    print("\nSample of main dataframe with IV and Delta:")
    display_cols = ['TckrSymb', 'FininstrmActlXpryDt', 'StrkPric', 'OptnTp', 
                   'LastPric', 'IV', 'Delta', 'Delta_OI', 'DTE']
    print(df_with_greeks[display_cols].head(10).to_string(index=False))
    
    print("\nSummary dataframe:")
    print(summary_df.head(10).to_string(index=False))
    
    print(f"\n🎉 Analysis completed successfully in {overall_time:.2f} seconds!")
    
    return df_with_greeks, summary_df

# Run the analysis
if __name__ == "__main__":
    df_with_greeks, summary_df = main()