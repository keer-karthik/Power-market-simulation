#!/usr/bin/env python3
"""Analyze the new price threshold behavior"""

from Power_Market_Model import create_test_system
import numpy as np

print('=== Threshold Analysis Test ===')

# Create market and run simulation
market = create_test_system()
results = market.simulate()

# Analyze price thresholds over time
prices = results['price']
time_hours = results['time']

# Get storage system for analysis
storage = market.storage[0]
print(f'Storage price history length: {len(storage.price_history)}')

# Calculate threshold evolution
thresholds_over_time = []
for i in range(len(prices)):
    if i >= 24:  # Once we have 24 hours of history
        recent_prices = prices[max(0, i-24):i]
        p25 = np.percentile(recent_prices, 25)
        p75 = np.percentile(recent_prices, 75)
        charge_threshold = p25 * 0.9
        discharge_threshold = p75 * 1.15
        
        thresholds_over_time.append({
            'time': time_hours[i],
            'price': prices[i],
            'p25': p25,
            'p75': p75,
            'charge_threshold': charge_threshold,
            'discharge_threshold': discharge_threshold,
            'should_charge': prices[i] < charge_threshold,
            'should_discharge': prices[i] > discharge_threshold
        })

print(f'\n=== Price Threshold Analysis ===')
print(f'Overall price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}/MWh')
print(f'Price mean: ${np.mean(prices):.2f}/MWh')
print(f'Price std: ${np.std(prices):.2f}/MWh')

if thresholds_over_time:
    # Analyze final thresholds
    final = thresholds_over_time[-1]
    print(f'\nFinal thresholds (hour {final["time"]:.1f}):')
    print(f'  Current price: ${final["price"]:.2f}/MWh')
    print(f'  P25: ${final["p25"]:.2f}/MWh')
    print(f'  P75: ${final["p75"]:.2f}/MWh')
    print(f'  Charge threshold: ${final["charge_threshold"]:.2f}/MWh')
    print(f'  Discharge threshold: ${final["discharge_threshold"]:.2f}/MWh')
    print(f'  Gap between thresholds: ${final["discharge_threshold"] - final["charge_threshold"]:.2f}/MWh')
    
    # Count charging and discharging opportunities
    charge_opportunities = sum(1 for t in thresholds_over_time if t['should_charge'])
    discharge_opportunities = sum(1 for t in thresholds_over_time if t['should_discharge'])
    
    print(f'\nOpportunity Analysis:')
    print(f'  Charging opportunities: {charge_opportunities}/{len(thresholds_over_time)} ({charge_opportunities/len(thresholds_over_time)*100:.1f}%)')
    print(f'  Discharging opportunities: {discharge_opportunities}/{len(thresholds_over_time)} ({discharge_opportunities/len(thresholds_over_time)*100:.1f}%)')
    
    # Sample some decisions
    print(f'\nSample threshold decisions:')
    sample_indices = [len(thresholds_over_time)//4, len(thresholds_over_time)//2, 3*len(thresholds_over_time)//4]
    for idx in sample_indices:
        if idx < len(thresholds_over_time):
            t = thresholds_over_time[idx]
            action = 'CHARGE' if t['should_charge'] else 'DISCHARGE' if t['should_discharge'] else 'IDLE'
            print(f'  Hour {t["time"]:5.1f}: Price=${t["price"]:5.2f}, Thresholds=[${t["charge_threshold"]:5.2f}, ${t["discharge_threshold"]:5.2f}] -> {action}')

# Analyze actual storage behavior
storage_data = results['storage']
print(f'\n=== Actual Storage Behavior ===')
for storage_key, data in storage_data.items():
    power_data = data['power']
    soc_data = data['soc']
    
    # Charging/discharging periods
    charging_periods = np.sum(power_data < -1)
    discharging_periods = np.sum(power_data > 1)
    idle_periods = np.sum(np.abs(power_data) <= 1)
    
    print(f'\n{storage_key}:')
    print(f'  Charging periods: {charging_periods} ({charging_periods/len(power_data)*100:.1f}%)')
    print(f'  Discharging periods: {discharging_periods} ({discharging_periods/len(power_data)*100:.1f}%)')
    print(f'  Idle periods: {idle_periods} ({idle_periods/len(power_data)*100:.1f}%)')
    
    # Power utilization
    max_power = market.storage[0].max_power if storage_key == 'battery_0' else market.storage[1].max_power
    avg_charge_power = np.mean(np.abs(power_data[power_data < -1])) if np.sum(power_data < -1) > 0 else 0
    avg_discharge_power = np.mean(power_data[power_data > 1]) if np.sum(power_data > 1) > 0 else 0
    
    print(f'  Average charge power: {avg_charge_power:.1f} MW ({avg_charge_power/max_power*100:.1f}% of max)')
    print(f'  Average discharge power: {avg_discharge_power:.1f} MW ({avg_discharge_power/max_power*100:.1f}% of max)')

print('\n=== Analysis Complete ===')