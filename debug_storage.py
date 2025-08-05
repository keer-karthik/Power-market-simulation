#!/usr/bin/env python3
"""Debug storage dispatch to understand price threshold behavior"""

from Power_Market_Model import create_test_system
import numpy as np

print('=== Storage Dispatch Debug Analysis ===')

# Create market with enhanced debug storage
market = create_test_system()

# Add debug logging to one storage system
storage = market.storage[0]
storage.debug_log = []

# Patch the storage update method to add logging
original_update = storage.update_storage

def debug_update_storage(price, dt, grid_frequency=60.0, current_hour=12):
    """Debug version that logs decisions"""
    # Call original method
    original_update(price, dt, grid_frequency, current_hour)
    
    # Log key decision points
    if hasattr(storage, 'price_history') and len(storage.price_history) > 24:
        p25, p75 = storage.calculate_price_percentiles()
        charge_threshold_night = p25 * 0.95 if (22 <= current_hour or current_hour <= 6) else p25 * 0.7
        discharge_threshold = p75 * 1.05 if (17 <= current_hour <= 21) else p75 * 1.4
        
        should_charge = storage.should_charge_aggressively(price, current_hour)
        should_discharge = storage.should_discharge_aggressively(price, current_hour)
        
        if abs(storage.power_output) > 1:  # Log significant actions
            storage.debug_log.append({
                'hour': current_hour,
                'price': price,
                'p25': p25,
                'p75': p75,
                'charge_threshold': charge_threshold_night,
                'discharge_threshold': discharge_threshold,
                'power': storage.power_output,
                'soc': storage.soc,
                'should_charge': should_charge,
                'should_discharge': should_discharge
            })

# Replace with debug version
storage.update_storage = debug_update_storage

print('Running simulation with debug logging...')
results = market.simulate()

# Analyze the debug log
print('\n=== Debug Analysis ===')
print(f'Total debug log entries: {len(storage.debug_log)}')

if len(storage.debug_log) > 0:
    charges = [entry for entry in storage.debug_log if entry['power'] < 0]
    discharges = [entry for entry in storage.debug_log if entry['power'] > 0]
    
    print(f'\nCharging events: {len(charges)}')
    if charges:
        avg_charge_price = np.mean([e['price'] for e in charges])
        avg_p25_during_charge = np.mean([e['p25'] for e in charges])
        print(f'  Average charge price: ${avg_charge_price:.2f}/MWh')
        print(f'  Average P25 during charge: ${avg_p25_during_charge:.2f}/MWh')
        print('  Sample charging decisions:')
        for i, entry in enumerate(charges[:5]):  # Show first 5
            print(f'    Hour {entry["hour"]:02d}: Price=${entry["price"]:.2f}, Threshold=${entry["charge_threshold"]:.2f}, Power={entry["power"]:.1f}MW')
    
    print(f'\nDischarging events: {len(discharges)}')
    if discharges:
        avg_discharge_price = np.mean([e['price'] for e in discharges])
        avg_p75_during_discharge = np.mean([e['p75'] for e in discharges])
        print(f'  Average discharge price: ${avg_discharge_price:.2f}/MWh')
        print(f'  Average P75 during discharge: ${avg_p75_during_discharge:.2f}/MWh')
        print('  Sample discharging decisions:')
        for i, entry in enumerate(discharges[:5]):  # Show first 5
            print(f'    Hour {entry["hour"]:02d}: Price=${entry["price"]:.2f}, Threshold=${entry["discharge_threshold"]:.2f}, Power={entry["power"]:.1f}MW')

# Overall price analysis
prices = results['price']
print(f'\n=== Overall Market Analysis ===')
print(f'Price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}/MWh')
print(f'Price mean: ${np.mean(prices):.2f}/MWh')
print(f'Price P25: ${np.percentile(prices, 25):.2f}/MWh')
print(f'Price P75: ${np.percentile(prices, 75):.2f}/MWh')

# Time-of-day price analysis
time_hours = results['time']
hourly_avg_prices = []
for hour in range(24):
    hour_mask = (time_hours % 24 >= hour) & (time_hours % 24 < hour + 1)
    if np.sum(hour_mask) > 0:
        hourly_avg_prices.append(np.mean(prices[hour_mask]))
    else:
        hourly_avg_prices.append(0)

print(f'\n=== Hourly Price Patterns ===')
print('Hour  Avg Price')
for hour, avg_price in enumerate(hourly_avg_prices):
    marker = ''
    if 22 <= hour or hour <= 6:
        marker = ' (NIGHT - should charge)'
    elif 17 <= hour <= 21:
        marker = ' (EVENING PEAK - should discharge)'
    elif 6 <= hour <= 10:
        marker = ' (MORNING PEAK - should discharge)'
    elif 11 <= hour <= 15:
        marker = ' (MIDDAY - should charge if low)'
        
    print(f'{hour:02d}:00 ${avg_price:6.2f}/MWh{marker}')

print('\n=== Analysis Complete ===')