#!/usr/bin/env python3
"""Test improved battery storage dispatch"""

print('=== Testing Improved Battery Storage Dispatch ===')

from Power_Market_Model import create_test_system
import numpy as np

# Create market with improved storage
market = create_test_system()
print(f'Market created with {len(market.storage)} storage systems')

# Check storage configuration
for i, storage in enumerate(market.storage):
    print(f'Battery {i}: {storage.capacity} MWh, {storage.max_power} MW')
    print(f'  C-rates: {storage.c_rate_charge}C charge, {storage.c_rate_discharge}C discharge')
    print(f'  Safety margins: {storage.min_soc_safe*100:.0f}%-{storage.max_soc_safe*100:.0f}% SOC')

print('\nRunning 24-hour simulation with improved arbitrage...')
results = market.simulate()

# Analyze storage performance
storage_data = results['storage']
prices = results['price']

print('\n=== Storage Performance Analysis ===')
for storage_key, data in storage_data.items():
    power_data = data['power']
    soc_data = data['soc']
    
    # Calculate energy throughput and cycling
    total_charge = np.sum(np.abs(power_data[power_data < 0])) * market.params.dt
    total_discharge = np.sum(power_data[power_data > 0]) * market.params.dt
    
    # SOC statistics
    min_soc = np.min(soc_data) * 100
    max_soc = np.max(soc_data) * 100
    soc_range = max_soc - min_soc
    
    # Price correlation analysis
    charging_times = power_data < -1  # Significant charging
    discharging_times = power_data > 1  # Significant discharging
    
    if np.sum(charging_times) > 0:
        avg_charge_price = np.mean(prices[charging_times])
    else:
        avg_charge_price = 0
        
    if np.sum(discharging_times) > 0:
        avg_discharge_price = np.mean(prices[discharging_times])
    else:
        avg_discharge_price = 0
    
    print(f'\n{storage_key}:')
    print(f'  Energy charged: {total_charge:.1f} MWh')
    print(f'  Energy discharged: {total_discharge:.1f} MWh')
    print(f'  SOC range: {min_soc:.1f}% - {max_soc:.1f}% (Range: {soc_range:.1f}%)')
    print(f'  Avg charge price: ${avg_charge_price:.2f}/MWh')
    print(f'  Avg discharge price: ${avg_discharge_price:.2f}/MWh')
    
    if avg_discharge_price > avg_charge_price and total_charge > 1:
        arbitrage_value = (avg_discharge_price - avg_charge_price) * min(total_charge, total_discharge)
        print(f'  Estimated arbitrage value: ${arbitrage_value:.2f}')
        print(f'  Price spread captured: ${avg_discharge_price - avg_charge_price:.2f}/MWh')
    else:
        print(f'  Warning: Limited arbitrage activity!')

print(f'\nOverall price range: ${np.min(prices):.2f} - ${np.max(prices):.2f}/MWh')

# Check if storage is cycling properly
total_operations = 0
for storage_key, data in storage_data.items():
    power_changes = np.abs(np.diff(data['power']))
    operations = np.sum(power_changes > 1.0)  # Significant power changes
    total_operations += operations
    print(f'{storage_key}: {operations} significant power changes')

if total_operations > 100:
    print('SUCCESS: Storage systems are actively cycling!')
else:
    print('WARNING: Storage systems may not be cycling enough')

print('=== Test Complete ===')