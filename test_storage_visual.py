#!/usr/bin/env python3
"""Visual test of improved storage dispatch"""

from Power_Market_Model import create_test_system
import numpy as np
import matplotlib.pyplot as plt

print('=== Visual Test of Improved Storage Dispatch ===')

# Create and run simulation
market = create_test_system()
results = market.simulate()

# Create detailed analysis plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price and Storage Power
ax1 = axes[0, 0]
time_hours = results['time']
prices = results['price']

ax1.plot(time_hours, prices, 'b-', linewidth=2, label='Market Price', alpha=0.7)
ax1.set_ylabel('Price ($/MWh)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Add storage power on secondary axis
ax1_twin = ax1.twinx()
for storage_key, data in results['storage'].items():
    power = data['power']
    ax1_twin.plot(time_hours, power, linewidth=2, alpha=0.8, 
                  label=f'{storage_key} Power')

ax1_twin.set_ylabel('Storage Power (MW)', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Price vs Storage Operation')
ax1.set_xlabel('Time (hours)')
ax1.grid(True, alpha=0.3)

# Plot 2: SOC Evolution
ax2 = axes[0, 1]
for storage_key, data in results['storage'].items():
    soc = data['soc'] * 100  # Convert to percentage
    ax2.plot(time_hours, soc, linewidth=2, label=f'{storage_key} SOC')

ax2.set_ylabel('State of Charge (%)')
ax2.set_xlabel('Time (hours)')
ax2.set_title('Storage State of Charge')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# Plot 3: Price vs Storage Action Analysis
ax3 = axes[1, 0]
# Scatter plot of price vs power for all storage systems
all_prices = []
all_powers = []
all_colors = []

for storage_key, data in results['storage'].items():
    power = data['power']
    storage_prices = prices.copy()
    
    # Color code by action: blue=charging, red=discharging, gray=idle
    colors = ['blue' if p < -1 else 'red' if p > 1 else 'gray' for p in power]
    
    all_prices.extend(storage_prices)
    all_powers.extend(power)
    all_colors.extend(colors)

ax3.scatter(all_prices, all_powers, c=all_colors, alpha=0.6, s=20)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax3.set_xlabel('Market Price ($/MWh)')
ax3.set_ylabel('Storage Power (MW)')
ax3.set_title('Price vs Storage Action\n(Blue=Charge, Red=Discharge, Gray=Idle)')
ax3.grid(True, alpha=0.3)

# Plot 4: Time-of-Use Analysis
ax4 = axes[1, 1]
# Calculate average price and storage activity by hour of day
hourly_prices = np.zeros(24)
hourly_charge = np.zeros(24)
hourly_discharge = np.zeros(24)

for hour in range(24):
    hour_mask = (time_hours % 24 >= hour) & (time_hours % 24 < hour + 1)
    if np.sum(hour_mask) > 0:
        hourly_prices[hour] = np.mean(prices[hour_mask])
        
        # Sum all storage activity for this hour
        total_charge = 0
        total_discharge = 0
        for storage_key, data in results['storage'].items():
            power = data['power'][hour_mask]
            total_charge += np.sum(np.abs(power[power < 0]))
            total_discharge += np.sum(power[power > 0])
        
        hourly_charge[hour] = total_charge
        hourly_discharge[hour] = total_discharge

hours = np.arange(24)
ax4_twin = ax4.twinx()

# Plot hourly average prices
ax4.bar(hours, hourly_prices, alpha=0.6, color='orange', label='Avg Price')
ax4.set_ylabel('Average Price ($/MWh)', color='orange')
ax4.tick_params(axis='y', labelcolor='orange')

# Plot storage activity
ax4_twin.bar(hours - 0.2, hourly_charge, width=0.4, alpha=0.7, color='blue', label='Charging')
ax4_twin.bar(hours + 0.2, hourly_discharge, width=0.4, alpha=0.7, color='red', label='Discharging')
ax4_twin.set_ylabel('Storage Activity (MW)', color='black')

ax4.set_xlabel('Hour of Day')
ax4.set_title('Time-of-Use: Price vs Storage Activity')
ax4.set_xticks(hours[::2])
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary analysis
print('\n=== Storage Dispatch Analysis ===')
for storage_key, data in results['storage'].items():
    power = data['power']
    soc = data['soc']
    
    # Calculate efficiency and arbitrage metrics
    charge_energy = np.sum(np.abs(power[power < 0])) * market.params.dt
    discharge_energy = np.sum(power[power > 0]) * market.params.dt
    round_trip_efficiency = discharge_energy / charge_energy if charge_energy > 0 else 0
    
    # Price-based metrics
    charging_mask = power < -1
    discharging_mask = power > 1
    
    if np.sum(charging_mask) > 0 and np.sum(discharging_mask) > 0:
        avg_charge_price = np.mean(prices[charging_mask])
        avg_discharge_price = np.mean(prices[discharging_mask])
        price_differential = avg_discharge_price - avg_charge_price
    else:
        avg_charge_price = avg_discharge_price = price_differential = 0
    
    print(f'\n{storage_key}:')
    print(f'  Round-trip efficiency: {round_trip_efficiency:.1%}')
    print(f'  SOC utilization: {(np.max(soc) - np.min(soc))*100:.1f}%')
    print(f'  Price differential: ${price_differential:.2f}/MWh')
    print(f'  Charge/discharge ratio: {charge_energy/discharge_energy:.2f}' if discharge_energy > 0 else '  No discharge activity')

print('\nAnalysis complete!')