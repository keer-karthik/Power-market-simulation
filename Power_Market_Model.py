import numpy as np
import matplotlib.pyplot as plt

def simulate_power_market():
    """
    Simulate dynamic electricity market under perfect competition
    Based on Liu, Ni, Wu (2004) "Control Theory Application in Power Market Stability Analysis"
    """
    
    # Generator parameters (6 generators - IEEE 30-bus system)
    n_generators = 6
    b = np.array([2, 1.75, 3, 3, 1, 3.25])      # Linear cost coefficients
    c = np.array([0.02, 0.0175, 0.025, 0.025, 0.0625, 0.00834])  # Quadratic cost coefficients
    A = np.array([3, 4, 2.5, 3, 2, 4])          # Dynamic adjustment parameters
    a = np.array([0, 0, 0, 0, 0, 0])             # Constant cost terms (not used in dynamics)
    
    # Demand curve parameters: p = e - f * sum(qi)
    e = 50    # Demand intercept
    f = 0.02  # Demand slope
    
    # Simulation parameters
    hours = 24
    minutes_per_hour = 60
    total_steps = hours * minutes_per_hour  # 1440 steps (1-minute resolution)
    dt = 1.0 / 60.0  # Time step in hours (1 minute)
    
    # Initialize arrays
    q = np.zeros((total_steps + 1, n_generators))  # Generator outputs [MW]
    p = np.zeros(total_steps + 1)                  # Market price [$/MWh]
    time_hours = np.linspace(0, hours, total_steps + 1)
    
    # Initial conditions: all generators start at 0 MW
    q[0, :] = 0.0
    
    # Calculate initial market price
    total_supply = np.sum(q[0, :])
    p[0] = e - f * total_supply
    
    print("=== Dynamic Power Market Simulation ===")
    print("Based on Liu, Ni, Wu (2004) - Perfect Competition Model")
    print(f"Generators: {n_generators}")
    print(f"Simulation time: {hours} hours")
    print(f"Time step: {dt*60:.1f} minutes")
    print(f"Total steps: {total_steps}")
    print()
    
    # Main simulation loop using Euler's method
    for t in range(total_steps):
        # Current market price
        total_supply = np.sum(q[t, :])
        p[t] = e - f * total_supply
        
        # Update each generator's output using Equation (8):
        # dqi/dt = Ai * (p - bi - ci*qi)
        for i in range(n_generators):
            dq_dt = A[i] * (p[t] - b[i] - c[i] * q[t, i])
            q[t+1, i] = q[t, i] + dt * dq_dt
            
            # Ensure non-negative output
            q[t+1, i] = max(0, q[t+1, i])
    
    # Calculate final market price
    total_supply = np.sum(q[-1, :])
    p[-1] = e - f * total_supply
    
    # Calculate theoretical equilibrium from Equation (14)
    sum_b_over_c = np.sum(b / c)
    sum_1_over_c = np.sum(1 / c)
    
    p_star = (e + f * sum_b_over_c) / (1 + f * sum_1_over_c)
    q_star = (p_star - b) / c
    
    # Display results
    print("=== Simulation Results ===")
    print(f"Final market price: ${p[-1]:.2f}/MWh")
    print(f"Theoretical equilibrium price: ${p_star:.2f}/MWh")
    print(f"Price error: {abs(p[-1] - p_star):.4f} $/MWh ({abs(p[-1] - p_star)/p_star*100:.2f}%)")
    print()
    
    print("Generator Outputs (MW):")
    print("Gen | Simulated | Theoretical | Error")
    print("----|-----------|-------------|-------")
    for i in range(n_generators):
        error = abs(q[-1, i] - q_star[i])
        print(f" {i+1}  | {q[-1, i]:8.2f}  | {q_star[i]:10.2f}  | {error:.3f}")
    
    print()
    total_simulated = np.sum(q[-1, :])
    total_theoretical = np.sum(q_star)
    print(f"Total Supply: {total_simulated:.2f} MW (simulated) vs {total_theoretical:.2f} MW (theoretical)")
    
    # Check stability condition (Equation 12): ci > 0
    print("\n=== Stability Analysis ===")
    print("Perfect competition stability condition: ci > 0 for all i")
    for i in range(n_generators):
        stable = "✓" if c[i] > 0 else "✗"
        print(f"Generator {i+1}: c{i+1} = {c[i]:.5f} {stable}")
    
    all_stable = all(c > 0)
    print(f"Market stability: {'STABLE' if all_stable else 'UNSTABLE'}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Market price over time
    ax1.plot(time_hours, p, 'b-', linewidth=2, label='Market Price')
    ax1.axhline(y=p_star, color='r', linestyle='--', linewidth=2, label=f'Equilibrium Price (${p_star:.2f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($/MWh)')
    ax1.set_title('Market Price Evolution - Perfect Competition')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Set x-axis labels for 24-hour period (12 AM to 12 AM in 4-hour intervals)
    hour_ticks = np.arange(0, 25, 4)  # 0, 4, 8, 12, 16, 20, 24
    hour_labels = ['12 AM', '4 AM', '8 AM', '12 PM', '4 PM', '8 PM', '12 AM']
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels(hour_labels)
    ax1.set_xlim(0, 24)
    
    # Plot 2: Generator outputs over time
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for i in range(n_generators):
        ax2.plot(time_hours, q[:, i], color=colors[i], linewidth=2, 
                label=f'Gen {i+1} (→{q_star[i]:.1f} MW)')
        ax2.axhline(y=q_star[i], color=colors[i], linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Output (MW)')
    ax2.set_title('Generator Output Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set same x-axis labels for generator plot
    ax2.set_xticks(hour_ticks)
    ax2.set_xticklabels(hour_labels)
    ax2.set_xlim(0, 24)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: convergence metrics
    print("\n=== Convergence Analysis ===")
    
    # Find when price reaches 95% of equilibrium
    price_threshold = 0.95 * p_star
    convergence_idx = np.where(np.abs(p - p_star) <= 0.05 * p_star)[0]
    if len(convergence_idx) > 0:
        convergence_time = time_hours[convergence_idx[0]]
        print(f"Price converged to within 5% of equilibrium after {convergence_time:.2f} hours")
    else:
        print("Price did not converge to within 5% of equilibrium in simulation time")
    
    # Calculate system eigenvalues (approximate from final convergence rates)
    print(f"\nSystem parameters:")
    print(f"Time constants Ti = 1/(Ai*ci):")
    T = 1 / (A * c)
    for i in range(n_generators):
        print(f"  T{i+1} = {T[i]:.3f} hours")
    
    print(f"\nApproximate eigenvalues (poles): -1/Ti")
    eigenvalues = -1 / T
    for i in range(n_generators):
        print(f"  λ{i+1} = {eigenvalues[i]:.3f}")
    
    return {
        'time': time_hours,
        'price': p,
        'outputs': q,
        'equilibrium_price': p_star,
        'equilibrium_outputs': q_star,
        'parameters': {'b': b, 'c': c, 'A': A, 'e': e, 'f': f}
    }

if __name__ == "__main__":
    results = simulate_power_market()