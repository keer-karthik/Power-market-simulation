#!/usr/bin/env python3
"""
Comprehensive Enhanced Power Market Simulation Demo

This script demonstrates all the major enhancements to the original power market model:
1. Energy Storage Systems (BESS)
2. Renewable Energy with Stochastic Profiles  
3. Demand Response
4. Enhanced Stability Analysis
5. Multi-Market Co-optimization
6. Uncertainty Quantification (Monte Carlo)
7. Machine Learning Stability Prediction

Based on research from:
- Haugen et al. (Power market models for the clean energy transition)
- Tamrakar et al. (Stability Analysis)
- Liu et al. (Control Theory Application in Power Market Stability Analysis)
"""
import sys
import os

# Ensure the script's directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ...existing code...

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
import os

print("Checking for required files...")
required_files = [
    'Power_Market_Model.py',
    'Multi_Market_Extension.py',
    'Uncertainty_Analysis.py',
    'ML_Stability_Predictor.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join(os.path.dirname(__file__), file)):
        missing_files.append(file)
        print(f"[MISSING] {file}")
    else:
        print(f"[FOUND] {file}")

if missing_files:
    print("Missing required files:", missing_files)
    modules_available = False
else:
    print("All files found. Attempting imports...")
    try:
        from Power_Market_Model import (
            EnhancedPowerMarket, MarketParameters, create_test_system
        )
        from Multi_Market_Extension import (
            MultiMarketSimulation, ReserveRequirement
        )
        from Uncertainty_Analysis import (
            MonteCarloAnalyzer, UncertaintyParameters
        )
        from ML_Stability_Predictor import (
            StabilityPredictor, RealTimeStabilityMonitor
        )
        modules_available = True
        print("All modules imported successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
        modules_available = False


def run_basic_enhanced_simulation():
    """Run basic enhanced power market simulation"""
    print("=" * 60)
    print("1. BASIC ENHANCED POWER MARKET SIMULATION")
    print("=" * 60)
    
    # Create enhanced market
    market = create_test_system()
    
    # Run simulation
    results = market.simulate()
    
    # Display key results
    print(f"\nSimulation Results:")
    print(f"Final price: ${results['price'][-1]:.2f}/MWh")
    print(f"Average price: ${np.mean(results['price']):.2f}/MWh")
    print(f"Price volatility: ${np.std(results['price']):.2f}/MWh")
    
    # Generation mix summary
    total_conventional = sum(np.sum(gen_data) for key, gen_data in results['generation'].items() 
                           if key.startswith('gen_'))
    total_renewable = sum(np.sum(gen_data) for key, gen_data in results['generation'].items() 
                        if key.startswith('ren_'))
    total_generation = total_conventional + total_renewable
    
    print(f"\nGeneration Mix:")
    print(f"Conventional: {total_conventional:.0f} MWh ({total_conventional/total_generation*100:.1f}%)")
    print(f"Renewable: {total_renewable:.0f} MWh ({total_renewable/total_generation*100:.1f}%)")
    
    # Storage operation summary
    if results['storage']:
        total_storage_energy = sum(np.sum(np.abs(data['power'])) for data in results['storage'].values())
        print(f"Total storage operation: {total_storage_energy:.0f} MWh")
    
    # System stability
    if results['stability']['stable']:
        print(f"\nSystem Status: STABLE")
        if 'convergence' in results['stability']:
            settling_time = results['stability']['convergence'].get('settling_time', 'N/A')
            print(f"Estimated settling time: {settling_time} hours")
    else:
        print(f"\nSystem Status: UNSTABLE")
    
    # Plot results
    market.plot_results(results)
    
    return results

def run_multi_market_simulation():
    """Run multi-market simulation with energy and ancillary services"""
    print("\n" + "=" * 60)
    print("2. MULTI-MARKET SIMULATION (Energy + Reserves)")
    print("=" * 60)
    
    # Note: This requires CVXPY for optimization
    try:
        import cvxpy as cp
        
        params = MarketParameters(simulation_hours=12, time_step_minutes=15)
        multi_market = MultiMarketSimulation(params)
        
        # Add generators with reserve capabilities
        multi_market.add_enhanced_generator(2.0, 0.02, 3.0, 500, 30)
        multi_market.add_enhanced_generator(1.75, 0.0175, 4.0, 600, 50)
        multi_market.add_enhanced_generator(3.0, 0.025, 2.5, 400, 20)
        
        # Add storage with fast response capabilities
        multi_market.add_enhanced_storage(100, 50, 0.9)
        multi_market.add_enhanced_storage(200, 100, 0.85)
        
        # Run multi-market simulation
        results = multi_market.simulate_multi_market()
        
        print(f"\nMulti-Market Results:")
        print(f"Average energy price: ${np.mean(results['prices']['energy']):.2f}/MWh")
        print(f"Average spinning reserve price: ${np.mean(results['prices']['spinning_reserve']):.2f}/MW-h")
        print(f"Average regulation price: ${np.mean(results['prices']['regulation_up']):.2f}/MW-h")
        
        # Revenue analysis
        print(f"\nRevenue Summary:")
        for unit, rev in results['total_revenue'].items():
            print(f"{unit}: Energy=${rev['energy']:.0f}, Ancillary=${rev['ancillary']:.0f}, "
                  f"Total=${rev['total']:.0f} (Ancillary: {rev['ancillary_share']*100:.1f}%)")
        
        # Plot multi-market results
        plot_multi_market_results(results)
        
    except ImportError:
        print("CVXPY not available - skipping multi-market optimization")
        print("Install with: pip install cvxpy")
        results = None
    
    return results

def run_uncertainty_analysis():
    """Run Monte Carlo uncertainty analysis"""
    print("\n" + "=" * 60)
    print("3. UNCERTAINTY ANALYSIS (Monte Carlo)")
    print("=" * 60)
    
    # Create base market for uncertainty analysis
    params = MarketParameters(simulation_hours=6, time_step_minutes=30)  # Smaller for demo
    base_market = EnhancedPowerMarket(params)
    
    # Add components
    base_market.add_generator(2.0, 0.02, 3.0, 500)
    base_market.add_generator(1.75, 0.0175, 4.0, 600)
    base_market.add_renewable(200, 'solar')
    base_market.add_renewable(150, 'wind')
    base_market.add_battery_storage(100, 50)
    
    # Set up uncertainty parameters
    uncertainty_params = UncertaintyParameters(
        demand_uncertainty=0.08,        # ±8% demand uncertainty
        renewable_uncertainty=0.25,     # ±25% renewable forecast error
        outage_probability=0.002,       # 0.2% hourly outage probability
        fuel_price_volatility=0.15      # ±15% fuel price uncertainty
    )
    
    # Run Monte Carlo analysis
    mc_analyzer = MonteCarloAnalyzer(base_market, uncertainty_params)
    mc_results = mc_analyzer.run_monte_carlo(n_scenarios=50, parallel=False)  # Small number for demo
    
    # Display uncertainty results
    if 'error' not in mc_results['analysis']:
        analysis = mc_results['analysis']
        print(f"\nUncertainty Analysis Results:")
        print(f"Success rate: {analysis['success_rate']:.1%}")
        
        if 'price_mean' in analysis['metrics']:
            price_stats = analysis['metrics']['price_mean']
            print(f"Price uncertainty:")
            print(f"  Mean: ${price_stats['mean']:.2f} ± ${price_stats['std']:.2f}/MWh")
            print(f"  Range (5th-95th percentile): ${price_stats['percentiles']['5th']:.2f} - ${price_stats['percentiles']['95th']:.2f}/MWh")
        
        print(f"\nRisk Metrics:")
        risk = analysis['risk_metrics']
        print(f"  Price VaR (90% confidence): ${risk['price_var']:.2f}/MWh")
        print(f"  Extreme price probability: {risk['extreme_price_prob']:.1%}")
        
        # Plot uncertainty analysis
        mc_analyzer.plot_uncertainty_analysis(mc_results)
    else:
        print(f"Uncertainty analysis failed: {mc_results['analysis']['error']}")
    
    return mc_results

def run_ml_stability_prediction():
    """Run machine learning stability prediction"""
    print("\n" + "=" * 60)
    print("4. MACHINE LEARNING STABILITY PREDICTION")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create and train stability predictor
        predictor = StabilityPredictor()
        print("Training ML stability predictor...")
        training_results = predictor.train(n_training_samples=200)  # Small for demo
        
        print(f"\nTraining Results:")
        print(f"Stability classification accuracy: {training_results['stability_accuracy']:.3f}")
        print(f"Eigenvalue regression R²: {training_results['eigenvalue_r2']:.3f}")
        
        # Test real-time monitoring
        print(f"\nTesting real-time stability monitoring...")
        monitor = RealTimeStabilityMonitor(predictor)
        
        # Simulate some operating conditions
        test_scenarios = [
            {"price_level": 35, "renewable_share": 0.2, "description": "Normal operation"},
            {"price_level": 65, "renewable_share": 0.4, "description": "High prices, moderate renewables"},
            {"price_level": 45, "renewable_share": 0.7, "description": "High renewable penetration"},
            {"price_level": 95, "renewable_share": 0.8, "description": "Extreme conditions"},
            {"price_level": 120, "renewable_share": 0.9, "description": "Critical conditions"}
        ]
        
        print(f"\nStability Monitoring Results:")
        print(f"{'Scenario':<25} {'Alert':<8} {'Stable?':<8} {'Prob':<6} {'Recommendation'}")
        print("-" * 90)
        
        for scenario in test_scenarios:
            # Create test market data
            test_market_data = create_test_market_data(scenario)
            test_system_state = create_test_system_state()
            
            result = monitor.monitor_step(test_market_data, test_system_state)
            pred = result['prediction']
            
            print(f"{scenario['description']:<25} {result['alert_level']:<8} "
                  f"{'Yes' if pred['is_stable'] else 'No':<8} {pred['stability_probability']:<6.3f} "
                  f"{result['recommendation'][:50]}...")
        
        # Plot monitoring trend
        monitor.plot_stability_trend()
        
        return training_results, monitor
        
    except ImportError:
        print("Scikit-learn not available - skipping ML prediction")
        print("Install with: pip install scikit-learn")
        return None, None

def create_test_market_data(scenario):
    """Create test market data for ML prediction"""
    n_hours = 5
    base_price = scenario['price_level']
    renewable_share = scenario['renewable_share']
    
    # Price with some volatility
    prices = base_price * (1 + 0.1 * np.random.normal(0, 1, n_hours))
    
    # Generation mix
    total_gen = 1000
    renewable_gen = total_gen * renewable_share
    conventional_gen = total_gen * (1 - renewable_share)
    
    generation = {
        'gen_0': np.ones(n_hours) * conventional_gen * 0.6,
        'gen_1': np.ones(n_hours) * conventional_gen * 0.4,
        'ren_0': np.ones(n_hours) * renewable_gen
    }
    
    # Storage (simplified)
    storage = {
        'battery_0': {
            'power': np.random.normal(0, 10, n_hours),
            'soc': 0.5 + 0.1 * np.sin(np.arange(n_hours))
        }
    }
    
    # Demand
    demand = np.ones(n_hours) * 1000
    
    return {
        'time': np.arange(n_hours),
        'price': prices,
        'generation': generation,
        'storage': storage,
        'demand': demand
    }

def create_test_system_state():
    """Create test system state for ML prediction"""
    from Power_Market_Model import Generator
    
    generators = [
        Generator(0, 2.0, 0.02, 3.0, 500),
        Generator(1, 1.8, 0.018, 3.5, 400)
    ]
    
    return {'generators': generators}

def plot_multi_market_results(results):
    """Plot multi-market simulation results"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price comparison
    ax = axes[0, 0]
    ax.plot(results['time'], results['prices']['energy'], 'b-', linewidth=2, label='Energy')
    ax.plot(results['time'], results['prices']['spinning_reserve'], 'r-', linewidth=2, label='Spinning Reserve')
    ax.plot(results['time'], results['prices']['regulation_up'], 'g-', linewidth=2, label='Regulation Up')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Price ($/MWh or $/MW-h)')
    ax.set_title('Multi-Market Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Revenue breakdown
    ax = axes[0, 1]
    units = list(results['total_revenue'].keys())
    energy_rev = [results['total_revenue'][unit]['energy'] for unit in units]
    ancillary_rev = [results['total_revenue'][unit]['ancillary'] for unit in units]
    
    width = 0.35
    x = np.arange(len(units))
    ax.bar(x - width/2, energy_rev, width, label='Energy', alpha=0.8)
    ax.bar(x + width/2, ancillary_rev, width, label='Ancillary', alpha=0.8)
    ax.set_xlabel('Units')
    ax.set_ylabel('Revenue ($)')
    ax.set_title('Revenue by Source')
    ax.set_xticks(x)
    ax.set_xticklabels(units, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Dispatch for first generator
    ax = axes[1, 0]
    if 'gen_0' in results['dispatch']:
        gen_dispatch = results['dispatch']['gen_0']
        ax.plot(results['time'], gen_dispatch['energy'], 'b-', linewidth=2, label='Energy')
        ax.plot(results['time'], gen_dispatch['spinning'], 'r-', linewidth=2, label='Spinning Reserve')
        ax.plot(results['time'], gen_dispatch['regulation_up'], 'g-', linewidth=2, label='Regulation Up')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Dispatch (MW)')
        ax.set_title('Generator 1 Dispatch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Storage SOC
    ax = axes[1, 1]
    if any('storage_' in key for key in results['dispatch'].keys()):
        for key, data in results['dispatch'].items():
            if key.startswith('storage_') and 'soc' in data:
                ax.plot(results['time'], data['soc'] * 100, linewidth=2, label=key)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('State of Charge (%)')
        ax.set_title('Storage State of Charge')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def run_comprehensive_comparison():
    """Compare original vs enhanced model performance"""
    print("\n" + "=" * 60)
    print("5. COMPREHENSIVE COMPARISON")
    print("=" * 60)
    
    # Run original model (simplified version)
    print("Running original model...")
    original_results = run_original_model_simulation()
    
    # Run enhanced model
    print("Running enhanced model...")
    enhanced_results = run_basic_enhanced_simulation()
    
    # Compare results
    print(f"\nPerformance Comparison:")
    print(f"{'Metric':<25} {'Original':<12} {'Enhanced':<12} {'Improvement'}")
    print("-" * 60)
    
    metrics = [
        ('Average Price ($/MWh)', np.mean(original_results['price']), np.mean(enhanced_results['price'])),
        ('Price Volatility ($/MWh)', np.std(original_results['price']), np.std(enhanced_results['price'])),
        ('Peak Price ($/MWh)', np.max(original_results['price']), np.max(enhanced_results['price'])),
    ]
    
    for metric_name, original_val, enhanced_val in metrics:
        if 'Volatility' in metric_name or 'Peak' in metric_name:
            improvement = f"{(original_val - enhanced_val)/original_val*100:+.1f}%"
        else:
            improvement = f"{(enhanced_val - original_val)/original_val*100:+.1f}%"
        
        print(f"{metric_name:<25} {original_val:<12.2f} {enhanced_val:<12.2f} {improvement}")
    
    return original_results, enhanced_results

def run_original_model_simulation():
    """Run simplified version of original model for comparison"""
    # Simplified version of original model
    params = MarketParameters()
    market = EnhancedPowerMarket(params)
    
    # Add only conventional generators (like original)
    market.add_generator(2.0, 0.02, 3.0, 500)
    market.add_generator(1.75, 0.0175, 4.0, 600) 
    market.add_generator(3.0, 0.025, 2.5, 400)
    market.add_generator(3.0, 0.025, 3.0, 450)
    
    # No renewables, storage, or demand response
    results = market.simulate()
    return results

def main():
    """Run comprehensive power market enhancement demonstration"""
    print("ENHANCED POWER MARKET SIMULATION SUITE")
    print("=" * 60)
    print("Demonstrating modern power market modeling capabilities")
    print("Based on research by Haugen et al., Tamrakar et al., and Liu et al.")
    print("=" * 60)
    
    if not modules_available:
        print("ERROR: Required modules not available. Please ensure all files are in the same directory.")
        return
    
    results = {}
    
    try:
        # 1. Basic Enhanced Simulation
        results['basic'] = run_basic_enhanced_simulation()
        
        # 2. Multi-Market Simulation  
        results['multi_market'] = run_multi_market_simulation()
        
        # 3. Uncertainty Analysis
        results['uncertainty'] = run_uncertainty_analysis()
        
        # 4. ML Stability Prediction
        results['ml_training'], results['ml_monitor'] = run_ml_stability_prediction()
        
        # 5. Comprehensive Comparison
        results['original'], results['enhanced'] = run_comprehensive_comparison()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All enhancements successfully demonstrated:")
        print("[OK] Energy Storage Systems (BESS)")
        print("[OK] Renewable Energy Integration")
        print("[OK] Demand Response")
        print("[OK] Enhanced Stability Analysis")
        print("[OK] Multi-Market Co-optimization")
        print("[OK] Uncertainty Quantification")
        print("[OK] Machine Learning Stability Prediction")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    return results

if __name__ == "__main__":
    # Check for required packages
    required_packages = ['numpy', 'matplotlib', 'scipy']
    optional_packages = ['cvxpy', 'sklearn', 'pandas', 'joblib']
    
    print("Checking package availability...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package} (REQUIRED)")
    
    for package in optional_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[OPTIONAL] {package} (optional - some features may be limited)")
    
    print()
    
    # Run the comprehensive demonstration
    demo_results = main()