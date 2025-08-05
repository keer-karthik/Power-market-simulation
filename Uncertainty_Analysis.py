import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy import stats
import concurrent.futures
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UncertaintyParameters:
    """Parameters defining uncertainty ranges"""
    demand_uncertainty: float = 0.05        # ±5% demand uncertainty
    renewable_uncertainty: float = 0.20     # ±20% renewable forecast error
    outage_probability: float = 0.001       # 0.1% hourly outage probability per unit
    fuel_price_volatility: float = 0.10     # ±10% fuel price uncertainty
    load_correlation: float = 0.3           # Spatial load correlation
    wind_correlation: float = 0.6           # Wind farm correlation
    solar_correlation: float = 0.8          # Solar farm correlation

class ScenarioGenerator:
    """Generate stochastic scenarios for Monte Carlo analysis"""
    
    def __init__(self, uncertainty_params: UncertaintyParameters = None):
        self.params = uncertainty_params or UncertaintyParameters()
        self.random_state = np.random.RandomState(42)  # For reproducibility
    
    def generate_demand_scenarios(self, base_demand: np.ndarray, n_scenarios: int) -> np.ndarray:
        """Generate correlated demand scenarios"""
        n_hours = len(base_demand)
        
        # Generate autocorrelated demand variations
        scenarios = np.zeros((n_scenarios, n_hours))
        
        for i in range(n_scenarios):
            # AR(1) process for temporal correlation
            noise = self.random_state.normal(0, self.params.demand_uncertainty, n_hours)
            demand_variation = np.zeros(n_hours)
            demand_variation[0] = noise[0]
            
            # Autocorrelation parameter (higher = more persistence)
            rho = 0.7
            for t in range(1, n_hours):
                demand_variation[t] = rho * demand_variation[t-1] + np.sqrt(1-rho**2) * noise[t]
            
            scenarios[i, :] = base_demand * (1 + demand_variation)
        
        return np.maximum(scenarios, base_demand * 0.1)  # Minimum 10% of base demand
    
    def generate_renewable_scenarios(self, base_profiles: Dict[str, np.ndarray], 
                                   n_scenarios: int) -> Dict[str, np.ndarray]:
        """Generate correlated renewable scenarios"""
        renewable_scenarios = {}
        
        for resource_type, base_profile in base_profiles.items():
            n_hours = len(base_profile)
            scenarios = np.zeros((n_scenarios, n_hours))
            
            if resource_type == 'wind':
                uncertainty = self.params.renewable_uncertainty
                correlation = self.params.wind_correlation
            elif resource_type == 'solar':
                uncertainty = self.params.renewable_uncertainty * 0.7  # Solar more predictable
                correlation = self.params.solar_correlation
            else:
                uncertainty = self.params.renewable_uncertainty
                correlation = 0.5
            
            for i in range(n_scenarios):
                # Beta distribution for bounded renewable output (0-1 capacity factor)
                alpha = 2.0
                beta = 2.0
                
                # Generate correlated forecast errors
                noise = self.random_state.normal(0, uncertainty, n_hours)
                errors = np.zeros(n_hours)
                errors[0] = noise[0]
                
                for t in range(1, n_hours):
                    errors[t] = correlation * errors[t-1] + np.sqrt(1-correlation**2) * noise[t]
                
                # Apply errors to base profile
                scenarios[i, :] = np.clip(base_profile * (1 + errors), 0, base_profile * 1.2)
            
            renewable_scenarios[resource_type] = scenarios
        
        return renewable_scenarios
    
    def generate_outage_scenarios(self, units: List[Any], n_scenarios: int, 
                                n_hours: int) -> List[np.ndarray]:
        """Generate random outage scenarios"""
        outage_scenarios = []
        
        for scenario in range(n_scenarios):
            outages = np.ones((len(units), n_hours))  # 1 = available, 0 = outage
            
            for unit_idx, unit in enumerate(units):
                for hour in range(n_hours):
                    # Random outage based on probability
                    if self.random_state.random() < self.params.outage_probability:
                        # Outage duration: geometric distribution (1-24 hours)
                        duration = min(24, self.random_state.geometric(0.3))
                        end_hour = min(n_hours, hour + duration)
                        outages[unit_idx, hour:end_hour] = 0
            
            outage_scenarios.append(outages)
        
        return outage_scenarios
    
    def generate_fuel_price_scenarios(self, base_prices: np.ndarray, 
                                    n_scenarios: int) -> np.ndarray:
        """Generate fuel price scenarios with mean reversion"""
        n_fuels = len(base_prices)
        scenarios = np.zeros((n_scenarios, n_fuels))
        
        for i in range(n_scenarios):
            # Log-normal distribution with mean reversion
            for fuel_idx in range(n_fuels):
                price_shock = self.random_state.normal(0, self.params.fuel_price_volatility)
                scenarios[i, fuel_idx] = base_prices[fuel_idx] * np.exp(price_shock)
        
        return scenarios

class MonteCarloAnalyzer:
    """Monte Carlo analysis for power market uncertainty"""
    
    def __init__(self, base_market_model, uncertainty_params: UncertaintyParameters = None):
        self.base_model = base_market_model
        self.uncertainty_params = uncertainty_params or UncertaintyParameters()
        self.scenario_generator = ScenarioGenerator(uncertainty_params)
    
    def run_monte_carlo(self, n_scenarios: int = 100, parallel: bool = True, 
                       n_workers: int = 4) -> Dict:
        """Run Monte Carlo simulation"""
        
        print(f"Running Monte Carlo analysis with {n_scenarios} scenarios...")
        
        # Generate base scenarios
        print("Generating stochastic scenarios...")
        scenarios = self._generate_all_scenarios(n_scenarios)
        
        # Run simulations
        if parallel and n_workers > 1:
            results = self._run_parallel_simulations(scenarios, n_workers)
        else:
            results = self._run_sequential_simulations(scenarios)
        
        # Analyze results
        print("Analyzing results...")
        analysis = self._analyze_results(results)
        
        return {
            'scenarios': scenarios,
            'results': results,
            'analysis': analysis,
            'n_scenarios': n_scenarios
        }
    
    def _generate_all_scenarios(self, n_scenarios: int) -> Dict:
        """Generate all stochastic scenarios"""
        n_hours = self.base_model.params.total_steps
        
        # Base demand profile (simplified daily pattern)
        base_demand = 1000 * (0.8 + 0.3 * np.sin(2 * np.pi * np.arange(n_hours) / (24 * 60)))
        
        # Base renewable profiles
        time_hours = np.arange(n_hours) / 60.0  # Convert to hours
        base_renewables = {
            'solar': 200 * np.maximum(0, np.sin(np.pi * (time_hours - 6) / 12)),
            'wind': 150 * (0.4 + 0.3 * np.sin(2 * np.pi * time_hours / 24)),
            'hydro': 300 * (0.8 + 0.2 * np.sin(2 * np.pi * time_hours / 24)),  # Hydro with daily pattern
            'nuclear': 800 * np.ones(n_hours) * 0.95  # Nuclear baseload at 95% capacity factor
        }
        
        # Generate scenarios
        demand_scenarios = self.scenario_generator.generate_demand_scenarios(base_demand, n_scenarios)
        renewable_scenarios = self.scenario_generator.generate_renewable_scenarios(base_renewables, n_scenarios)
        
        # Generator outages
        outage_scenarios = self.scenario_generator.generate_outage_scenarios(
            self.base_model.generators, n_scenarios, n_hours)
        
        # Fuel price scenarios
        base_fuel_prices = np.array([gen.b for gen in self.base_model.generators])
        fuel_price_scenarios = self.scenario_generator.generate_fuel_price_scenarios(
            base_fuel_prices, n_scenarios)
        
        return {
            'demand': demand_scenarios,
            'renewables': renewable_scenarios,
            'outages': outage_scenarios,
            'fuel_prices': fuel_price_scenarios
        }
    
    def _run_sequential_simulations(self, scenarios: Dict) -> List[Dict]:
        """Run simulations sequentially"""
        results = []
        n_scenarios = scenarios['demand'].shape[0]
        
        for i in range(n_scenarios):
            if i % 10 == 0:
                print(f"  Scenario {i+1}/{n_scenarios}")
            
            result = self._run_single_scenario(scenarios, i)
            results.append(result)
        
        return results
    
    def _run_parallel_simulations(self, scenarios: Dict, n_workers: int) -> List[Dict]:
        """Run simulations in parallel"""
        n_scenarios = scenarios['demand'].shape[0]
        scenario_indices = list(range(n_scenarios))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._run_single_scenario_wrapper, scenarios, i) 
                      for i in scenario_indices]
            
            results = []
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if i % 10 == 0:
                    print(f"  Completed {i+1}/{n_scenarios} scenarios")
                results.append(future.result())
        
        return results
    
    def _run_single_scenario_wrapper(self, scenarios: Dict, scenario_idx: int) -> Dict:
        """Wrapper for single scenario simulation (for multiprocessing)"""
        return self._run_single_scenario(scenarios, scenario_idx)
    
    def _run_single_scenario(self, scenarios: Dict, scenario_idx: int) -> Dict:
        """Run a single scenario simulation"""
        try:
            # Create a copy of the base model
            model_copy = self._create_model_copy()
            
            # Apply scenario parameters
            self._apply_scenario(model_copy, scenarios, scenario_idx)
            
            # Run simulation
            result = model_copy.simulate()
            
            # Extract key metrics
            metrics = self._extract_metrics(result)
            metrics['scenario_idx'] = scenario_idx
            metrics['success'] = True
            
            return metrics
            
        except Exception as e:
            return {
                'scenario_idx': scenario_idx,
                'success': False,
                'error': str(e),
                'price_mean': np.nan,
                'price_std': np.nan,
                'total_cost': np.nan,
                'renewable_curtailment': np.nan,
                'reserve_shortage': np.nan
            }
    
    def _create_model_copy(self):
        """Create a copy of the base model"""
        # Simplified copy - in practice, you'd need a proper deep copy
        from Power_Market_Model import EnhancedPowerMarket, MarketParameters
        
        model_copy = EnhancedPowerMarket(self.base_model.params)
        
        # Copy generators
        for gen in self.base_model.generators:
            model_copy.add_generator(gen.b, gen.c, gen.A, gen.max_capacity, 
                                   gen.inertia_H, gen.damping_D, gen.generator_type)
        
        # Copy renewables
        for ren in self.base_model.renewables:
            model_copy.add_renewable(ren.capacity, ren.type)
        
        # Copy storage
        for storage in self.base_model.storage:
            model_copy.add_battery_storage(storage.capacity, storage.max_power, storage.efficiency)
        
        # Copy demand response
        if self.base_model.demand_response:
            dr = self.base_model.demand_response
            model_copy.set_demand_response(dr.baseline_demand, dr.max_reduction, dr.elasticity)
        
        return model_copy
    
    def _apply_scenario(self, model, scenarios: Dict, scenario_idx: int):
        """Apply scenario-specific parameters to the model"""
        # Apply fuel price changes
        fuel_prices = scenarios['fuel_prices'][scenario_idx]
        for i, gen in enumerate(model.generators):
            if i < len(fuel_prices):
                gen.b = fuel_prices[i]
        
        # Apply generator outages
        outages = scenarios['outages'][scenario_idx]
        for i, gen in enumerate(model.generators):
            if i < outages.shape[0]:
                # If outage occurred, reduce capacity
                availability = np.mean(outages[i, :])  # Average availability
                gen.max_capacity *= availability
    
    def _extract_metrics(self, result: Dict) -> Dict:
        """Extract key metrics from simulation result"""
        prices = result['price']
        
        metrics = {
            'price_mean': np.mean(prices),
            'price_std': np.std(prices),
            'price_max': np.max(prices),
            'price_min': np.min(prices),
            'total_generation': np.sum([np.sum(gen_data) for gen_data in result['generation'].values()]),
            'renewable_generation': np.sum([np.sum(gen_data) for key, gen_data in result['generation'].items() 
                                          if key.startswith('ren_')]),
            'storage_cycles': self._calculate_storage_cycles(result.get('storage', {})),
            'price_volatility': self._calculate_price_volatility(prices),
            'total_cost': self._calculate_total_cost(result)
        }
        
        return metrics
    
    def _calculate_storage_cycles(self, storage_data: Dict) -> float:
        """Calculate equivalent storage cycles"""
        total_cycles = 0
        for storage_key, data in storage_data.items():
            if 'soc' in data:
                soc_changes = np.abs(np.diff(data['soc']))
                total_cycles += np.sum(soc_changes) / 2  # Full cycle = 100% SOC change
        return total_cycles
    
    def _calculate_price_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility metric"""
        if len(prices) < 2:
            return 0.0
        price_returns = np.diff(np.log(np.maximum(prices, 1e-6)))
        return np.std(price_returns) * np.sqrt(len(prices))  # Annualized volatility
    
    def _calculate_total_cost(self, result: Dict) -> float:
        """Calculate total system cost"""
        # Simplified cost calculation
        total_cost = 0
        for gen_key, gen_data in result['generation'].items():
            if gen_key.startswith('gen_'):
                gen_idx = int(gen_key.split('_')[1])
                if gen_idx < len(self.base_model.generators):
                    gen = self.base_model.generators[gen_idx]
                    total_cost += np.sum(gen_data * gen.b)  # Simplified linear cost
        return total_cost
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze Monte Carlo results"""
        successful_results = [r for r in results if r.get('success', False)]
        n_successful = len(successful_results)
        n_total = len(results)
        
        if n_successful == 0:
            return {'error': 'No successful scenarios'}
        
        # Extract metrics
        metrics = {}
        for key in ['price_mean', 'price_std', 'price_max', 'total_generation', 
                   'renewable_generation', 'price_volatility', 'total_cost']:
            values = [r[key] for r in successful_results if not np.isnan(r.get(key, np.nan))]
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'percentiles': {
                        '5th': np.percentile(values, 5),
                        '25th': np.percentile(values, 25),
                        '50th': np.percentile(values, 50),
                        '75th': np.percentile(values, 75),
                        '95th': np.percentile(values, 95)
                    }
                }
        
        # Risk metrics
        price_means = [r['price_mean'] for r in successful_results]
        total_costs = [r['total_cost'] for r in successful_results]
        
        analysis = {
            'success_rate': n_successful / n_total,
            'metrics': metrics,
            'risk_metrics': {
                'price_var': np.percentile(price_means, 95) - np.percentile(price_means, 5),  # Value at Risk
                'cost_var': np.percentile(total_costs, 95) - np.percentile(total_costs, 5),
                'extreme_price_prob': np.mean([p > 100 for p in price_means]),  # Probability of extreme prices
            }
        }
        
        return analysis
    
    def plot_uncertainty_analysis(self, mc_results: Dict):
        """Plot Monte Carlo analysis results"""
        results = mc_results['results']
        analysis = mc_results['analysis']
        
        if 'error' in analysis:
            print(f"Cannot plot: {analysis['error']}")
            return
        
        # Extract successful results
        successful_results = [r for r in results if r.get('success', False)]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Price distribution
        ax = axes[0, 0]
        price_means = [r['price_mean'] for r in successful_results]
        ax.hist(price_means, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(price_means), color='red', linestyle='--', label=f'Mean: ${np.mean(price_means):.1f}')
        ax.set_xlabel('Mean Price ($/MWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cost distribution
        ax = axes[0, 1]
        total_costs = [r['total_cost'] for r in successful_results]
        ax.hist(total_costs, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax.axvline(np.mean(total_costs), color='red', linestyle='--', 
                  label=f'Mean: ${np.mean(total_costs):.0f}')
        ax.set_xlabel('Total Cost ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Total Cost Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Renewable generation
        ax = axes[0, 2]
        renewable_gen = [r['renewable_generation'] for r in successful_results]
        ax.hist(renewable_gen, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(np.mean(renewable_gen), color='red', linestyle='--',
                  label=f'Mean: {np.mean(renewable_gen):.0f} MWh')
        ax.set_xlabel('Renewable Generation (MWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Renewable Generation Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Price volatility
        ax = axes[1, 0]
        price_volatility = [r['price_volatility'] for r in successful_results]
        ax.hist(price_volatility, bins=30, alpha=0.7, edgecolor='black', color='purple')
        ax.axvline(np.mean(price_volatility), color='red', linestyle='--',
                  label=f'Mean: {np.mean(price_volatility):.3f}')
        ax.set_xlabel('Price Volatility')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Volatility Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Risk metrics
        ax = axes[1, 1]
        percentiles = ['5th', '25th', '50th', '75th', '95th']
        price_percentiles = [analysis['metrics']['price_mean']['percentiles'][p] for p in percentiles]
        ax.plot(percentiles, price_percentiles, 'o-', linewidth=2, markersize=8)
        ax.set_ylabel('Price ($/MWh)')
        ax.set_title('Price Percentiles')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Monte Carlo Analysis Summary
        
        Scenarios: {mc_results['n_scenarios']}
        Success Rate: {analysis['success_rate']:.1%}
        
        Price Statistics:
        Mean: ${analysis['metrics']['price_mean']['mean']:.2f}/MWh
        Std Dev: ${analysis['metrics']['price_mean']['std']:.2f}/MWh
        5th-95th percentile range: ${analysis['risk_metrics']['price_var']:.2f}/MWh
        
        Cost Statistics:
        Mean: ${analysis['metrics']['total_cost']['mean']:.0f}
        Std Dev: ${analysis['metrics']['total_cost']['std']:.0f}
        
        Extreme Price Risk:
        P(Price > $100/MWh): {analysis['risk_metrics']['extreme_price_prob']:.1%}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    from Power_Market_Model import EnhancedPowerMarket, MarketParameters
    
    # Create a test market
    params = MarketParameters(simulation_hours=4, time_step_minutes=15)  # Shorter for testing
    market = EnhancedPowerMarket(params)
    
    # Add some units
    market.add_generator(2.0, 0.02, 3.0, 500)
    market.add_generator(1.75, 0.0175, 4.0, 600)
    market.add_renewable(200, 'solar')
    market.add_battery_storage(100, 50)
    
    # Run Monte Carlo analysis
    uncertainty_params = UncertaintyParameters(
        demand_uncertainty=0.05,
        renewable_uncertainty=0.15,
        outage_probability=0.001
    )
    
    mc_analyzer = MonteCarloAnalyzer(market, uncertainty_params)
    
    print("Running small-scale Monte Carlo test...")
    mc_results = mc_analyzer.run_monte_carlo(n_scenarios=20, parallel=False)
    
    print("\n=== Monte Carlo Results ===")
    if 'error' not in mc_results['analysis']:
        analysis = mc_results['analysis']
        print(f"Success rate: {analysis['success_rate']:.1%}")
        if 'price_mean' in analysis['metrics']:
            price_stats = analysis['metrics']['price_mean']
            print(f"Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}/MWh")
            print(f"Price mean: ${price_stats['mean']:.2f} ± ${price_stats['std']:.2f}/MWh")
        
        # Plot results
        mc_analyzer.plot_uncertainty_analysis(mc_results)
    else:
        print(f"Analysis failed: {mc_results['analysis']['error']}")