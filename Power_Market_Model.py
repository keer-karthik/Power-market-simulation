import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

@dataclass
class MarketParameters:
    """Market configuration parameters"""
    demand_intercept: float = 50.0      # e - demand curve intercept
    demand_slope: float = 0.02          # f - demand curve slope
    simulation_hours: int = 24
    time_step_minutes: float = 1.0
    
    @property
    def dt(self) -> float:
        """Time step in hours"""
        return self.time_step_minutes / 60.0
    
    @property
    def total_steps(self) -> int:
        """Total simulation steps"""
        return int(self.simulation_hours * 60 / self.time_step_minutes)

class Generator:
    """Base generator class"""
    def __init__(self, gen_id: int, linear_cost: float, quadratic_cost: float, 
                 adjustment_param: float, max_capacity: float = 1000.0):
        self.id = gen_id
        self.b = linear_cost          # Linear cost coefficient [$/MWh]
        self.c = quadratic_cost       # Quadratic cost coefficient [$/MW²h]
        self.A = adjustment_param     # Dynamic adjustment parameter
        self.max_capacity = max_capacity
        self.output = 0.0            # Current output [MW]
        
    def marginal_cost(self, output: float = None) -> float:
        """Calculate marginal cost at given output"""
        if output is None:
            output = self.output
        return self.b + self.c * output
    
    def update_output(self, price: float, dt: float) -> None:
        """Update generator output using perfect competition dynamics"""
        dq_dt = self.A * (price - self.marginal_cost())
        self.output = max(0, min(self.max_capacity, self.output + dt * dq_dt))

class RenewableGenerator:
    """Renewable energy generator with stochastic output"""
    def __init__(self, gen_id: int, capacity: float, renewable_type: str = 'solar'):
        self.id = gen_id
        self.capacity = capacity
        self.type = renewable_type
        self.output = 0.0
        self.forecast_error_std = 0.15  # 15% forecast uncertainty
        
    def generate_profile(self, time_hours: np.ndarray) -> np.ndarray:
        """Generate renewable output profile with uncertainty"""
        if self.type == 'solar':
            # Solar follows daily pattern
            base_profile = np.maximum(0, np.sin(np.pi * (time_hours - 6) / 12))
            uncertainty = np.random.normal(0, self.forecast_error_std, len(time_hours))
            profile = np.maximum(0, np.minimum(1.0, base_profile + uncertainty))
        elif self.type == 'wind':
            # Wind with more variability
            base_profile = 0.4 + 0.3 * np.sin(2 * np.pi * time_hours / 24) 
            uncertainty = np.random.normal(0, 0.25, len(time_hours))
            profile = np.maximum(0, np.minimum(1.0, base_profile + uncertainty))
        else:
            profile = np.ones(len(time_hours)) * 0.3  # Constant baseline
            
        return profile * self.capacity
    
    def update_output(self, time_hour: float) -> None:
        """Update renewable output based on time of day"""
        profile = self.generate_profile(np.array([time_hour]))
        self.output = profile[0]

class BatteryStorage:
    """Battery Energy Storage System"""
    def __init__(self, storage_id: int, capacity_mwh: float, power_mw: float, 
                 efficiency: float = 0.9):
        self.id = storage_id
        self.capacity = capacity_mwh    # Energy capacity [MWh]
        self.max_power = power_mw       # Power rating [MW]
        self.efficiency = efficiency    # Round-trip efficiency
        self.soc = 0.5                 # State of charge (0-1)
        self.power_output = 0.0        # Current power output [MW] (positive = discharge)
        self.cost_charge = 0.01        # Opportunity cost for charging [$/MWh]
        
    @property
    def energy_stored(self) -> float:
        """Current energy stored [MWh]"""
        return self.soc * self.capacity
    
    def can_charge(self, power: float, dt: float) -> bool:
        """Check if battery can charge at given power for time dt"""
        energy_to_add = power * dt * self.efficiency
        return (self.soc + energy_to_add / self.capacity) <= 1.0
    
    def can_discharge(self, power: float, dt: float) -> bool:
        """Check if battery can discharge at given power for time dt"""
        energy_to_remove = power * dt / self.efficiency
        return (self.soc - energy_to_remove / self.capacity) >= 0.0
    
    def update_storage(self, price: float, dt: float) -> None:
        """Update battery operation based on market price"""
        # Simple arbitrage strategy: charge when price is low, discharge when high
        # This could be enhanced with optimization
        
        if price < 30 and self.soc < 0.9:  # Charge when price is low
            charge_power = min(self.max_power, 
                             (0.9 - self.soc) * self.capacity / (dt * self.efficiency))
            if self.can_charge(charge_power, dt):
                self.power_output = -charge_power  # Negative = charging
                self.soc += charge_power * dt * self.efficiency / self.capacity
        elif price > 40 and self.soc > 0.1:  # Discharge when price is high
            discharge_power = min(self.max_power,
                                (self.soc - 0.1) * self.capacity * self.efficiency / dt)
            if self.can_discharge(discharge_power, dt):
                self.power_output = discharge_power  # Positive = discharging
                self.soc -= discharge_power * dt / (self.efficiency * self.capacity)
        else:
            self.power_output = 0.0

class DemandResponse:
    """Flexible demand that responds to price signals"""
    def __init__(self, baseline_demand: float, max_reduction: float, 
                 price_elasticity: float = -0.1):
        self.baseline_demand = baseline_demand
        self.max_reduction = max_reduction
        self.elasticity = price_elasticity
        self.current_demand = baseline_demand
        
    def calculate_response(self, price: float, baseline_price: float = 35.0) -> float:
        """Calculate demand response based on price"""
        price_ratio = price / baseline_price
        # Demand reduction = elasticity * (price_ratio - 1) * max_reduction
        reduction = -self.elasticity * (price_ratio - 1) * self.max_reduction
        reduction = np.clip(reduction, 0, self.max_reduction)
        
        self.current_demand = self.baseline_demand - reduction
        return self.current_demand

class PowerNetwork:
    """Power system network representation"""
    def __init__(self, n_buses: int):
        self.n_buses = n_buses
        self.adjacency_matrix = np.zeros((n_buses, n_buses))
        self.line_capacities = {}
        self.bus_voltages = np.ones(n_buses)  # Per unit voltages
        
    def add_line(self, from_bus: int, to_bus: int, capacity: float):
        """Add transmission line between buses"""
        self.adjacency_matrix[from_bus, to_bus] = 1
        self.adjacency_matrix[to_bus, from_bus] = 1
        self.line_capacities[(from_bus, to_bus)] = capacity
        self.line_capacities[(to_bus, from_bus)] = capacity
    
    def analyze_connectivity(self) -> float:
        """Analyze network connectivity using algebraic connectivity"""
        G = nx.from_numpy_array(self.adjacency_matrix)
        laplacian = nx.laplacian_matrix(G).astype(float)
        eigenvals, _ = eig(laplacian)
        # Algebraic connectivity (second smallest eigenvalue)
        eigenvals_real = np.sort(eigenvals.real)
        return eigenvals_real[1] if len(eigenvals_real) > 1 else 0.0

class StabilityAnalyzer:
    """Power system stability analysis"""
    def __init__(self):
        pass
    
    def small_signal_stability(self, generators: List[Generator], 
                             market_params: MarketParameters) -> Tuple[bool, np.ndarray]:
        """Analyze small-signal stability using linearized model"""
        n = len(generators)
        # System matrix for perfect competition dynamics
        A_matrix = np.zeros((n, n))
        
        for i, gen in enumerate(generators):
            # Diagonal elements: -Ai * ci (self-regulation)
            A_matrix[i, i] = -gen.A * gen.c
            # Off-diagonal elements: -Ai * f (market coupling)
            for j in range(n):
                if i != j:
                    A_matrix[i, j] = -gen.A * market_params.demand_slope
        
        eigenvals, _ = eig(A_matrix)
        stable = np.all(eigenvals.real < 0)
        
        return stable, eigenvals
    
    def convergence_analysis(self, generators: List[Generator]) -> Dict:
        """Analyze convergence properties"""
        time_constants = []
        eigenvalues = []
        
        for gen in generators:
            T = 1 / (gen.A * gen.c)  # Time constant
            eigenval = -1 / T        # Approximate eigenvalue
            time_constants.append(T)
            eigenvalues.append(eigenval)
        
        return {
            'time_constants': np.array(time_constants),
            'eigenvalues': np.array(eigenvalues),
            'settling_time': max(time_constants) * 4  # 4 time constants for 98% settling
        }

class EnhancedPowerMarket:
    """Enhanced power market simulation with multiple technologies"""
    
    def __init__(self, market_params: MarketParameters = None):
        self.params = market_params or MarketParameters()
        self.generators: List[Generator] = []
        self.renewables: List[RenewableGenerator] = []
        self.storage: List[BatteryStorage] = []
        self.demand_response: Optional[DemandResponse] = None
        self.network: Optional[PowerNetwork] = None
        self.stability_analyzer = StabilityAnalyzer()
        
        # Results storage
        self.time_array = np.linspace(0, self.params.simulation_hours, 
                                    self.params.total_steps + 1)
        self.price_history = np.zeros(self.params.total_steps + 1)
        self.demand_history = np.zeros(self.params.total_steps + 1)
        self.generation_history = {}
        self.storage_history = {}
        
    def add_generator(self, linear_cost: float, quadratic_cost: float, 
                     adjustment_param: float, max_capacity: float = 1000.0) -> int:
        """Add conventional generator"""
        gen_id = len(self.generators)
        gen = Generator(gen_id, linear_cost, quadratic_cost, adjustment_param, max_capacity)
        self.generators.append(gen)
        self.generation_history[f'gen_{gen_id}'] = np.zeros(self.params.total_steps + 1)
        return gen_id
    
    def add_renewable(self, capacity: float, renewable_type: str = 'solar') -> int:
        """Add renewable generator"""
        ren_id = len(self.renewables)
        ren = RenewableGenerator(ren_id, capacity, renewable_type)
        self.renewables.append(ren)
        self.generation_history[f'ren_{ren_id}'] = np.zeros(self.params.total_steps + 1)
        return ren_id
    
    def add_battery_storage(self, capacity_mwh: float, power_mw: float, 
                          efficiency: float = 0.9) -> int:
        """Add battery storage system"""
        storage_id = len(self.storage)
        battery = BatteryStorage(storage_id, capacity_mwh, power_mw, efficiency)
        self.storage.append(battery)
        self.storage_history[f'battery_{storage_id}'] = {
            'power': np.zeros(self.params.total_steps + 1),
            'soc': np.zeros(self.params.total_steps + 1)
        }
        return storage_id
    
    def set_demand_response(self, baseline_demand: float, max_reduction: float, 
                          elasticity: float = -0.1):
        """Set demand response parameters"""
        self.demand_response = DemandResponse(baseline_demand, max_reduction, elasticity)
    
    def calculate_total_supply(self, time_step: int) -> float:
        """Calculate total system supply"""
        total = 0.0
        
        # Conventional generation
        for gen in self.generators:
            total += gen.output
            
        # Renewable generation
        for ren in self.renewables:
            total += ren.output
            
        # Storage discharge (positive = discharge)
        for storage in self.storage:
            total += storage.power_output
            
        return total
    
    def calculate_demand(self, price: float) -> float:
        """Calculate system demand including demand response"""
        # Base demand (could be made time-varying)
        base_demand = self.params.demand_intercept / self.params.demand_slope
        
        if self.demand_response:
            demand = self.demand_response.calculate_response(price)
        else:
            demand = base_demand
            
        return demand
    
    def simulate(self) -> Dict:
        """Run the enhanced power market simulation"""
        print("=== Enhanced Power Market Simulation ===")
        print(f"Conventional generators: {len(self.generators)}")
        print(f"Renewable generators: {len(self.renewables)}")
        print(f"Storage systems: {len(self.storage)}")
        print(f"Demand response: {'Yes' if self.demand_response else 'No'}")
        print(f"Simulation time: {self.params.simulation_hours} hours")
        print(f"Time step: {self.params.time_step_minutes} minutes")
        print()
        
        # Generate renewable profiles
        renewable_profiles = {}
        for ren in self.renewables:
            renewable_profiles[ren.id] = ren.generate_profile(self.time_array)
        
        # Main simulation loop
        for t in range(self.params.total_steps):
            time_hour = self.time_array[t]
            
            # Update renewable outputs
            for i, ren in enumerate(self.renewables):
                ren.output = renewable_profiles[ren.id][t]
                self.generation_history[f'ren_{ren.id}'][t] = ren.output
            
            # Calculate current supply
            total_supply = self.calculate_total_supply(t)
            
            # Calculate market price using inverse demand function
            # p = e - f * (total_supply)
            price = self.params.demand_intercept - self.params.demand_slope * total_supply
            self.price_history[t] = price
            
            # Calculate demand including demand response
            demand = self.calculate_demand(price)
            self.demand_history[t] = demand
            
            # Update conventional generators
            for gen in self.generators:
                gen.update_output(price, self.params.dt)
                self.generation_history[f'gen_{gen.id}'][t] = gen.output
            
            # Update storage systems
            for storage in self.storage:
                storage.update_storage(price, self.params.dt)
                self.storage_history[f'battery_{storage.id}']['power'][t] = storage.power_output
                self.storage_history[f'battery_{storage.id}']['soc'][t] = storage.soc
        
        # Final values
        final_supply = self.calculate_total_supply(-1)
        final_price = self.params.demand_intercept - self.params.demand_slope * final_supply
        self.price_history[-1] = final_price
        
        for gen in self.generators:
            self.generation_history[f'gen_{gen.id}'][-1] = gen.output
        for i, ren in enumerate(self.renewables):
            self.generation_history[f'ren_{ren.id}'][-1] = ren.output
        for storage in self.storage:
            self.storage_history[f'battery_{storage.id}']['power'][-1] = storage.power_output
            self.storage_history[f'battery_{storage.id}']['soc'][-1] = storage.soc
        
        # Stability analysis
        if self.generators:
            stable, eigenvals = self.stability_analyzer.small_signal_stability(
                self.generators, self.params)
            convergence_info = self.stability_analyzer.convergence_analysis(self.generators)
        else:
            stable, eigenvals = True, np.array([])
            convergence_info = {}
        
        return {
            'time': self.time_array,
            'price': self.price_history,
            'demand': self.demand_history,
            'generation': self.generation_history,
            'storage': self.storage_history,
            'stability': {
                'stable': stable,
                'eigenvalues': eigenvals,
                'convergence': convergence_info
            },
            'parameters': self.params
        }
    
    def plot_results(self, results: Dict):
        """Plot comprehensive simulation results"""
        fig = plt.figure(figsize=(15, 12))
        
        # Price evolution
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(results['time'], results['price'], 'b-', linewidth=2, label='Market Price')
        plt.xlabel('Time (hours)')
        plt.ylabel('Price ($/MWh)')
        plt.title('Market Price Evolution')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Generation mix
        ax2 = plt.subplot(3, 2, 2)
        bottom = np.zeros(len(results['time']))
        colors = plt.cm.Set3(np.linspace(0, 1, len(results['generation'])))
        
        for i, (key, data) in enumerate(results['generation'].items()):
            plt.fill_between(results['time'], bottom, bottom + data, 
                           label=key, alpha=0.7, color=colors[i])
            bottom += data
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Generation (MW)')
        plt.title('Generation Mix')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Storage operation
        if results['storage']:
            ax3 = plt.subplot(3, 2, 3)
            for key, data in results['storage'].items():
                plt.plot(results['time'], data['power'], label=f'{key} Power', linewidth=2)
            plt.xlabel('Time (hours)')
            plt.ylabel('Storage Power (MW)')
            plt.title('Storage Operation (+ = Discharge, - = Charge)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Storage SOC
            ax4 = plt.subplot(3, 2, 4)
            for key, data in results['storage'].items():
                plt.plot(results['time'], data['soc'] * 100, label=f'{key} SOC', linewidth=2)
            plt.xlabel('Time (hours)')
            plt.ylabel('State of Charge (%)')
            plt.title('Storage State of Charge')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Supply-demand balance
        ax5 = plt.subplot(3, 2, 5)
        total_generation = sum(results['generation'].values())
        plt.plot(results['time'], total_generation, 'g-', linewidth=2, label='Total Supply')
        plt.plot(results['time'], results['demand'], 'r--', linewidth=2, label='Demand')
        plt.xlabel('Time (hours)')
        plt.ylabel('Power (MW)')
        plt.title('Supply-Demand Balance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Stability analysis
        ax6 = plt.subplot(3, 2, 6)
        if 'eigenvalues' in results['stability'] and len(results['stability']['eigenvalues']) > 0:
            eigenvals = results['stability']['eigenvalues']
            plt.scatter(eigenvals.real, eigenvals.imag, s=50, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.title(f"System Eigenvalues (Stable: {results['stability']['stable']})")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No conventional\ngenerators for\nstability analysis', 
                    ha='center', va='center', transform=ax6.transAxes)
            plt.title('Stability Analysis')
        
        plt.tight_layout()
        plt.show()

def create_test_system() -> EnhancedPowerMarket:
    """Create a test power system with mixed generation"""
    
    # Market parameters
    params = MarketParameters(
        demand_intercept=50.0,
        demand_slope=0.02,
        simulation_hours=24,
        time_step_minutes=1.0
    )
    
    market = EnhancedPowerMarket(params)
    
    # Add conventional generators (from original model)
    market.add_generator(linear_cost=2.0, quadratic_cost=0.02, adjustment_param=3.0, max_capacity=500)
    market.add_generator(linear_cost=1.75, quadratic_cost=0.0175, adjustment_param=4.0, max_capacity=600)
    market.add_generator(linear_cost=3.0, quadratic_cost=0.025, adjustment_param=2.5, max_capacity=400)
    market.add_generator(linear_cost=3.0, quadratic_cost=0.025, adjustment_param=3.0, max_capacity=450)
    
    # Add renewable generation
    market.add_renewable(capacity=200, renewable_type='solar')
    market.add_renewable(capacity=150, renewable_type='wind')
    
    # Add battery storage
    market.add_battery_storage(capacity_mwh=100, power_mw=50, efficiency=0.9)
    market.add_battery_storage(capacity_mwh=200, power_mw=100, efficiency=0.85)
    
    # Add demand response
    market.set_demand_response(baseline_demand=1200, max_reduction=200, elasticity=-0.15)
    
    return market

if __name__ == "__main__":
    # Create and run test system
    market = create_test_system()
    results = market.simulate()
    
    # Display results
    print("\n=== Simulation Results ===")
    print(f"Final market price: ${results['price'][-1]:.2f}/MWh")
    print(f"Average price: ${np.mean(results['price']):.2f}/MWh")
    print(f"Price volatility (std): ${np.std(results['price']):.2f}/MWh")
    
    if results['stability']['stable']:
        print("\nSystem stability: STABLE")
        if 'convergence' in results['stability'] and 'settling_time' in results['stability']['convergence']:
            print(f"Estimated settling time: {results['stability']['convergence']['settling_time']:.2f} hours")
    else:
        print("\nSystem stability: UNSTABLE")
    
    # Plot results
    market.plot_results(results)