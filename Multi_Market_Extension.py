import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

@dataclass
class ReserveRequirement:
    """System reserve requirements"""
    spinning_reserve: float = 0.0      # MW
    regulation_up: float = 0.0         # MW  
    regulation_down: float = 0.0       # MW
    load_following: float = 0.0        # MW
    
class AncillaryService:
    """Base class for ancillary services"""
    def __init__(self, service_type: str, price: float = 0.0):
        self.service_type = service_type
        self.price = price              # $/MW for capacity, $/MWh for energy
        self.requirements = 0.0         # MW requirement
        self.provision = {}            # {provider_id: capacity_MW}

class ReserveMarket:
    """Reserve market for spinning reserves and regulation"""
    def __init__(self):
        self.spinning_reserve = AncillaryService("spinning_reserve")
        self.regulation_up = AncillaryService("regulation_up") 
        self.regulation_down = AncillaryService("regulation_down")
        self.load_following = AncillaryService("load_following")
        
    def set_requirements(self, requirements: ReserveRequirement):
        """Set system reserve requirements"""
        self.spinning_reserve.requirements = requirements.spinning_reserve
        self.regulation_up.requirements = requirements.regulation_up
        self.regulation_down.requirements = requirements.regulation_down
        self.load_following.requirements = requirements.load_following

class EnhancedGenerator:
    """Generator with capability to provide multiple services"""
    def __init__(self, gen_id: int, linear_cost: float, quadratic_cost: float, 
                 adjustment_param: float, max_capacity: float = 1000.0,
                 ramp_rate: float = 50.0, inertia_H: float = 5.0, 
                 damping_D: float = 1.0, generator_type: str = 'thermal'):
        self.id = gen_id
        self.b = linear_cost          # Linear cost coefficient [$/MWh]
        self.c = quadratic_cost       # Quadratic cost coefficient [$/MW²h]
        self.A = adjustment_param     # Dynamic adjustment parameter
        self.max_capacity = max_capacity
        self.ramp_rate = ramp_rate    # MW/min ramping capability
        self.output = 0.0            # Current output [MW]
        
        # Physical parameters for stability analysis
        self.inertia_H = inertia_H    # Inertia constant [s]
        self.damping_D = damping_D    # Damping coefficient
        self.generator_type = generator_type
        self.M = 2 * inertia_H * max_capacity / (100.0 * (2 * np.pi * 60)**2)  # Inertia coefficient
        
        # Reserve capabilities and costs
        self.spinning_reserve_cost = linear_cost * 0.1      # 10% of energy cost
        self.regulation_cost = linear_cost * 0.15           # 15% of energy cost
        self.load_following_cost = linear_cost * 0.05       # 5% of energy cost
        
        # Current reserve commitments
        self.spinning_reserve_commitment = 0.0
        self.regulation_up_commitment = 0.0
        self.regulation_down_commitment = 0.0
        self.load_following_commitment = 0.0
        
    @property
    def available_capacity(self) -> float:
        """Available capacity for energy after reserve commitments"""
        return max(0, self.max_capacity - self.total_reserve_commitment - self.output)
    
    @property 
    def total_reserve_commitment(self) -> float:
        """Total capacity committed to reserves"""
        return (self.spinning_reserve_commitment + 
                self.regulation_up_commitment +
                self.regulation_down_commitment +
                self.load_following_commitment)
    
    def marginal_cost(self, output: float = None) -> float:
        """Calculate marginal cost at given output"""
        if output is None:
            output = self.output
        return self.b + self.c * output
    
    def can_provide_reserve(self, reserve_type: str, amount: float) -> bool:
        """Check if generator can provide additional reserve"""
        if reserve_type == "spinning":
            return (self.total_reserve_commitment + amount) <= (self.max_capacity - self.output)
        elif reserve_type in ["regulation_up", "load_following"]:
            # Need headroom above current output
            return (self.output + amount + self.total_reserve_commitment) <= self.max_capacity
        elif reserve_type == "regulation_down":
            # Need ability to reduce output
            return (self.output - amount) >= 0
        return False

class EnhancedBatteryStorage:
    """Battery storage with ancillary services capability"""
    def __init__(self, storage_id: int, capacity_mwh: float, power_mw: float, 
                 efficiency: float = 0.9):
        self.id = storage_id
        self.capacity = capacity_mwh
        self.max_power = power_mw
        self.efficiency = efficiency
        self.soc = 0.5
        self.power_output = 0.0
        
        # Fast response capabilities for ancillary services
        self.regulation_cost = 5.0      # $/MW-h for regulation
        self.frequency_response_cost = 3.0  # $/MW-h for frequency response
        
        # Service commitments
        self.regulation_up_commitment = 0.0
        self.regulation_down_commitment = 0.0
        self.frequency_response_commitment = 0.0
        
    @property
    def available_power_up(self) -> float:
        """Available power for discharge/regulation up"""
        return max(0, self.max_power - self.power_output - self.regulation_up_commitment)
    
    @property 
    def available_power_down(self) -> float:
        """Available power for charge/regulation down"""
        return max(0, self.max_power + self.power_output - self.regulation_down_commitment)

class MultiMarketOptimizer:
    """Co-optimize energy and ancillary services markets"""
    
    def __init__(self):
        self.energy_price = 0.0
        self.reserve_prices = {}
        
    def solve_joint_dispatch(self, generators: List[EnhancedGenerator], 
                           storage: List[EnhancedBatteryStorage],
                           energy_demand: float, 
                           reserve_requirements: ReserveRequirement) -> Dict:
        """
        Solve joint energy and reserve dispatch using CVXPY
        Minimize total cost while meeting energy and reserve requirements
        """
        
        if not CVXPY_AVAILABLE:
            raise ImportError("CVXPY is required for multi-market optimization. Install with: pip install cvxpy")
        
        n_gen = len(generators)
        n_storage = len(storage)
        
        # Decision variables
        # Generators
        gen_energy = cp.Variable(n_gen, nonneg=True)
        gen_spinning = cp.Variable(n_gen, nonneg=True)
        gen_reg_up = cp.Variable(n_gen, nonneg=True)
        gen_reg_down = cp.Variable(n_gen, nonneg=True)
        
        # Storage
        storage_energy = cp.Variable(n_storage)  # Can be negative (charging)
        storage_reg_up = cp.Variable(n_storage, nonneg=True)
        storage_reg_down = cp.Variable(n_storage, nonneg=True)
        
        # Objective: minimize total cost
        cost = 0
        
        # Generator costs
        for i, gen in enumerate(generators):
            # Energy cost (quadratic)
            cost += gen.b * gen_energy[i] + gen.c * cp.power(gen_energy[i], 2)
            # Reserve costs (linear)
            cost += gen.spinning_reserve_cost * gen_spinning[i]
            cost += gen.regulation_cost * (gen_reg_up[i] + gen_reg_down[i])
        
        # Storage costs
        for i, stor in enumerate(storage):
            # Opportunity cost for charging (simplified)
            cost += 0.01 * cp.abs(storage_energy[i])  # Small charging cost
            cost += stor.regulation_cost * (storage_reg_up[i] + storage_reg_down[i])
        
        # Constraints
        constraints = []
        
        # Energy balance
        total_gen_energy = cp.sum(gen_energy) + cp.sum(storage_energy)
        constraints.append(total_gen_energy == energy_demand)
        
        # Reserve requirements
        constraints.append(cp.sum(gen_spinning) >= reserve_requirements.spinning_reserve)
        constraints.append(cp.sum(gen_reg_up) + cp.sum(storage_reg_up) >= reserve_requirements.regulation_up)
        constraints.append(cp.sum(gen_reg_down) + cp.sum(storage_reg_down) >= reserve_requirements.regulation_down)
        
        # Generator constraints
        for i, gen in enumerate(generators):
            # Capacity constraints
            constraints.append(gen_energy[i] + gen_spinning[i] + gen_reg_up[i] <= gen.max_capacity)
            constraints.append(gen_energy[i] - gen_reg_down[i] >= 0)
            
            # Reserve capability constraints
            constraints.append(gen_spinning[i] <= gen.max_capacity - gen_energy[i])
            constraints.append(gen_reg_up[i] <= gen.max_capacity - gen_energy[i])
        
        # Storage constraints
        for i, stor in enumerate(storage):
            # Power constraints
            constraints.append(storage_energy[i] + storage_reg_up[i] <= stor.max_power)
            constraints.append(-storage_energy[i] + storage_reg_down[i] <= stor.max_power)
            constraints.append(storage_energy[i] >= -stor.max_power)
            constraints.append(storage_energy[i] <= stor.max_power)
            
            # Energy/SOC constraints (prevent overcharge/overdischarge)
            # Assume single time period dispatch - in reality would need multi-period optimization
            current_energy = stor.soc * stor.capacity  # Current energy stored
            
            # Charging constraint: cannot exceed 100% SOC
            max_charge_energy = stor.capacity - current_energy  # Energy to full capacity
            if max_charge_energy > 0:
                max_charge_power = max_charge_energy / 1.0  # Assume 1-hour time step
                constraints.append(-storage_energy[i] <= max_charge_power / stor.efficiency)
            else:
                constraints.append(storage_energy[i] >= 0)  # No charging if full
            
            # Discharging constraint: cannot go below 0% SOC
            max_discharge_energy = current_energy  # Energy available for discharge
            if max_discharge_energy > 0:
                max_discharge_power = max_discharge_energy * stor.efficiency  # Account for efficiency
                constraints.append(storage_energy[i] <= max_discharge_power)
            else:
                constraints.append(storage_energy[i] <= 0)  # No discharging if empty
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # Use OSQP solver which is reliable for this problem type
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=1000)
            
            if problem.status == cp.OPTIMAL:
                # Extract dual prices (marginal costs)
                energy_price = constraints[0].dual_value
                spinning_price = constraints[1].dual_value
                reg_up_price = constraints[2].dual_value
                reg_down_price = constraints[3].dual_value
                
                return {
                    'status': 'optimal',
                    'total_cost': problem.value,
                    'energy_price': abs(energy_price) if energy_price else 35.0,
                    'reserve_prices': {
                        'spinning': abs(spinning_price) if spinning_price else 10.0,
                        'regulation_up': abs(reg_up_price) if reg_up_price else 15.0,
                        'regulation_down': abs(reg_down_price) if reg_down_price else 15.0
                    },
                    'dispatch': {
                        'gen_energy': gen_energy.value,
                        'gen_spinning': gen_spinning.value,
                        'gen_reg_up': gen_reg_up.value,
                        'gen_reg_down': gen_reg_down.value,
                        'storage_energy': storage_energy.value,
                        'storage_reg_up': storage_reg_up.value,
                        'storage_reg_down': storage_reg_down.value
                    }
                }
            else:
                # Fallback to simple pricing if optimization fails
                return self._fallback_dispatch(generators, storage, energy_demand, reserve_requirements)
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            return self._fallback_dispatch(generators, storage, energy_demand, reserve_requirements)
    
    def _fallback_dispatch(self, generators: List[EnhancedGenerator], 
                          storage: List[EnhancedBatteryStorage],
                          energy_demand: float, 
                          reserve_requirements: ReserveRequirement) -> Dict:
        """Simple fallback dispatch when optimization fails"""
        
        # Simple merit order dispatch
        gen_costs = [(i, gen.marginal_cost()) for i, gen in enumerate(generators)]
        gen_costs.sort(key=lambda x: x[1])
        
        remaining_demand = energy_demand
        gen_dispatch = np.zeros(len(generators))
        
        for i, cost in gen_costs:
            if remaining_demand <= 0:
                break
            dispatch = min(remaining_demand, generators[i].max_capacity * 0.8)  # Reserve 20% for reserves
            gen_dispatch[i] = dispatch
            remaining_demand -= dispatch
        
        # Simple reserve allocation
        spinning_dispatch = np.zeros(len(generators))
        remaining_spinning = reserve_requirements.spinning_reserve
        
        for i, cost in gen_costs:
            if remaining_spinning <= 0:
                break
            available = generators[i].max_capacity - gen_dispatch[i]
            allocation = min(remaining_spinning, available * 0.5)
            spinning_dispatch[i] = allocation
            remaining_spinning -= allocation
        
        return {
            'status': 'fallback',
            'total_cost': sum(gen_dispatch[i] * generators[i].marginal_cost() for i in range(len(generators))),
            'energy_price': 35.0,  # Default price
            'reserve_prices': {
                'spinning': 10.0,
                'regulation_up': 15.0,
                'regulation_down': 15.0
            },
            'dispatch': {
                'gen_energy': gen_dispatch,
                'gen_spinning': spinning_dispatch,
                'gen_reg_up': np.zeros(len(generators)),
                'gen_reg_down': np.zeros(len(generators)),
                'storage_energy': np.zeros(len(storage)),
                'storage_reg_up': np.zeros(len(storage)),
                'storage_reg_down': np.zeros(len(storage))
            }
        }

class MultiMarketSimulation:
    """Extended power market with multiple services"""
    
    def __init__(self, market_params):
        self.params = market_params
        self.generators: List[EnhancedGenerator] = []
        self.storage: List[EnhancedBatteryStorage] = []
        self.reserve_market = ReserveMarket()
        self.optimizer = MultiMarketOptimizer()
        
        # Results storage
        self.time_array = np.linspace(0, self.params.simulation_hours, 
                                    self.params.total_steps + 1)
        self.price_history = {
            'energy': np.zeros(self.params.total_steps + 1),
            'spinning_reserve': np.zeros(self.params.total_steps + 1),
            'regulation_up': np.zeros(self.params.total_steps + 1),
            'regulation_down': np.zeros(self.params.total_steps + 1)
        }
        self.dispatch_history = {}
        self.revenue_history = {}
        
    def add_enhanced_generator(self, linear_cost: float, quadratic_cost: float, 
                             adjustment_param: float, max_capacity: float = 1000.0,
                             ramp_rate: float = 50.0, inertia_H: float = 5.0,
                             damping_D: float = 1.0, generator_type: str = 'thermal') -> int:
        """Add generator with ancillary service capability"""
        gen_id = len(self.generators)
        gen = EnhancedGenerator(gen_id, linear_cost, quadratic_cost, 
                              adjustment_param, max_capacity, ramp_rate,
                              inertia_H, damping_D, generator_type)
        self.generators.append(gen)
        
        # Initialize history tracking
        self.dispatch_history[f'gen_{gen_id}'] = {
            'energy': np.zeros(self.params.total_steps + 1),
            'spinning': np.zeros(self.params.total_steps + 1),
            'regulation_up': np.zeros(self.params.total_steps + 1),
            'regulation_down': np.zeros(self.params.total_steps + 1)
        }
        self.revenue_history[f'gen_{gen_id}'] = {
            'energy': np.zeros(self.params.total_steps + 1),
            'ancillary': np.zeros(self.params.total_steps + 1)
        }
        
        return gen_id
    
    def add_enhanced_storage(self, capacity_mwh: float, power_mw: float, 
                           efficiency: float = 0.9) -> int:
        """Add storage with ancillary service capability"""
        storage_id = len(self.storage)
        storage = EnhancedBatteryStorage(storage_id, capacity_mwh, power_mw, efficiency)
        self.storage.append(storage)
        
        self.dispatch_history[f'storage_{storage_id}'] = {
            'energy': np.zeros(self.params.total_steps + 1),
            'regulation_up': np.zeros(self.params.total_steps + 1),
            'regulation_down': np.zeros(self.params.total_steps + 1),
            'soc': np.zeros(self.params.total_steps + 1)
        }
        self.revenue_history[f'storage_{storage_id}'] = {
            'energy': np.zeros(self.params.total_steps + 1),
            'ancillary': np.zeros(self.params.total_steps + 1)
        }
        
        return storage_id
    
    def calculate_reserve_requirements(self, total_demand: float, 
                                     renewable_capacity: float = 0.0) -> ReserveRequirement:
        """Calculate dynamic reserve requirements"""
        # Spinning reserve: typically 3-5% of load
        spinning = total_demand * 0.03
        
        # Regulation reserves: scale with renewable penetration
        renewable_factor = min(1.0, renewable_capacity / total_demand)
        regulation_up = total_demand * (0.01 + 0.02 * renewable_factor)
        regulation_down = total_demand * (0.01 + 0.015 * renewable_factor)
        
        # Load following: for slower variations
        load_following = total_demand * 0.02
        
        return ReserveRequirement(
            spinning_reserve=spinning,
            regulation_up=regulation_up,
            regulation_down=regulation_down,
            load_following=load_following
        )
    
    def simulate_multi_market(self, base_demand: float = 1000.0, 
                            renewable_capacity: float = 200.0) -> Dict:
        """Run multi-market simulation"""
        
        print("=== Multi-Market Power System Simulation ===")
        print(f"Generators: {len(self.generators)}")
        print(f"Storage systems: {len(self.storage)}")
        print(f"Markets: Energy + Spinning Reserve + Regulation")
        print()
        
        for t in range(self.params.total_steps):
            time_hour = self.time_array[t]
            
            # Calculate time-varying demand (simplified daily pattern)
            demand_factor = 0.8 + 0.3 * np.sin(2 * np.pi * (time_hour - 6) / 24)
            current_demand = base_demand * demand_factor
            
            # Calculate reserve requirements
            reserve_req = self.calculate_reserve_requirements(current_demand, renewable_capacity)
            
            # Solve joint energy and reserve dispatch
            result = self.optimizer.solve_joint_dispatch(
                self.generators, self.storage, current_demand, reserve_req)
            
            # Store prices
            self.price_history['energy'][t] = result['energy_price']
            self.price_history['spinning_reserve'][t] = result['reserve_prices']['spinning']
            self.price_history['regulation_up'][t] = result['reserve_prices']['regulation_up']
            self.price_history['regulation_down'][t] = result['reserve_prices']['regulation_down']
            
            # Store dispatch results
            if result['status'] == 'optimal':
                dispatch = result['dispatch']
                
                # Generator dispatch
                for i, gen in enumerate(self.generators):
                    gen_key = f'gen_{i}'
                    if gen_key in self.dispatch_history:
                        self.dispatch_history[gen_key]['energy'][t] = dispatch['gen_energy'][i]
                        self.dispatch_history[gen_key]['spinning'][t] = dispatch['gen_spinning'][i]
                        self.dispatch_history[gen_key]['regulation_up'][t] = dispatch['gen_reg_up'][i]
                        self.dispatch_history[gen_key]['regulation_down'][t] = dispatch['gen_reg_down'][i]
                        
                        # Calculate revenues
                        energy_rev = dispatch['gen_energy'][i] * result['energy_price']
                        ancillary_rev = (dispatch['gen_spinning'][i] * result['reserve_prices']['spinning'] +
                                       dispatch['gen_reg_up'][i] * result['reserve_prices']['regulation_up'] +
                                       dispatch['gen_reg_down'][i] * result['reserve_prices']['regulation_down'])
                        
                        self.revenue_history[gen_key]['energy'][t] = energy_rev
                        self.revenue_history[gen_key]['ancillary'][t] = ancillary_rev
                
                # Storage dispatch
                for i, storage in enumerate(self.storage):
                    storage_key = f'storage_{i}'
                    if storage_key in self.dispatch_history:
                        self.dispatch_history[storage_key]['energy'][t] = dispatch['storage_energy'][i]
                        self.dispatch_history[storage_key]['regulation_up'][t] = dispatch['storage_reg_up'][i]
                        self.dispatch_history[storage_key]['regulation_down'][t] = dispatch['storage_reg_down'][i]
                        
                        # Update SOC (simplified)
                        energy_change = dispatch['storage_energy'][i] * self.params.dt
                        if energy_change > 0:  # Discharging
                            storage.soc -= energy_change / (storage.capacity * storage.efficiency)
                        else:  # Charging
                            storage.soc += abs(energy_change) * storage.efficiency / storage.capacity
                        
                        storage.soc = np.clip(storage.soc, 0.05, 0.95)  # Operating limits
                        self.dispatch_history[storage_key]['soc'][t] = storage.soc
                        
                        # Calculate storage revenues
                        energy_rev = dispatch['storage_energy'][i] * result['energy_price']
                        ancillary_rev = (dispatch['storage_reg_up'][i] * result['reserve_prices']['regulation_up'] +
                                       dispatch['storage_reg_down'][i] * result['reserve_prices']['regulation_down'])
                        
                        self.revenue_history[storage_key]['energy'][t] = energy_rev
                        self.revenue_history[storage_key]['ancillary'][t] = ancillary_rev
        
        return {
            'time': self.time_array,
            'prices': self.price_history,
            'dispatch': self.dispatch_history,
            'revenues': self.revenue_history,
            'total_revenue': self._calculate_total_revenues()
        }
    
    def _calculate_total_revenues(self) -> Dict:
        """Calculate total revenues by source"""
        total_revenues = {}
        
        for unit_key, revenues in self.revenue_history.items():
            total_energy = np.sum(revenues['energy'])
            total_ancillary = np.sum(revenues['ancillary'])
            total_revenues[unit_key] = {
                'energy': total_energy,
                'ancillary': total_ancillary,
                'total': total_energy + total_ancillary,
                'ancillary_share': total_ancillary / (total_energy + total_ancillary) if (total_energy + total_ancillary) > 0 else 0
            }
        
        return total_revenues

# Example usage
if __name__ == "__main__":
    from Power_Market_Model import MarketParameters
    
    # Test the multi-market system
    params = MarketParameters(simulation_hours=24, time_step_minutes=5)
    multi_market = MultiMarketSimulation(params)
    
    # Add generators
    multi_market.add_enhanced_generator(2.0, 0.02, 3.0, 500, 30)
    multi_market.add_enhanced_generator(1.75, 0.0175, 4.0, 600, 50)
    multi_market.add_enhanced_generator(3.0, 0.025, 2.5, 400, 20)
    
    # Add storage
    multi_market.add_enhanced_storage(100, 50, 0.9)
    multi_market.add_enhanced_storage(200, 100, 0.85)
    
    # Run simulation
    try:
        results = multi_market.simulate_multi_market()
        
        print("\n=== Multi-Market Results ===")
        print(f"Average energy price: ${np.mean(results['prices']['energy']):.2f}/MWh")
        print(f"Average spinning reserve price: ${np.mean(results['prices']['spinning_reserve']):.2f}/MW-h")
        print(f"Average regulation price: ${np.mean(results['prices']['regulation_up']):.2f}/MW-h")
        
        print("\nRevenue Summary:")
        for unit, rev in results['total_revenue'].items():
            print(f"{unit}: Energy=${rev['energy']:.0f}, Ancillary=${rev['ancillary']:.0f}, "
                  f"Total=${rev['total']:.0f} (Ancillary: {rev['ancillary_share']*100:.1f}%)")
            
    except ImportError:
        print("CVXPY not available - multi-market optimization requires: pip install cvxpy")
        print("Using simplified dispatch instead")