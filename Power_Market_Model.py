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
                 adjustment_param: float, max_capacity: float  = 1000.0,
                 inertia_H: float = 5.0, damping_D: float = 1.0, 
                 generator_type: str = 'thermal'):
        self.id = gen_id
        self.b = linear_cost          # Linear cost coefficient [$/MWh]
        self.c = quadratic_cost       # Quadratic cost coefficient [$/MW²h]
        self.A = adjustment_param     # Dynamic adjustment parameter
        self.max_capacity = max_capacity
        self.output = 0.0            # Current output [MW]
        
        # Physical parameters for stability analysis
        self.inertia_H = inertia_H    # Inertia constant [s]
        self.damping_D = damping_D    # Damping coefficient
        self.generator_type = generator_type
        self.M = 2 * inertia_H * max_capacity / (100.0 * (2 * np.pi * 60)**2)  # Inertia coefficient
        
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
            # Solar follows daily pattern (peak at midday)
            base_profile = np.maximum(0, np.sin(np.pi * (time_hours - 6) / 12))
            uncertainty = np.random.normal(0, self.forecast_error_std, len(time_hours))
            profile = np.maximum(0, np.minimum(1.0, base_profile + uncertainty))
            
        elif self.type == 'wind':
            # Wind with more variability (higher at night)
            base_profile = 0.4 + 0.3 * np.sin(2 * np.pi * time_hours / 24) 
            uncertainty = np.random.normal(0, 0.25, len(time_hours))
            profile = np.maximum(0, np.minimum(1.0, base_profile + uncertainty))
            
        elif self.type == 'hydro':
            # Hydropower - dispatchable but weather dependent
            # Base high availability with seasonal and weather variations
            seasonal_factor = 0.8 + 0.2 * np.sin(2 * np.pi * (time_hours / 24 / 365) * 2)  # Seasonal variation
            daily_pattern = 0.9 + 0.1 * np.sin(2 * np.pi * time_hours / 24)  # Slight daily pattern
            base_profile = seasonal_factor * daily_pattern
            # Lower uncertainty than wind/solar due to water reservoir buffering
            uncertainty = np.random.normal(0, 0.05, len(time_hours))  # 5% uncertainty
            profile = np.maximum(0.3, np.minimum(1.0, base_profile + uncertainty))  # Min 30% availability
            
        elif self.type == 'nuclear':
            # Nuclear - baseload with very high capacity factor
            # Occasional planned outages and very rare forced outages
            base_profile = np.ones(len(time_hours)) * 0.95  # 95% capacity factor
            # Very low uncertainty but occasional outages
            outage_probability = 0.001  # 0.1% hourly outage probability
            random_outages = np.random.random(len(time_hours)) < outage_probability
            maintenance_pattern = np.sin(2 * np.pi * time_hours / (24 * 365)) < -0.99  # Annual maintenance
            outages = random_outages | maintenance_pattern
            profile = base_profile * (~outages)  # Zero output during outages
            # Small uncertainty when operating
            uncertainty = np.random.normal(0, 0.02, len(time_hours))  # 2% uncertainty
            profile = np.maximum(0, np.minimum(1.0, profile + uncertainty * (~outages)))
            
        else:
            # Default renewable profile
            profile = np.ones(len(time_hours)) * 0.3  # Constant baseline
            
        return profile * self.capacity
    
    def update_output(self, time_hour: float) -> None:
        """Update renewable output based on time of day"""
        profile = self.generate_profile(np.array([time_hour]))
        self.output = profile[0]

class BatteryStorage:
    """Battery Energy Storage System with C-rates, safety margins, and frequency regulation"""
    def __init__(self, storage_id: int, capacity_mwh: float, power_mw: float, 
                 efficiency: float = 0.9, c_rate_charge: float = 1.0, c_rate_discharge: float = 1.0,
                 safety_margins: bool = True, freq_regulation: bool = True):
        self.id = storage_id
        self.capacity = capacity_mwh    # Energy capacity [MWh]
        self.max_power = power_mw       # Power rating [MW]
        self.efficiency = efficiency    # Round-trip efficiency
        self.soc = 0.5                 # State of charge (0-1)
        self.power_output = 0.0        # Current power output [MW] (positive = discharge)
        self.cost_charge = 0.01        # Opportunity cost for charging [$/MWh]
        
        # C-rate parameters
        self.c_rate_charge = c_rate_charge      # Charging C-rate (1C = 1 hour to full)
        self.c_rate_discharge = c_rate_discharge # Discharging C-rate
        
        # Calculate C-rate limited power
        self.max_charge_power_c_rate = capacity_mwh * c_rate_charge    # MW from C-rate
        self.max_discharge_power_c_rate = capacity_mwh * c_rate_discharge  # MW from C-rate
        
        # Safety margins for battery health
        self.safety_margins = safety_margins
        self.min_soc_safe = 0.1 if safety_margins else 0.0    # 10% minimum for health
        self.max_soc_safe = 0.9 if safety_margins else 1.0    # 90% maximum for health
        
        # Frequency regulation capability
        self.freq_regulation = freq_regulation
        self.freq_response_power = min(power_mw * 0.1, 10)    # 10% of power or 10MW max
        self.target_frequency = 60.0    # Hz
        self.freq_deadband = 0.05       # ±0.05 Hz deadband
        
        # NEW: Price history tracking
        self.price_history = []
        self.max_history_length = 168  # 1 week of hourly data
        
        # NEW: Action tracking for minimum runtime constraints
        self.last_action = 'idle'  # 'charge', 'discharge', 'idle'
        self.action_duration = 0   # How long in current action (hours)
        self.min_action_time = 0.25  # Minimum 15 minutes in same action
        
        # Statistics for monitoring
        self.limit_events = {
            'overcharge_prevented': 0, 
            'overdischarge_prevented': 0,
            'c_rate_limited': 0,
            'safety_margin_limited': 0,
            'freq_regulation_events': 0
        }
        self.max_soc_reached = 0.5
        self.min_soc_reached = 0.5
        
    @property
    def energy_stored(self) -> float:
        """Current energy stored [MWh]"""
        return self.soc * self.capacity
    
    @property
    def soc_percentage(self) -> float:
        """State of charge as percentage (0-100%)"""
        return self.soc * 100.0
    
    @property
    def is_full(self) -> bool:
        """Check if battery is at maximum capacity"""
        return self.soc >= 1.0
    
    @property
    def is_empty(self) -> bool:
        """Check if battery is at minimum capacity"""
        return self.soc <= 0.0
    
    def get_available_charge_capacity(self) -> float:
        """Get remaining charge capacity in MWh"""
        return (1.0 - self.soc) * self.capacity
    
    def get_available_discharge_capacity(self) -> float:
        """Get available discharge capacity in MWh"""
        return self.soc * self.capacity
    
    def get_effective_charge_power_limit(self) -> float:
        """Get effective charging power limit considering C-rate and power rating"""
        return min(self.max_power, self.max_charge_power_c_rate)
    
    def get_effective_discharge_power_limit(self) -> float:
        """Get effective discharging power limit considering C-rate and power rating"""
        return min(self.max_power, self.max_discharge_power_c_rate)
    
    def get_safe_soc_range(self) -> tuple:
        """Get safe SOC operating range"""
        return (self.min_soc_safe, self.max_soc_safe)
    
    def in_safe_operating_range(self) -> bool:
        """Check if SOC is within safe operating range"""
        return self.min_soc_safe <= self.soc <= self.max_soc_safe
    
    def get_operation_statistics(self) -> dict:
        """Get comprehensive battery operation statistics"""
        return {
            'current_soc_pct': self.soc_percentage,
            'current_energy_mwh': self.energy_stored,
            'max_soc_reached_pct': self.max_soc_reached * 100,
            'min_soc_reached_pct': self.min_soc_reached * 100,
            'safe_soc_range': f"{self.min_soc_safe*100:.0f}%-{self.max_soc_safe*100:.0f}%",
            'in_safe_range': self.in_safe_operating_range(),
            'c_rate_charge': self.c_rate_charge,
            'c_rate_discharge': self.c_rate_discharge,
            'max_charge_power_c_rate': self.max_charge_power_c_rate,
            'max_discharge_power_c_rate': self.max_discharge_power_c_rate,
            'freq_regulation_enabled': self.freq_regulation,
            'freq_response_power': self.freq_response_power,
            'limit_events': self.limit_events.copy(),
            'energy_limits_working': True
        }
    
    def can_charge(self, power: float, dt: float) -> bool:
        """Check if battery can charge at given power for time dt"""
        energy_to_add = power * dt * self.efficiency
        return (self.soc + energy_to_add / self.capacity) <= 1.0
    
    def can_discharge(self, power: float, dt: float) -> bool:
        """Check if battery can discharge at given power for time dt"""
        energy_to_remove = power * dt / self.efficiency
        return (self.soc - energy_to_remove / self.capacity) >= self.min_soc_safe
    
    def add_price_to_history(self, price: float):
        """Track price history for dynamic thresholds"""
        self.price_history.append(price)
        if len(self.price_history) > self.max_history_length:
            self.price_history.pop(0)
    
    def calculate_price_percentiles(self):
        """Calculate dynamic price thresholds based on recent history"""
        if len(self.price_history) < 24:  # Need at least 24 hours
            return 25.0, 45.0  # Default thresholds
        
        import numpy as np
        prices = np.array(self.price_history)
        p25 = np.percentile(prices, 25)
        p75 = np.percentile(prices, 75)
        return p25, p75
    
    
    def get_realistic_power_limits(self, desired_power: float, dt: float) -> float:
        """Calculate realistic power limits considering all constraints"""
        
        # C-rate limits
        if desired_power > 0:  # Discharging
            c_rate_limit = self.get_effective_discharge_power_limit()
            soc_limit = (self.soc - self.min_soc_safe) * self.capacity / dt * self.efficiency
        else:  # Charging  
            c_rate_limit = self.get_effective_charge_power_limit()
            soc_limit = (self.max_soc_safe - self.soc) * self.capacity / dt / self.efficiency
        
        # Hardware power limit
        hardware_limit = self.max_power
        
        # Take the minimum of all limits
        max_possible = min(c_rate_limit, abs(soc_limit), hardware_limit)
        
        return min(abs(desired_power), max_possible) * (1 if desired_power >= 0 else -1)

    def frequency_regulation_response(self, grid_frequency: float, dt: float) -> float:
        """Provide frequency regulation response based on grid frequency"""
        if not self.freq_regulation:
            return 0.0
        
        freq_deviation = grid_frequency - self.target_frequency
        
        # Only respond if outside deadband
        if abs(freq_deviation) <= self.freq_deadband:
            return 0.0
        
        # Calculate proportional response
        response_power = 0.0
        
        if freq_deviation < -self.freq_deadband:  # Frequency too low, need to discharge
            if self.soc > self.min_soc_safe:
                max_discharge = min(self.freq_response_power, self.get_effective_discharge_power_limit())
                # Scale response by frequency deviation (more deviation = more response)
                response_power = max_discharge * min(abs(freq_deviation) / 0.5, 1.0)
                
                # Check if we can actually discharge this amount
                if self.can_discharge(response_power, dt):
                    self.limit_events['freq_regulation_events'] += 1
                    return response_power  # Positive = discharge
                    
        elif freq_deviation > self.freq_deadband:  # Frequency too high, need to charge
            if self.soc < self.max_soc_safe:
                max_charge = min(self.freq_response_power, self.get_effective_charge_power_limit())
                # Scale response by frequency deviation
                response_power = max_charge * min(abs(freq_deviation) / 0.5, 1.0)
                
                # Check if we can actually charge this amount
                if self.can_charge(response_power, dt):
                    self.limit_events['freq_regulation_events'] += 1
                    return -response_power  # Negative = charge
        
        return 0.0
    
    def update_storage(self, price: float, dt: float, grid_frequency: float = 60.0) -> None:
        """FIXED: Robust battery operation with proper price-responsive logic and minimum action constraints"""
        
        # Add price to history for analysis
        self.add_price_to_history(price)
        
        # 1. Handle frequency regulation first (highest priority)
        freq_response = self.frequency_regulation_response(grid_frequency, dt)
        if abs(freq_response) > 0:
            self.power_output = freq_response
            if freq_response > 0:  # Discharging for frequency support
                energy_removed = freq_response * dt / self.efficiency
                new_soc = self.soc - energy_removed / self.capacity
                self.soc = max(new_soc, self.min_soc_safe)
            else:  # Charging for frequency support
                energy_added = abs(freq_response) * dt * self.efficiency
                new_soc = self.soc + energy_added / self.capacity
                self.soc = min(new_soc, self.max_soc_safe)
            self.action_duration += dt
            return  # Skip market arbitrage when doing frequency regulation
        
        # 2. FIXED: Dynamic price thresholds with conservative margins
        if len(self.price_history) >= 24:
            recent_prices = np.array(self.price_history[-24:])
            p25 = np.percentile(recent_prices, 25)  # Bottom 25% for charging
            p75 = np.percentile(recent_prices, 75)  # Top 25% for discharging
            
            # Add margins to prevent continuous cycling while allowing reasonable opportunities
            charge_threshold = p25 * 0.95   # Charge when price < 95% of 25th percentile
            discharge_threshold = p75 * 1.10  # Discharge when price > 110% of 75th percentile
        else:
            # Conservative default thresholds until we have history
            charge_threshold = 25.0
            discharge_threshold = 45.0
        
        # 3. Check minimum action time constraint to prevent erratic cycling
        if self.action_duration < self.min_action_time:
            # Continue current action for minimum time
            if self.last_action == 'charge' and self.soc < self.max_soc_safe:
                # Continue charging at reduced rate
                continue_power = min(5.0, self.get_effective_charge_power_limit())
                self.power_output = -continue_power
                energy_added = continue_power * dt * self.efficiency
                self.soc = min(self.soc + energy_added / self.capacity, self.max_soc_safe)
            elif self.last_action == 'discharge' and self.soc > self.min_soc_safe:
                # Continue discharging at reduced rate
                continue_power = min(10.0, self.get_effective_discharge_power_limit())
                self.power_output = continue_power
                energy_removed = continue_power * dt / self.efficiency
                self.soc = max(self.soc - energy_removed / self.capacity, self.min_soc_safe)
            else:
                self.power_output = 0.0
                self.last_action = 'idle'
            
            self.action_duration += dt
            return
        
        # 4. CHARGING LOGIC (more restrictive)
        if price < charge_threshold and self.soc < self.max_soc_safe:
            # Calculate reasonable charge power with all constraints
            available_capacity = (self.max_soc_safe - self.soc) * self.capacity
            max_charge_power = min(
                self.get_effective_charge_power_limit(),
                available_capacity / (dt * self.efficiency),
                self.max_power * 0.5  # Limit to 50% of max power for smoother operation
            )
            
            # Scale by price incentive (more aggressive when prices are very low)
            price_factor = min(1.0, (charge_threshold - price) / charge_threshold)
            target_power = max_charge_power * max(0.3, price_factor)  # At least 30% power
            
            if target_power > 1.0:  # Minimum 1 MW threshold
                self.power_output = -target_power  # Negative = charging
                energy_added = target_power * dt * self.efficiency
                self.soc = min(self.soc + energy_added / self.capacity, self.max_soc_safe)
                
                # Update action tracking
                if self.last_action != 'charge':
                    self.action_duration = 0
                self.last_action = 'charge'
        
        # 5. DISCHARGING LOGIC (more restrictive)  
        elif price > discharge_threshold and self.soc > self.min_soc_safe:
            # Calculate reasonable discharge power with all constraints
            available_energy = (self.soc - self.min_soc_safe) * self.capacity
            max_discharge_power = min(
                self.get_effective_discharge_power_limit(),
                available_energy * self.efficiency / dt,
                self.max_power * 0.6  # Limit to 60% of max power
            )
            
            # Scale by price incentive (more aggressive when prices are very high)
            price_factor = min(1.0, (price - discharge_threshold) / discharge_threshold)
            target_power = max_discharge_power * max(0.2, price_factor)  # At least 20% power
            
            if target_power > 1.0:  # Minimum 1 MW threshold
                self.power_output = target_power  # Positive = discharging
                energy_removed = target_power * dt / self.efficiency
                self.soc = max(self.soc - energy_removed / self.capacity, self.min_soc_safe)
                
                # Update action tracking
                if self.last_action != 'discharge':
                    self.action_duration = 0
                self.last_action = 'discharge'
        
        # 6. IDLE STATE with gentle SOC rebalancing
        else:
            # Gradual rebalancing toward 60% SOC during neutral price periods
            target_soc = 0.6
            soc_error = target_soc - self.soc
            
            # Only rebalance if price is in neutral range and error is significant
            if abs(soc_error) > 0.15 and charge_threshold < price < discharge_threshold:
                rebalance_power = min(5.0, abs(soc_error) * self.capacity / 8)  # Gentle rebalancing
                
                if soc_error > 0:  # Need to charge to reach target
                    self.power_output = -rebalance_power
                    energy_added = rebalance_power * dt * self.efficiency
                    self.soc = min(self.soc + energy_added / self.capacity, self.max_soc_safe)
                else:  # Need to discharge to reach target
                    self.power_output = rebalance_power
                    energy_removed = rebalance_power * dt / self.efficiency
                    self.soc = max(self.soc - energy_removed / self.capacity, self.min_soc_safe)
                
                # Update action tracking for idle rebalancing
                if self.last_action != 'idle':
                    self.action_duration = 0
                self.last_action = 'idle'
            else:
                self.power_output = 0.0
                if self.last_action != 'idle':
                    self.action_duration = 0
                self.last_action = 'idle'
        
        # Update action duration
        self.action_duration += dt
        
        # Final safety checks
        self.soc = np.clip(self.soc, 0.0, 1.0)
        
        # Update SOC extremes
        self.max_soc_reached = max(self.max_soc_reached, self.soc)
        self.min_soc_reached = min(self.min_soc_reached, self.soc)

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
        self.omega_s = 2 * np.pi * 60  # Synchronous frequency [rad/s]
        
    def swing_equation_stability(self, generators: List[Generator], 
                               S_base: float = 100.0) -> Tuple[bool, np.ndarray, Dict]:
        """
        Analyze small-signal stability using proper swing equation linearization
        Based on multi-machine swing equation: M_i * d²δ_i/dt² + D_i * dδ_i/dt = P_mi - P_ei
        """
        n = len(generators)
        if n == 0:
            return True, np.array([]), {}
        
        # Only include generators with physical inertia (synchronous machines)
        sync_gens = [gen for gen in generators if gen.inertia_H > 0]
        n_sync = len(sync_gens)
        
        if n_sync == 0:
            print("WARNING: No synchronous generators with inertia found!")
            return False, np.array([]), {'warning': 'No inertia in system'}
        
        # Build system matrix A for linearized swing equations
        # State vector: [δ₁, δ₂, ..., δₙ, ω₁, ω₂, ..., ωₙ]
        # Where δᵢ = power angle, ωᵢ = angular frequency deviation
        A = np.zeros((2*n_sync, 2*n_sync))
        
        total_inertia = sum(gen.inertia_H * gen.max_capacity for gen in sync_gens)
        avg_inertia = total_inertia / sum(gen.max_capacity for gen in sync_gens)
        
        # Coupling strength (simplified - could be network-based)
        K_coupling = 1.0  # Synchronizing power coefficient
        
        for i, gen_i in enumerate(sync_gens):
            # δ equations: dδᵢ/dt = ωᵢ
            A[i, n_sync + i] = 1.0
            
            # ω equations: dωᵢ/dt = (P_mi - P_ei - D_i*ω_i)/M_i
            M_i = gen_i.M  # Inertia coefficient
            D_i = gen_i.damping_D
            
            # Self-regulation term: -D_i/M_i
            A[n_sync + i, n_sync + i] = -D_i / M_i
            
            # Synchronizing power terms: -K/M_i
            A[n_sync + i, i] = -K_coupling / M_i
            
            # Cross-coupling with other generators
            for j, gen_j in enumerate(sync_gens):
                if i != j:
                    # Coupling between generators
                    coupling_ij = K_coupling * 0.1  # Reduced coupling between machines
                    A[n_sync + i, j] = coupling_ij / M_i
        
        eigenvals, eigenvecs = eig(A)
        stable = np.all(eigenvals.real < 0)
        stability_metrics = {
            'total_system_inertia': total_inertia,
            'average_inertia': avg_inertia,
            'synchronous_generators': n_sync,
            'total_generators': n,
            'inertia_ratio': n_sync / n if n > 0 else 0,
            'min_damping_ratio': min(gen.damping_D for gen in sync_gens),
            'system_matrix_condition': np.linalg.cond(A),
            'dominant_eigenvalue': eigenvals[np.argmax(eigenvals.real)],
            'eigenvalue_real_parts': eigenvals.real,
            'stability_margin': -np.max(eigenvals.real) if len(eigenvals) > 0 else 0
        }

        return stable, eigenvals, stability_metrics
    
    def small_signal_stability(self, generators: List[Generator], 
                             market_params: MarketParameters) -> Tuple[bool, np.ndarray]:
        """Legacy method - now calls proper swing equation analysis"""
        stable, eigenvals, _ = self.swing_equation_stability(generators)
    
    def convergence_analysis(self, generators: List[Generator]) -> Dict:
        """Analyze convergence properties"""
        sync_gens = [gen for gen in generators if gen.inertia_H > 0]
        
        if not sync_gens:
            return {'warning': 'No synchronous generators for analysis'}
        time_constants = []
        eigenvalues = []
        settling_times = []

        for gen in sync_gens:
            # Physical time constant: T = 2H/(D*ω_s)
            if gen.damping_D > 0:
                T = 2 * gen.inertia_H / (gen.damping_D * self.omega_s)
                settling_time = 4 * T  # 4 time constants for 98% settling
            else:
                T = float('inf')  # Undamped
                settling_time = float('inf')
            
            # Approximate eigenvalue for single machine
            if gen.damping_D > 0:
                eigenval = -gen.damping_D / (2 * gen.inertia_H)
            else:
                eigenval = 0.0

            time_constants.append(T)
            eigenvalues.append(eigenval)
        
        return {
            'time_constants': np.array(time_constants),
            'eigenvalues': np.array(eigenvalues),
            'settling_times': np.array(settling_times),
            'max_settling_time': max(settling_times) if settling_times else 0,
            'min_time_constant': min(time_constants) if time_constants else 0,
            'system_inertia': sum(gen.inertia_H * gen.max_capacity for gen in sync_gens),
            'system_damping': sum(gen.damping_D * gen.max_capacity for gen in sync_gens)
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
                     adjustment_param: float, max_capacity: float = 1000.0,
                     inertia_H: float = 5.0, damping_D: float = 1.0,
                     generator_type: str = 'thermal') -> int:
        """Add conventional generator with reasonable initial output"""
        gen_id = len(self.generators)
        gen = Generator(gen_id, linear_cost, quadratic_cost, adjustment_param, max_capacity, inertia_H, damping_D, generator_type)
        
        # FIXED: Initialize with reasonable starting output (30-50% of capacity) to prevent initial price spikes
        gen.output = max_capacity * np.random.uniform(0.3, 0.5)
        
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
                          efficiency: float = 0.9, c_rate_charge: float = 1.0, 
                          c_rate_discharge: float = 1.0, safety_margins: bool = True,
                          freq_regulation: bool = True) -> int:
        """Add enhanced battery storage system with C-rates and safety features"""
        storage_id = len(self.storage)
        battery = BatteryStorage(storage_id, capacity_mwh, power_mw, efficiency,
                               c_rate_charge, c_rate_discharge, safety_margins, freq_regulation)
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
                # Generate realistic grid frequency variation
                base_frequency = 60.0
                freq_noise = np.random.normal(0, 0.02)  # ±0.02 Hz random variation
                grid_frequency = base_frequency + freq_noise
                
                storage.update_storage(price, self.params.dt, grid_frequency)
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
            stable, eigenvals, stability_metrics = self.stability_analyzer.swing_equation_stability(
                self.generators)
            convergence_info = self.stability_analyzer.convergence_analysis(self.generators)
        else:
            stable, eigenvals, stability_metrics = True, np.array([]), {}
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
                'metrics': stability_metrics,
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
            plt.scatter(eigenvals.real, eigenvals.imag, s=50, alpha=0.7, c='blue')
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Stability Boundary')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')

            # Add stability information
            stable = results['stability']['stable']
            metrics = results['stability'].get('metrics', {})
            title_text = f"System Eigenvalues (Stable: {stable})"
            if 'total_system_inertia' in metrics:
                title_text += f"\nInertia: {metrics['total_system_inertia']:.1f} MW·s"
            plt.title(f"System Eigenvalues (Stable: {results['stability']['stable']})")
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No synchronous\ngenerators with\ninertia for proper\nstability analysis',  
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
    
    # Add conventional generators
    # Thermal generators: H = 2-10 seconds, D = 1-2 per unit
    market.add_generator(linear_cost=2.0, quadratic_cost=0.02, adjustment_param=3.0, max_capacity=500, inertia_H=6.0, damping_D=1.5, generator_type='thermal')
    market.add_generator(linear_cost=1.75, quadratic_cost=0.0175, adjustment_param=4.0, max_capacity=600,  inertia_H=7.5, damping_D=1.2, generator_type='thermal')
    market.add_generator(linear_cost=3.0, quadratic_cost=0.025, adjustment_param=2.5, max_capacity=400, inertia_H=5.0, damping_D=1.8, generator_type='thermal')
    market.add_generator(linear_cost=3.0, quadratic_cost=0.025, adjustment_param=3.0, max_capacity=450, inertia_H=4.5, damping_D=1.6, generator_type='thermal')
    

    # Add renewable generation
    market.add_renewable(capacity=200, renewable_type='solar')
    market.add_renewable(capacity=150, renewable_type='wind')
    market.add_renewable(capacity=300, renewable_type='hydro')  # Hydropower plant
    market.add_renewable(capacity=800, renewable_type='nuclear')  # Nuclear baseload
    
    # Add battery storage with different C-rates and features
    # Fast-charging battery (2C rate = 30 min to full charge)
    market.add_battery_storage(capacity_mwh=100, power_mw=50, efficiency=0.9, 
                             c_rate_charge=2.0, c_rate_discharge=1.5, 
                             safety_margins=True, freq_regulation=True)
    
    # Slower, larger battery (0.5C rate = 2 hours to full charge) - more typical for grid storage
    market.add_battery_storage(capacity_mwh=200, power_mw=100, efficiency=0.85,
                             c_rate_charge=0.5, c_rate_discharge=0.75,
                             safety_margins=True, freq_regulation=True)
    
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
    
    stability = results['stability']
    metrics = stability.get('metrics', {})
    convergence = stability.get('convergence', {})
    
    print(f"\n=== Enhanced Stability Analysis ===")
    if stability['stable']:
        print("System stability: STABLE")
    else:
        print("System stability: UNSTABLE")
    
    if 'total_system_inertia' in metrics:
        print(f"Total system inertia: {metrics['total_system_inertia']:.2f} MW·s")
        print(f"Average generator inertia: {metrics['average_inertia']:.2f} s")
        print(f"Synchronous generators: {metrics['synchronous_generators']}")
        print(f"Inertia ratio: {metrics['inertia_ratio']:.2%}")
        print(f"Stability margin: {metrics['stability_margin']:.4f}")
    
    if 'max_settling_time' in convergence:
        print(f"Maximum settling time: {convergence['max_settling_time']:.2f} hours")
        print(f"System inertia (convergence): {convergence['system_inertia']:.2f} MW·s")
    
    # Plot results
    market.plot_results(results)