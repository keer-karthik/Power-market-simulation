import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class StabilityFeatureExtractor:
    """Extract features for stability prediction from power system data"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def extract_features(self, market_data: Dict, system_state: Dict) -> np.ndarray:
        """Extract comprehensive features for stability analysis"""
        features = []
        feature_names = []
        
        # Market-based features
        prices = market_data.get('price', np.array([35.0]))
        if len(prices) > 1:
            # Price statistics
            features.extend([
                np.mean(prices),                    # Mean price
                np.std(prices),                     # Price volatility
                np.max(prices),                     # Peak price
                np.min(prices),                     # Minimum price
                np.percentile(prices, 95),          # 95th percentile
                np.percentile(prices, 5),           # 5th percentile
                len(prices[prices > np.mean(prices) * 1.5])  # Price spike count
            ])
            feature_names.extend([
                'price_mean', 'price_std', 'price_max', 'price_min',
                'price_p95', 'price_p5', 'price_spike_count'
            ])
            
            # Price dynamics
            if len(prices) > 2:
                price_diff = np.diff(prices)
                features.extend([
                    np.mean(price_diff),            # Mean price change
                    np.std(price_diff),             # Price change volatility
                    np.max(np.abs(price_diff)),     # Maximum price jump
                    np.sum(price_diff > 0) / len(price_diff)  # Fraction increasing
                ])
                feature_names.extend([
                    'price_change_mean', 'price_change_std', 
                    'price_max_jump', 'price_increase_fraction'
                ])
        else:
            # Default values for insufficient data
            features.extend([35.0, 1.0, 35.0, 35.0, 35.0, 35.0, 0.0, 0.0, 1.0, 0.0, 0.5])
            feature_names.extend([
                'price_mean', 'price_std', 'price_max', 'price_min',
                'price_p95', 'price_p5', 'price_spike_count',
                'price_change_mean', 'price_change_std', 
                'price_max_jump', 'price_increase_fraction'
            ])
        
        # Generation mix features
        generation = market_data.get('generation', {})
        total_generation = 0
        conventional_gen = 0
        renewable_gen = 0
        
        for gen_key, gen_data in generation.items():
            gen_total = np.sum(gen_data) if hasattr(gen_data, '__len__') else gen_data
            total_generation += gen_total
            
            if gen_key.startswith('gen_'):
                conventional_gen += gen_total
            elif gen_key.startswith('ren_'):
                renewable_gen += gen_total
        
        if total_generation > 0:
            renewable_share = renewable_gen / total_generation
            conventional_share = conventional_gen / total_generation
        else:
            renewable_share = 0.0
            conventional_share = 1.0
        
        features.extend([
            total_generation,
            renewable_share,
            conventional_share,
            renewable_gen,
            conventional_gen
        ])
        feature_names.extend([
            'total_generation', 'renewable_share', 'conventional_share',
            'renewable_generation', 'conventional_generation'
        ])
        
        # Storage features
        storage = market_data.get('storage', {})
        total_storage_power = 0
        avg_soc = 0
        storage_cycles = 0
        n_storage = len(storage)
        
        for storage_key, storage_data in storage.items():
            if 'power' in storage_data:
                power_data = storage_data['power']
                total_storage_power += np.sum(np.abs(power_data))
                
                # Calculate cycles (simplified)
                if 'soc' in storage_data:
                    soc_data = storage_data['soc']
                    avg_soc += np.mean(soc_data)
                    soc_changes = np.abs(np.diff(soc_data)) if len(soc_data) > 1 else [0]
                    storage_cycles += np.sum(soc_changes) / 2
        
        if n_storage > 0:
            avg_soc /= n_storage
        
        features.extend([
            total_storage_power,
            avg_soc,
            storage_cycles,
            n_storage
        ])
        feature_names.extend([
            'total_storage_power', 'avg_soc', 'storage_cycles', 'n_storage_units'
        ])
        
        # System-level features
        if 'generators' in system_state:
            generators = system_state['generators']
            n_generators = len(generators)
            
            # Generator characteristics
            total_capacity = sum(gen.max_capacity for gen in generators)
            avg_marginal_cost = np.mean([gen.marginal_cost() for gen in generators])
            capacity_utilization = total_generation / total_capacity if total_capacity > 0 else 0
            
            # System inertia proxy (larger generators typically have more inertia)
            weighted_capacity = sum(gen.max_capacity ** 1.5 for gen in generators)
            inertia_proxy = weighted_capacity / n_generators if n_generators > 0 else 0
            
            features.extend([
                n_generators,
                total_capacity,
                avg_marginal_cost,
                capacity_utilization,
                inertia_proxy
            ])
            feature_names.extend([
                'n_generators', 'total_capacity', 'avg_marginal_cost',
                'capacity_utilization', 'inertia_proxy'
            ])
        else:
            features.extend([4, 2000, 25.0, 0.5, 1000])
            feature_names.extend([
                'n_generators', 'total_capacity', 'avg_marginal_cost',
                'capacity_utilization', 'inertia_proxy'
            ])
        
        # Demand features
        demand = market_data.get('demand', np.array([1000.0]))
        if len(demand) > 1:
            features.extend([
                np.mean(demand),
                np.std(demand),
                np.max(demand),
                np.min(demand)
            ])
            feature_names.extend([
                'demand_mean', 'demand_std', 'demand_max', 'demand_min'
            ])
        else:
            avg_demand = demand[0] if len(demand) > 0 else 1000.0
            features.extend([avg_demand, 50.0, avg_demand, avg_demand])
            feature_names.extend([
                'demand_mean', 'demand_std', 'demand_max', 'demand_min'
            ])
        
        # Time-based features (if available)
        time_data = market_data.get('time', np.arange(24))
        if len(time_data) > 0:
            current_hour = time_data[-1] % 24
            features.extend([
                current_hour,
                np.sin(2 * np.pi * current_hour / 24),  # Cyclical encoding
                np.cos(2 * np.pi * current_hour / 24)
            ])
            feature_names.extend(['hour', 'hour_sin', 'hour_cos'])
        else:
            features.extend([12, 0, 1])
            feature_names.extend(['hour', 'hour_sin', 'hour_cos'])
        
        # Network features (simplified)
        features.extend([
            total_generation / (np.mean(demand) if len(demand) > 0 else 1000),  # Supply-demand ratio
            renewable_share * total_generation / 1000,  # Renewable penetration level
            1.0 if renewable_share > 0.3 else 0.0,     # High renewable flag
            1.0 if capacity_utilization > 0.8 else 0.0  # High utilization flag
        ])
        feature_names.extend([
            'supply_demand_ratio', 'renewable_penetration_level',
            'high_renewable_flag', 'high_utilization_flag'
        ])
        
        self.feature_names = feature_names
        return np.array(features)
    
    def fit_scaler(self, feature_matrix: np.ndarray):
        """Fit the feature scaler"""
        self.scaler.fit(feature_matrix)
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.scaler.transform(features)

class StabilityPredictor:
    """Machine learning model for power system stability prediction"""
    
    def __init__(self):
        self.feature_extractor = StabilityFeatureExtractor()
        self.stability_classifier = None
        self.eigenvalue_regressor = None
        self.feature_selector = None
        self.is_trained = False
        
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for stability prediction"""
        print(f"Generating {n_samples} training samples...")
        
        features = []
        stability_labels = []
        eigenvalue_targets = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  Generated {i}/{n_samples} samples")
            
            # Generate synthetic system parameters
            n_gen = np.random.randint(2, 8)
            total_capacity = np.random.uniform(1000, 3000)
            renewable_share = np.random.uniform(0, 0.8)
            demand_level = np.random.uniform(500, 2500)
            
            # Generate market data
            market_data = self._generate_synthetic_market_data(
                n_gen, total_capacity, renewable_share, demand_level)
            
            # Generate system state
            system_state = self._generate_synthetic_system_state(
                n_gen, total_capacity, renewable_share)
            
            # Extract features
            feature_vector = self.feature_extractor.extract_features(market_data, system_state)
            features.append(feature_vector)
            
            # Determine stability (heuristic-based for training)
            is_stable, min_eigenvalue = self._determine_stability_heuristic(
                market_data, system_state)
            
            stability_labels.append(1 if is_stable else 0)
            eigenvalue_targets.append(min_eigenvalue)
        
        return np.array(features), np.array(stability_labels), np.array(eigenvalue_targets)
    
    def _generate_synthetic_market_data(self, n_gen: int, total_capacity: float, 
                                      renewable_share: float, demand_level: float) -> Dict:
        """Generate synthetic market data"""
        n_hours = 24
        time_array = np.arange(n_hours)
        
        # Price data with volatility
        base_price = np.random.uniform(25, 45)
        price_volatility = np.random.uniform(0.1, 0.3)
        prices = base_price * (1 + price_volatility * np.random.normal(0, 1, n_hours))
        prices = np.maximum(prices, 5)  # Minimum price
        
        # Generation data
        generation = {}
        conventional_gen = total_capacity * (1 - renewable_share)
        renewable_gen = total_capacity * renewable_share
        
        for i in range(n_gen):
            gen_capacity = conventional_gen / n_gen
            utilization = np.random.uniform(0.3, 0.9)
            generation[f'gen_{i}'] = np.ones(n_hours) * gen_capacity * utilization
        
        # Renewable generation with daily pattern
        if renewable_share > 0:
            solar_pattern = np.maximum(0, np.sin(np.pi * (time_array - 6) / 12))
            generation['ren_0'] = renewable_gen * solar_pattern * np.random.uniform(0.7, 1.0)
        
        # Storage data
        storage = {}
        if np.random.random() > 0.3:  # 70% chance of having storage
            n_storage = np.random.randint(1, 3)
            for i in range(n_storage):
                power_profile = np.random.normal(0, 20, n_hours)
                soc_profile = 0.5 + 0.3 * np.sin(2 * np.pi * time_array / 24)
                storage[f'battery_{i}'] = {
                    'power': power_profile,
                    'soc': np.clip(soc_profile, 0.1, 0.9)
                }
        
        # Demand with daily pattern
        demand_pattern = 0.8 + 0.4 * np.sin(2 * np.pi * (time_array - 6) / 24)
        demand = demand_level * demand_pattern
        
        return {
            'time': time_array,
            'price': prices,
            'generation': generation,
            'storage': storage,
            'demand': demand
        }
    
    def _generate_synthetic_system_state(self, n_gen: int, total_capacity: float, 
                                       renewable_share: float) -> Dict:
        """Generate synthetic system state"""
        from Enhanced_Power_Market_Model import Generator
        
        generators = []
        conventional_capacity = total_capacity * (1 - renewable_share)
        
        for i in range(n_gen):
            linear_cost = np.random.uniform(1.5, 4.0)
            quadratic_cost = np.random.uniform(0.01, 0.05)
            adjustment_param = np.random.uniform(1.0, 5.0)
            capacity = conventional_capacity / n_gen
            
            gen = Generator(i, linear_cost, quadratic_cost, adjustment_param, capacity)
            generators.append(gen)
        
        return {'generators': generators}
    
    def _determine_stability_heuristic(self, market_data: Dict, system_state: Dict) -> Tuple[bool, float]:
        """Heuristic-based stability determination for training"""
        
        # Extract key indicators
        prices = market_data.get('price', [35])
        renewable_share = 0
        total_gen = 0
        
        generation = market_data.get('generation', {})
        for key, gen_data in generation.items():
            gen_total = np.sum(gen_data) if hasattr(gen_data, '__len__') else gen_data
            total_gen += gen_total
            if key.startswith('ren_'):
                renewable_share += gen_total
        
        if total_gen > 0:
            renewable_share /= total_gen
        
        # Stability indicators
        price_volatility = np.std(prices) if len(prices) > 1 else 0
        max_price = np.max(prices) if len(prices) > 0 else 35
        
        generators = system_state.get('generators', [])
        n_generators = len(generators)
        
        # Heuristic stability assessment
        instability_factors = 0
        
        # High price volatility
        if price_volatility > 10:
            instability_factors += 1
        
        # Extreme prices
        if max_price > 100:
            instability_factors += 2
        
        # High renewable penetration without enough generators
        if renewable_share > 0.6 and n_generators < 3:
            instability_factors += 1
        
        # Low number of generators
        if n_generators < 2:
            instability_factors += 2
        
        # Generator parameter issues
        if generators:
            for gen in generators:
                if gen.c <= 0.001:  # Very low quadratic cost
                    instability_factors += 1
                if gen.A > 10:  # Very high adjustment parameter
                    instability_factors += 1
        
        # Determine stability
        is_stable = instability_factors <= 2
        
        # Synthetic eigenvalue (negative for stable)
        if is_stable:
            min_eigenvalue = -np.random.uniform(0.1, 2.0)
        else:
            min_eigenvalue = np.random.uniform(-0.5, 1.0)
        
        return is_stable, min_eigenvalue
    
    def train(self, n_training_samples: int = 1000, test_size: float = 0.2):
        """Train the stability prediction models"""
        print("Training stability prediction models...")
        
        # Generate training data
        features, stability_labels, eigenvalue_targets = self.generate_training_data(n_training_samples)
        
        # Fit feature scaler
        self.feature_extractor.fit_scaler(features)
        features_scaled = self.feature_extractor.transform_features(features)
        
        # Feature selection
        print("Selecting important features...")
        self.feature_selector = SelectKBest(f_classif, k=min(20, features.shape[1]))
        features_selected = self.feature_selector.fit_transform(features_scaled, stability_labels)
        
        # Split data
        X_train, X_test, y_train_stab, y_test_stab = train_test_split(
            features_selected, stability_labels, test_size=test_size, random_state=42)
        
        _, _, y_train_eig, y_test_eig = train_test_split(
            features_selected, eigenvalue_targets, test_size=test_size, random_state=42)
        
        # Train stability classifier
        print("Training stability classifier...")
        self.stability_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.stability_classifier.fit(X_train, y_train_stab)
        
        # Train eigenvalue regressor
        print("Training eigenvalue regressor...")
        self.eigenvalue_regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.eigenvalue_regressor.fit(X_train, y_train_eig)
        
        # Evaluate models
        stab_accuracy = self.stability_classifier.score(X_test, y_test_stab)
        eig_score = self.eigenvalue_regressor.score(X_test, y_test_eig)
        
        print(f"\nModel Performance:")
        print(f"Stability classification accuracy: {stab_accuracy:.3f}")
        print(f"Eigenvalue regression R²: {eig_score:.3f}")
        
        # Feature importance
        feature_names = np.array(self.feature_extractor.feature_names)
        selected_features = self.feature_selector.get_support()
        selected_feature_names = feature_names[selected_features]
        
        importances = self.stability_classifier.feature_importances_
        feature_importance = list(zip(selected_feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature:25s}: {importance:.3f}")
        
        self.is_trained = True
        
        return {
            'stability_accuracy': stab_accuracy,
            'eigenvalue_r2': eig_score,
            'feature_importance': feature_importance
        }
    
    def predict_stability(self, market_data: Dict, system_state: Dict) -> Dict:
        """Predict system stability"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        features = self.feature_extractor.extract_features(market_data, system_state)
        features_scaled = self.feature_extractor.transform_features(features)
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Predict stability
        stability_prob = self.stability_classifier.predict_proba(features_selected)[0]
        stability_prediction = self.stability_classifier.predict(features_selected)[0]
        
        # Predict eigenvalue
        eigenvalue_prediction = self.eigenvalue_regressor.predict(features_selected)[0]
        
        # Handle single-class prediction case
        if len(stability_prob) == 1:
            # Only one class was seen during training
            if stability_prediction == 1:
                stable_prob = float(stability_prob[0])
                unstable_prob = 1.0 - stable_prob
            else:
                unstable_prob = float(stability_prob[0])
                stable_prob = 1.0 - unstable_prob
        else:
            # Normal two-class case
            unstable_prob = float(stability_prob[0])
            stable_prob = float(stability_prob[1])
        
        return {
            'is_stable': bool(stability_prediction),
            'stability_probability': stable_prob,
            'instability_probability': unstable_prob,
            'predicted_eigenvalue': float(eigenvalue_prediction),
            'confidence': float(max(stable_prob, unstable_prob))
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'feature_extractor': self.feature_extractor,
            'stability_classifier': self.stability_classifier,
            'eigenvalue_regressor': self.eigenvalue_regressor,
            'feature_selector': self.feature_selector
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.feature_extractor = model_data['feature_extractor']
        self.stability_classifier = model_data['stability_classifier']
        self.eigenvalue_regressor = model_data['eigenvalue_regressor']
        self.feature_selector = model_data['feature_selector']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

class RealTimeStabilityMonitor:
    """Real-time stability monitoring using ML predictions"""
    
    def __init__(self, stability_predictor: StabilityPredictor):
        self.predictor = stability_predictor
        self.history = []
        self.alert_threshold = 0.3  # Alert if stability probability < 30%
        
    def monitor_step(self, market_data: Dict, system_state: Dict) -> Dict:
        """Monitor one time step"""
        if not self.predictor.is_trained:
            return {'error': 'Predictor not trained'}
        
        prediction = self.predictor.predict_stability(market_data, system_state)
        
        # Determine alert level
        stability_prob = prediction['stability_probability']
        if stability_prob < self.alert_threshold:
            alert_level = 'HIGH'
        elif stability_prob < 0.6:
            alert_level = 'MEDIUM'
        else:
            alert_level = 'LOW'
        
        result = {
            'timestamp': len(self.history),
            'prediction': prediction,
            'alert_level': alert_level,
            'recommendation': self._generate_recommendation(prediction, market_data)
        }
        
        self.history.append(result)
        return result
    
    def _generate_recommendation(self, prediction: Dict, market_data: Dict) -> str:
        """Generate operational recommendations"""
        if prediction['is_stable']:
            return "System operating normally"
        
        stability_prob = prediction['stability_probability']
        
        if stability_prob < 0.2:
            return "CRITICAL: Consider emergency actions - reduce renewable output, start additional generators"
        elif stability_prob < 0.4:
            return "WARNING: Increase conventional generation, reduce system stress"
        else:
            return "CAUTION: Monitor system closely, prepare contingency actions"
    
    def plot_stability_trend(self, n_recent: int = 100):
        """Plot recent stability trend"""
        if not self.history:
            print("No monitoring history available")
            return
        
        recent_history = self.history[-n_recent:]
        timestamps = [h['timestamp'] for h in recent_history]
        stability_probs = [h['prediction']['stability_probability'] for h in recent_history]
        eigenvalues = [h['prediction']['predicted_eigenvalue'] for h in recent_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Stability probability
        ax1.plot(timestamps, stability_probs, 'b-', linewidth=2, label='Stability Probability')
        ax1.axhline(y=self.alert_threshold, color='r', linestyle='--', label='Alert Threshold')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Caution Threshold')
        ax1.set_ylabel('Stability Probability')
        ax1.set_title('Real-Time Stability Monitoring')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Eigenvalue prediction
        ax2.plot(timestamps, eigenvalues, 'g-', linewidth=2, label='Predicted Eigenvalue')
        ax2.axhline(y=0, color='r', linestyle='--', label='Stability Boundary')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Predicted Eigenvalue')
        ax2.set_title('Predicted System Eigenvalue')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Create and train the stability predictor
    predictor = StabilityPredictor()
    
    print("Training stability prediction model...")
    training_results = predictor.train(n_training_samples=500)  # Smaller for demo
    
    # Test prediction
    print("\nTesting prediction...")
    
    # Create synthetic test data
    test_market_data = {
        'price': np.array([35, 40, 45, 38, 42]),
        'generation': {
            'gen_0': np.array([200, 220, 210, 205, 215]),
            'gen_1': np.array([150, 160, 155, 158, 162]),
            'ren_0': np.array([80, 85, 90, 88, 92])
        },
        'storage': {
            'battery_0': {
                'power': np.array([10, -5, 8, -3, 12]),
                'soc': np.array([0.6, 0.58, 0.61, 0.59, 0.63])
            }
        },
        'demand': np.array([450, 460, 455, 451, 469]),
        'time': np.array([10, 11, 12, 13, 14])
    }
    
    from Enhanced_Power_Market_Model import Generator
    test_system_state = {
        'generators': [
            Generator(0, 2.0, 0.02, 3.0, 300),
            Generator(1, 1.8, 0.018, 3.5, 250)
        ]
    }
    
    prediction = predictor.predict_stability(test_market_data, test_system_state)
    
    print(f"\nStability Prediction Results:")
    print(f"Is Stable: {prediction['is_stable']}")
    print(f"Stability Probability: {prediction['stability_probability']:.3f}")
    print(f"Predicted Eigenvalue: {prediction['predicted_eigenvalue']:.3f}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    
    # Test real-time monitoring
    print("\nTesting real-time monitoring...")
    monitor = RealTimeStabilityMonitor(predictor)
    
    for i in range(5):
        # Modify test data slightly for each step
        test_market_data['price'] = test_market_data['price'] + np.random.normal(0, 2, 5)
        result = monitor.monitor_step(test_market_data, test_system_state)
        print(f"Step {i+1}: Alert Level = {result['alert_level']}, "
              f"Stability Prob = {result['prediction']['stability_probability']:.3f}")
    
    # Plot monitoring results
    monitor.plot_stability_trend()
    
    # Save model for future use
    # predictor.save_model('stability_model.pkl')