import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class SolarEnergySystem:
    def __init__(self):
        self.data = []  # Data repository for training machine learning models
        self.load_data()  # Load historical data for machine learning
        self.panel_angle = 0  # Initial panel angle
        self.machine_learning_model = self.train_machine_learning_model()

    def load_data(self):
        # Simulated historical data (features: temperature, light intensity; target: energy output)
        for _ in range(1000):
            temperature = random.uniform(10, 40)
            light_intensity = random.uniform(100, 1000)
            energy_output = temperature * 2 + light_intensity * 0.5 + random.uniform(-20, 20)
            self.data.append([temperature, light_intensity, energy_output])
        
    def train_machine_learning_model(self):
        # Prepare data for machine learning
        data = np.array(self.data)
        features = data[:, :-1]  # Temperature and light intensity as features
        targets = data[:, -1]  # Energy output as target

        # Split data into training and testing sets
        features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, random_state=42)

        # Train a machine learning model (Random Forest Regressor)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_train, targets_train)

        # Evaluate the model
        accuracy = model.score(features_test, targets_test)
        print(f"Machine Learning Model Accuracy: {accuracy:.2f}")

        return model

    def collect_real_time_data(self):
        # Simulated real-time sensor data
        temperature = random.uniform(10, 40)
        light_intensity = random.uniform(100, 1000)
        return temperature, light_intensity

    def adjust_panel_angle(self, temperature, light_intensity):
        # Use machine learning model to predict optimal panel angle
        predicted_energy_output = self.machine_learning_model.predict([[temperature, light_intensity]])
        self.panel_angle = int(predicted_energy_output)  # Convert energy output to panel angle

    def simulate_solar_panel_adjustment(self):
        # Simulated adjustment of physical solar panels (e.g., servo motors)
        print(f"Adjusting Solar Panels to {self.panel_angle} degrees")

    def run_system(self):
        while True:
            temperature, light_intensity = self.collect_real_time_data()
            self.adjust_panel_angle(temperature, light_intensity)
            self.simulate_solar_panel_adjustment()


# Main function
if __name__ == "__main__":
    solar_system = SolarEnergySystem()
    solar_system.run_system()
