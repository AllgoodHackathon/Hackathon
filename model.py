import numpy as np
import random
import json
import os

class ACSystem:
    def __init__(self, c=10000, eta=3.5, k =500, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.c = c  # heat capacity
        self.eta = eta  # efficiency
        self.k = 500 # adjustment factor
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon
        self.q_table = {}  # Q table
        self.actions = [i for i in range(16, 31)]  # possible temperature setting
        # features' weights
        # Load weights for future use
        if os.path.exists('weights.json'):
            with open('weights.json', 'r') as json_file:
                self.weights = json.load(json_file)
        else:
            # Initialize weights if the file does not exist
            self.weights = {
                'room_weight': 0.01,
                'target_weight': 0.01,
                'time_weight': 0.01
            }

    def Econsumption(self, roomTemperature, targetTemperature, settingTemperature, t):
        # Calculate energy consumption
        # k = -self.c / t * np.log(0.05 * settingTemperature / (roomTemperature - settingTemperature))
        # return k
        E = self.c * abs(roomTemperature - settingTemperature) / self.eta * (1 - np.exp(-self.k / self.c * t))
        return E

    def feature_extraction(self, roomTemperature, targetTemperature, t):
        # Calculate feature value based on current weights
        features = (self.weights['room_weight'] * roomTemperature +
                    self.weights['target_weight'] * targetTemperature +
                    self.weights['time_weight'] * t)
        return features

    def choose_action(self, roomTemperature, targetTemperature, t):
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: select a random action that is less than or equal to targetTemperature
            valid_actions = [action for action in self.actions if action <= targetTemperature]
            return random.choice(valid_actions) if valid_actions else None  # Handle case where no valid actions exist
        else:
            # Exploitation: select the best action based on Q-table
            features = self.feature_extraction(roomTemperature, targetTemperature, t)
            if features not in self.q_table:
                self.q_table[features] = {action: 0 for action in self.actions}
            
            # Filter actions to ensure they are less than or equal to targetTemperature
            valid_actions = [action for action in self.q_table[features] if action <= targetTemperature]
            return max(valid_actions, key=self.q_table[features].get) if valid_actions else None  # Handle case where no valid actions exist

    def q_learning(self, roomTemperature, targetTemperature, t, episodes=100):
        for episode in range(episodes):
            # Initialize state and choose an action
            action = self.choose_action(roomTemperature, targetTemperature, t)
            settingTemperature = action

            # Calculate energy consumption (e)
            e = self.Econsumption(roomTemperature, targetTemperature, settingTemperature, t)
            # Use the negative of e as the reward
            reward = -e  # Smaller e results in higher reward

            # Get the current state features
            current_state = self.feature_extraction(roomTemperature, targetTemperature, t)

            # Initialize the Q-value for the current state if not already done
            if current_state not in self.q_table:
                self.q_table[current_state] = {a: 0 for a in self.actions}

            # Choose the best action for the next state
            next_action = self.choose_action(roomTemperature, targetTemperature, t)
            next_state = self.feature_extraction(roomTemperature, targetTemperature, t)
            
            # Ensure next state is initialized in Q-table
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}

            # Calculate the next Q-value using the next action
            next_k = self.Econsumption(roomTemperature, targetTemperature, next_action, t)
            next_reward = -next_k  # Reward for the next action
            max_next_q = max(self.q_table[next_state].values())

            # Calculate the Q-value delta
            delta = reward + self.gamma * max_next_q - self.q_table[current_state][action]
            # Update Q-value
            self.q_table[current_state][action] += self.alpha * delta

            # Update weights using the delta
            self.update_weights_with_delta(delta, roomTemperature, targetTemperature, t)

        # After training, find and print the best setting temperature
        optimal_setting_temperature = self.choose_action(roomTemperature, targetTemperature, t)
        print(f"Optimal setting temperature for room {roomTemperature}°C, target {targetTemperature}°C, and time {t:.1f} hrs: {optimal_setting_temperature}")

    def update_weights_with_delta(self, delta, roomTemperature, targetTemperature, t):
        # Update weights based on the delta value
        self.weights['room_weight'] += self.alpha * delta * roomTemperature
        self.weights['target_weight'] += self.alpha * delta * targetTemperature
        self.weights['time_weight'] += self.alpha * delta * t
        # Normalize weights to ensure they sum up to 1
        total_weight = self.weights['room_weight'] + self.weights['target_weight'] + self.weights['time_weight']

        # Avoid division by zero
        if total_weight != 0:
            self.weights['room_weight'] /= total_weight
            self.weights['target_weight'] /= total_weight
            self.weights['time_weight'] /= total_weight

    def save_weights(self, filename):
        with open(filename, 'w') as json_file:
            json.dump(self.weights, json_file)

    def train_ac_system(self, episodes=100):
    # Enumerate all possible combinations of roomTemperature, targetTemperature, and t
        for roomTemperature in range(28, 40):  # Room temperature from 28 to 35°C
            for targetTemperature in range(16, min(roomTemperature, 29)):  # Target temperature must be < room temperature and between 16-28°C
                for t in np.arange(0.5, 2.1, 0.5):  # Time from 0.1 to 2 hours in increments of 0.5
                    # Train the Q-learning model for this combination
                    self.q_learning(roomTemperature=roomTemperature, targetTemperature=targetTemperature, t=t, episodes=episodes)

                    # After training, save the weights for this combination
                    self.save_weights("weights.json")
    def predict(self, roomTemperature, targetTemperature):
        # Predict the best settingTemperature based on the trained weights
        e = float('inf')
        min_t = 0
        for t in np.arange(0.5, 2.1, 0.5):
            settingTemperature = self.choose_action(roomTemperature, targetTemperature, t)
            if e > self.Econsumption(roomTemperature, targetTemperature, settingTemperature, t):
                e = self.Econsumption(roomTemperature, targetTemperature, settingTemperature, t)
                min_t = t
        return e, min_t, settingTemperature

ac_system = ACSystem() 
# Training
ac_system.train_ac_system()
# Testing
for _ in range(10):
    testRoomTemperature = np.random.randint(28, 40)
    testTrgetTemperature = np.random.randint(16, min(testRoomTemperature, 29))
    e, t, settingTemperature= ac_system.predict(testRoomTemperature, testTrgetTemperature)
    print(f"Optimal setting temperature for room {testRoomTemperature}°C, target {testTrgetTemperature}°C: time {t:.1f} hrs, setting temperature: {settingTemperature}, energy consumption: {e}.")