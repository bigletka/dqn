import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from visuals import visualize_power_usage
import uvicorn


MODEL_DIR = '/mnt/data/models/'


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = 5  # Number of features: cpuload, allocatedcpu, freemem, availmem, powerusage

# Initialize global variables
dqn = None
optimizer = None
criterion = nn.MSELoss()
gamma = 0.99
epsilon = 1.0  # Start with high exploration
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
learning_rate_decay = 0.999
memory = deque(maxlen=10000)
batch_size = 32
decision_count = 0


# Function to generate a unique model filename based on the number of nodes (actions)
def get_model_filename(action_size):
    return f'dqn_model_{action_size}_actions.pth'


# Function to save the model
def save_model(model, optimizer, filepath, epsilon, decision_count):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'decision_count': decision_count,
        'learning_rate': learning_rate,
        'learning_rate_decay': learning_rate_decay,
    }, filepath)


# Function to load the model
def load_model(filepath, state_size, action_size):
    global learning_rate
    model = DQN(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint.get('epsilon', 1.0)  # Default to 1.0 if not in the file
        decision_count = checkpoint.get('decision_count', 0)  # Default to 0 if not in the file
        learning_rate = checkpoint.get('learning_rate', 0.001)
        learning_rate_decay = checkpoint.get('learning_rate_decay', 0.999)
        print(f"Model loaded from {filepath}")
        return model, optimizer, epsilon, decision_count, learning_rate, learning_rate_decay, False  # Model was loaded, so don't reset global parameters
    else:
        print(f"No existing model found. Creating a new model.")
        return model, optimizer, 1.0, 0, 0.001, 0.999, True  # New model created, global parameters should be reset


# Modified initialize_dqn function
def initialize_dqn(action_size, state_size=5):
    global dqn, optimizer, epsilon, decision_count, learning_rate, learning_rate_decay
    
    # Ensure the model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_filename = get_model_filename(action_size)
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    # Check if the model is already loaded in memory
    if dqn is None:
        dqn, optimizer, epsilon, decision_count, learning_rate, learning_rate_decay, is_new_model = load_model(model_path, state_size, action_size)
        
        if is_new_model:
            # Reset global parameters if a new model is created
            epsilon = 1.0  # Reset exploration rate
            decision_count = 0
            learning_rate = 0.001
            learning_rate_decay = 0.999
            memory.clear()  # Clear the memory to start fresh
    elif dqn.fc3.out_features != action_size:
        dqn, optimizer, epsilon, decision_count, learning_rate, learning_rate_decay, is_new_model = load_model(model_path, state_size, action_size)
        
        if is_new_model:
            # Reset global parameters if a new model is created
            epsilon = 1.0  # Reset exploration rate
            decision_count = 0
            learning_rate = 0.001
            learning_rate_decay = 0.999
            memory.clear()  # Clear the memory to start fresh

    # Save the newly created or loaded model for future use
    save_model(dqn, optimizer, model_path, epsilon, decision_count)


def select_action(state, action_size):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            return torch.argmax(dqn(torch.FloatTensor(state))).item()


def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def train_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = dqn(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save the model after training
    model_filename = get_model_filename(dqn.fc3.out_features)
    model_path = os.path.join(MODEL_DIR, model_filename)
    save_model(dqn, optimizer, model_path, epsilon, decision_count, learning_rate, learning_rate_decay)
    print(f"Model saved to {model_path}")



app = FastAPI()


class Stats(BaseModel):
    vmmid: List[float]
    cpuload: List[float]
    allocatedcpu: List[float]
    freemem: List[float]
    availmem: List[float]
    powerusage: List[float]


power_usage_history = []
reward_history = []


@app.post("/predict")
def predict(stats: Stats):
    global epsilon, decision_count, learning_rate

    action_size = len(stats.cpuload)
    initialize_dqn(action_size)

    state = np.array([stats.cpuload, stats.allocatedcpu, stats.freemem, stats.availmem, stats.powerusage]).T.astype(np.float32)

    power_usage_history.append(stats.powerusage)

    actions = []

    for s in state:
        action = select_action(s, action_size)
        actions.append(action)

        next_state = s
        reward = -s[-1]
        done = False
        reward_history.append(reward)
        store_experience(s, action, reward, next_state, done)
        train_model()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        decision_count += 1

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * (learning_rate_decay ** decision_count)

     # After making a decision and updating the state, visualize
    visualize_power_usage(decision_count, power_usage_history, action_size, reward_history)
    # Return the action (node ID) for the last state in the batch
    return {"action": stats.vmmid[actions[-1]]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
