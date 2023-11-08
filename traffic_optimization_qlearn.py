import numpy as np

# Set the parameters for the Q-learning algorithm
GRID_SIZE = 10
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
EPISODES = 500
LEARNING_RATE = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Initialize Q-table
Q = np.zeros((N_STATES, N_ACTIONS))

# Define the heuristic function
def bnart_heuristic(state, destination):
    state_x, state_y = state // GRID_SIZE, state % GRID_SIZE
    dest_x, dest_y = destination // GRID_SIZE, destination % GRID_SIZE
    distance = abs(state_x - dest_x) + abs(state_y - dest_y)
    heuristic_value = -distance
    return heuristic_value

# Define the function to get the next state
def get_next_state(state, action):
    x, y = state // GRID_SIZE, state % GRID_SIZE
    if action == 0:   # Up
        x = max(0, x - 1)
    elif action == 1: # Down
        x = min(GRID_SIZE - 1, x + 1)
    elif action == 2: # Left
        y = max(0, y - 1)
    elif action == 3: # Right
        y = min(GRID_SIZE - 1, y + 1)
    next_state = x * GRID_SIZE + y
    return next_state

# Define the function to get the reward
def get_reward(state, next_state, destination):
    if next_state == destination:
        return 100  # Reward for reaching the destination
    else:
        return -1  # Slight negative reward for not being at the destination

# Training loop
destination = np.random.choice(N_STATES)
for episode in range(EPISODES):
    state = np.random.choice(N_STATES)
    done = False
    while not done:
        if np.random.uniform(0, 1) < EPSILON:
            action = np.random.choice(N_ACTIONS)  # Explore action space
        else:
            # Exploit learned values (adding heuristic)
            q_values = Q[state, :] + [bnart_heuristic(state, destination) for _ in range(N_ACTIONS)]
            action = np.argmax(q_values)

        next_state = get_next_state(state, action)
        reward = get_reward(state, next_state, destination)
        done = next_state == destination

        # Q-learning update rule incorporating the heuristic
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :] + bnart_heuristic(next_state, destination)) - Q[state, action])

        state = next_state

# Example use of the trained Q-table to find a route from a start to the destination
start_point = np.random.choice(N_STATES)
current_state = start_point
route = [start_point]
total_reward = 0
steps = 0

while current_state != destination:
    action = np.argmax(Q[current_state, :] + [bnart_heuristic(current_state, destination) for _ in range(N_ACTIONS)])
    next_state = get_next_state(current_state, action)
    reward = get_reward(current_state, next_state, destination)
    total_reward += reward
    current_state = next_state
    route.append(current_state)
    steps += 1
    if steps > 2 * GRID_SIZE:  # Avoid potentially infinite loops
        break

# Output
print(f"Example route from start ({start_point // GRID_SIZE}, {start_point % GRID_SIZE}) to destination ({destination // GRID_SIZE}, {destination % GRID_SIZE}): {route}")
print(f"Average reward: {total_reward / steps:.4f}")
print(f"Steps to destination: {steps}")
