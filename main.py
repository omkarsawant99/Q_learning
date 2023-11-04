from q_learning.q_learning import *
from env.pendulum_env import *
import matplotlib.pyplot as plt
import numpy as np

# Function which executes the policy
def policy(x):
  th, w = get_positions_in_qtable(x)
  u = POLICY[th, w]
  return u

# Initialize Q-table
q_table = np.random.uniform(0, 2, size=(DISCRETE_WINDOWS, DISCRETE_WINDOWS, len(CONTROLS)))

# Q-learning
learned_q_table, cost_per_episode = q_learning(q_table)

# Get the optimal policy and value function from the learned Q-table
POLICY, VALUE_FUNCTION = get_policy_and_value_function(learned_q_table)

# Simulate the optimal policy
t, x, u = simulate(INITIAL_STATE, policy, SIMULATION_TIME)

# Plot the cost per episode
plt.figure()
plt.plot(cost_per_episode)
plt.legend(['Cost'])
plt.xlabel('Episodes')

# Visualize the results
animate_robot(x)

