import numpy as np
from env.pendulum_env import *

# Constants
DISCOUNT = 0.99
LEARNING_RATE = 0.1
EPSILON = 0.1
EPISODES = 10000
SIMULATION_TIME = 10
HORIZON_LENGTH = int(SIMULATION_TIME/DELTA_T)              # time = 10 seconds divided by delta_t = 0.1
INITIAL_STATE = [0, 0]
DISCRETE_WINDOWS = 50
DISCRETIZED_THETA = np.linspace(0, 2*np.pi, DISCRETE_WINDOWS, endpoint=False)
DISCRETIZED_OMEGA = np.linspace(-6, 6, DISCRETE_WINDOWS)
CONTROLS = [-4, 0, 4]

def get_cost(x, u):
  theta = x[0]
  omega = x[1]
  J_curr = (theta-np.pi)**2 + 0.01*(omega**2) + 0.0001*(u**2)
  return J_curr


def get_positions_in_qtable(state):
  '''INPUT: state = [theta, omega]
    OUTPUT: index_in_discretized_theta = index of closest discretized theta in q-table
            index_in_discretized_omega = index of closest discretized omega in q-table
  '''
  theta = state[0]
  omega = state[1]

  # we can find the index of the closest element in the set of discretized states
  index_in_discretized_theta = np.argmin(np.abs(DISCRETIZED_THETA - theta))
  index_in_discretized_omega = np.argmin(np.abs(DISCRETIZED_OMEGA - omega))

  return index_in_discretized_theta, index_in_discretized_omega


def get_policy_and_value_function(q_table):
  '''INPUT: q_table
    OUTPUT: opt_policy = An array of size (DISCRETE_WINDOWS, DISCRETE_WINDOWS)
                        with each index corresponding to the optimal control
                        input for that state
            value_function = An array of size (DISCRETE_WINDOWS, DISCRETE_WINDOWS)
                            with each index corresponding to the value function
                            for that state
  '''
  opt_policy = np.zeros([DISCRETE_WINDOWS, DISCRETE_WINDOWS])
  value_function = np.zeros_like(opt_policy)

  for i in range(len(DISCRETIZED_THETA)):
    for j in range(len(DISCRETIZED_OMEGA)):
      # Find control input with minimum Q-value i.e. optimal control value
      cntr_idx = np.argmin(q_table[i][j])
      u_opt = CONTROLS[cntr_idx]
      opt_policy[i, j] = u_opt
      value_function[i, j] = np.min(q_table[i][j])

  return opt_policy, value_function


def get_value_function_through_network(learned_q_network, input):
  value_function = learned_q_network.predict(input)
  return value_function


def get_policy_through_network(learned_q_network):
  state = INITIAL_STATE
  opt_policy = []
  for n in range(HORIZON_LENGTH):
    for u in CONTROLS:
      input = np.append(state, [u], axis=0)
      input = np.expand_dims(input, axis=0)  # Convert to NumPy array and add batch dimension
      #print(input, input.shape)
      op = q_network.predict(input)
      all_Q.append(op)

    all_Q = np.array(all_Q)
    u_opt = CONTROLS[np.argmin(all_Q)]
    opt_policy.append(u_opt)

    state = get_next_state(state, u_opt)

    return opt_policy


def q_learning(q_table):
  '''INPUT: q_table = initialized unlearned Q-table
    OUTPUT: q_table = learned Q-table
            cost_per_episode = cost of the policy for each episode

  '''
  cost_per_episode = []
  for episode in range(EPISODES):
    # Initialize states every episode
    x = INITIAL_STATE
    x_d = get_positions_in_qtable(x)
    cost_of_episode = 0

    # Update the Q-table
    for i in range(HORIZON_LENGTH):

      # Epsilon greedy policy
      if np.random.random() > EPSILON:
        cntr_idx = np.argmin(q_table[x_d[0]][x_d[1]])
      else:
        cntr_idx = np.random.randint(0, 2.1)

      # Calculating control input for current stage
      control_curr = CONTROLS[cntr_idx]

      # Calculating stage cost
      cost_curr = get_cost(x, control_curr)

      # Calculating next state
      x_next = get_next_state(x, control_curr)
      x_d_next = get_positions_in_qtable(x_next)

      # Extracting Q-values for current and next state
      min_Q_next = np.min(q_table[x_d_next[0]][x_d_next[1]])
      Q_curr = q_table[x_d[0]][x_d[1]][cntr_idx]

      #TD Error
      TD_error = cost_curr + DISCOUNT*min_Q_next - Q_curr

      # Update step
      Q_curr_updated = Q_curr + LEARNING_RATE*TD_error
      q_table[x_d[0]][x_d[1]][cntr_idx] = Q_curr_updated

      # Adding cost per stage to get total cost of the episode
      cost_of_episode = cost_of_episode + DISCOUNT*get_cost(x, control_curr)

      # Passing the baton
      x_d = x_d_next
      x = x_next

    # Recording the cost of each episode
    cost_per_episode.append(cost_of_episode)

    # Indicates whether the desired state has been reached or not
    #if (x[0] > 3 and x[0] < 3.2) and (x[1] >= 0 and x[1] < 0.1):
    #  print('We reached on episode', episode)

  return q_table, cost_per_episode

