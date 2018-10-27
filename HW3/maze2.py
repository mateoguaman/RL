#Mateo Guaman Castro
#Homework 3
#Tufts University
#Comp 150: Reinforcement Learning
#Exercise 8.4: Dyna-Q+

import numpy as np
import matplotlib.pyplot as plt
import math

class Environment:
    def __init__(self):
        self.maze = self.generate_maze()
    def generate_maze(self):
        maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 100],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 0, 0, 0 ,0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 500, 1, 1, 1, 1, 1]])
        return maze


class Agent:
    def __init__(self, maze, epsilon, gamma, alpha, kappa, n):
        #Initialize maze
        self.maze = maze
        print("Initial maze")
        #Initialize action space A
        self.act = [-1, 0, 1]
        self.A = [(1,0), (-1, 0), (0, 1), (0, -1)]
        #Initialize state space S
        self.row_pos = [i for i in range(self.maze.shape[0])]
        self.col_pos = [i for i in range(self.maze.shape[1])]

        self.S = list(((x,y) for x in self.row_pos for y in self.col_pos))

        #Array to keep track of previously selected states and actions
        self.previously_selected = np.zeros((len(self.S), len(self.A)))
        self.Q, self.Model = self.initialize_Q_Model(self.S, self.A)
        self.lastVisited = np.zeros((len(self.S), len(self.A)))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.kappa = kappa
        self.n = n

    def initialize_Q_Model(self, state_space, action_space):
        '''
        Initializes Q(s, a) anc C(s, a) where s in Z^4 and a in Z^2
        Input:
                State space as list of (row_pos, col_pos, row_vel, col_vel)
                Action space as list of (row_delta, col_delta)
        Returns:
                Dictionary Q with key ((row_pos, col_pos, row_vel, col_vel), (row_delta, col_delta)) and value radnom number from normal distribution
                Dictionary Model with key ((row_pos, col_pos, row_vel, col_vel), (row_delta, col_delta)) and value [0,0] where the elements are [Reward, New_state]
        '''
        state_action_tuples = tuple((x, y) for x in state_space for y in action_space)
        Q = {l:np.random.normal(0,1) for l in state_action_tuples}
        Model = {l:[0, 0] for l in state_action_tuples}
        return Q, Model


    def argmax(self, state):
        '''
        Finds the argmax_a and max_a for a given state. Used for Dyna-Q and Dyna-Q+ with bonus on the reward
        Input:
            State as a tuple (row_pos, col_pos, row_vel, col_vel)
        Returns:
            arg: Index of what action in self.A has the highest value
            highest_Q: Q_value for action with highest value
        '''
        highest_Q = self.Q[state, self.A[0]]
        arg = 0
        for i in range(1, len(self.A)):
            if self.Q[state, self.A[i]] > highest_Q:
                highest_Q = self.Q[state, self.A[i]]
                arg = i
        return arg, highest_Q

    def argmax_bonus(self, state):
        '''
        Finds the argmax_a and max_a for a given state. Used Dyna-Q+ with bonus on the action value
        Input:
            State as a tuple (row_pos, col_pos, row_vel, col_vel)
        Returns:
            arg: Index of what action in self.A has the highest value + bonus
            highest_Q: Q_value for action with highest value
        '''
        highest_Q = self.Q[state, self.A[0]] + self.kappa * math.sqrt(self.lastVisited[self.S.index(state), 0])
        arg = 0
        for i in range(1, len(self.A)):
            if self.Q[state, self.A[i]] + self.kappa * math.sqrt(self.lastVisited[self.S.index(state), i]) > highest_Q:
                highest_Q = self.Q[state, self.A[i]]
                arg = i
        return arg, highest_Q

    def epsilon_greedy(self, state):
        '''
        Selects an action based on the epsilon-greedy method
        Input:
            State as a tuple (row_pos, col_pos, row_vel, col_vel)
        Returns:

        '''
        rand_val = np.random.random()
        if rand_val >= self.epsilon:
            action, _ = self.argmax(state)
        else:
            action = np.random.randint(0,len(self.A))
        return action

    def epsilon_greedy_bonus(self, state):
        '''
        Selects an action based on the epsilon-greedy method
        Input:
            State as a tuple (row_pos, col_pos, row_vel, col_vel)
        Returns:

        '''
        rand_val = np.random.random()
        if rand_val >= self.epsilon:
            action, _ = self.argmax_bonus(state)
        else:
            action = np.random.randint(0,len(self.A))
        return action

    def Q_step(self, state, action, reward, new_state):
        act = self.A[action]
        _, max_a_new_state = self.argmax(new_state)
        self.Q[state, act] += self.alpha * (reward + self.gamma * max_a_new_state - self.Q[state, act])
        return

    def update_model(self, state, action, reward, new_state):
        act = self.A[action]
        self.previously_selected[self.S.index(state), self.A.index(act)] = 1
        self.Model[state, act][0] = reward
        self.Model[state, act][1] = new_state
        return

    def planning_DynaQ(self):
        for _ in range(0,self.n):
            prev_state = self.random_observed_state()
            prev_action = self.random_prev_action(prev_state)
            state = self.S[prev_state]
            action = self.A[prev_action]
            model_sa = self.Model[state, action]
            reward = model_sa[0]
            new_state = model_sa[1]
            self.Q_step(state, prev_action, reward, new_state)
        return

    def planning_DynaQPlus_reward(self):
        for _ in range(0, self.n):
            prev_state = self.random_observed_state()
            prev_action = self.random_prev_action(prev_state)
            state = self.S[prev_state]
            action = self.A[prev_action]
            model_sa = self.Model[state, action]
            reward = model_sa[0] + self.kappa * math.sqrt(self.lastVisited[prev_state, prev_action])
            new_state = model_sa[1]
            self.Q_step(state, prev_action, reward, new_state)
        return

    def planning_DynaQPlus_action(self):
        for i in range(0, self.n):
            prev_state = self.random_observed_state()
            prev_action = self.random_prev_action(prev_state)
            state = self.S[prev_state]
            action = self.A[prev_action]
            if (self.previously_selected[prev_state, prev_action] == 1):
                model_sa = self.Model[state, action]
                reward = model_sa[0]
                new_state = model_sa[1]
            else:
                reward = 0
                new_state = state
            self.Q_step(state, prev_action, reward, new_state)
        return


    def random_observed_state(self):
        indeces = np.where(self.previously_selected == 1)[0]
        return np.random.choice(indeces)

    def random_prev_action(self, state_index):
        indeces = np.where(self.previously_selected[state_index, :] == 1)[0]
        return np.random.choice(indeces)

    def update_last_visited(self, state, action):
        act = self.A[action]
        self.lastVisited += 1
        self.lastVisited[self.S.index(state), self.A.index(act)] = 0
        return




def main():
    steps_per_n_avg = np.ndarray((10,10))
    epsilon = 0.3
    gamma = 0.95
    alpha = 0.7
    kappa = 0.01
    n = 10
    num_steps = 100000
    num_iterations_to_avg = 10

    avg_DynaQ = np.zeros((num_iterations_to_avg, num_steps))
    avg_DynaQPlus_reward = np.zeros((num_iterations_to_avg, num_steps))
    avg_DynaQPlus_action = np.zeros((num_iterations_to_avg, num_steps))

    for k in range(0, num_iterations_to_avg):
        cum_reward = []
        env = Environment()
        agent = Agent(env.maze, epsilon, gamma, alpha, kappa, n)
        episode_info, cum_reward = generate_episode_DynaQ(env, agent, num_steps)
        avg_DynaQ[k, :] = cum_reward


        cum_reward = []
        env = Environment()
        agent = Agent(env.maze, epsilon, gamma, alpha, kappa, n)
        episode_info, cum_reward = generate_episode_DynaQPlus_action(env, agent, num_steps)
        avg_DynaQPlus_action[k, :] = cum_reward



        cum_reward = []
        env = Environment()
        agent = Agent(env.maze, epsilon, gamma, alpha, kappa, n)
        episode_info, cum_reward = generate_episode_DynaQPlus_reward(env, agent, num_steps)
        avg_DynaQPlus_reward[k, :] = cum_reward

    avg_DynaQ = np.mean(avg_DynaQ, axis = 0)
    DynaQ_plot, = plt.plot(np.arange(num_steps), avg_DynaQ, 'r', label='Dyna-Q Learning (n = 4)')
    avg_DynaQPlus_action = np.mean(avg_DynaQPlus_action, axis = 0)
    DynaQAction_plot, = plt.plot(np.arange(num_steps), avg_DynaQPlus_action, 'g', label='Dyna-Q+ Learning, bonus on action (n = 4)')
    avg_DynaQPlus_reward = np.mean(avg_DynaQPlus_reward, axis = 0)
    DynaQReward_plot, = plt.plot(np.arange(num_steps), avg_DynaQPlus_reward, 'b', label='Dyna-Q+ Learning, bonus on reward (n = 4)')

    plt.title("Cumulative reward vs Number of steps")
    plt.ylabel("Cumulative reward")
    plt.xlabel("Number of steps")
    plt.legend(handles=[Q_plot, DynaQ_plot])
    plt.savefig('cumReward.png', dpi=300, bbox_inches='tight')

    plt.close()
    plot_on_track(env, episode_info)
    plt.imshow(env.maze * 5, cmap='hot', interpolation='nearest')
    plt.title(str("Number of steps: " + str(episode_info.shape[0])))
    plt.savefig('figure.png', dpi=300, bbox_inches='tight')


def generate_episode_DynaQ(env, agent, num_steps):
    cumReward = []
    trajectory = np.empty((0, 4))
    done = False
    state = starting_state(env)
    for i in range(0, num_steps):
        if i == 2000:
            change_maze(env, agent)
        action = agent.epsilon_greedy(state)
        new_state, crossed_boundary, crossed_finish, reward = state_transition(env, agent, state, action)
        agent.Q_step(state, action, reward, new_state)
        agent.update_model(state, action, reward, new_state)
        agent.update_last_visited(state, action)
        agent.planning_DynaQ()
        current = np.array([state, action, reward, new_state], ndmin = 2)
        trajectory = np.append(trajectory, current, axis = 0)
        cumReward = update_cum_reward(cumReward, reward)
        state = new_state
        done = crossed_finish
        if (done):
            state = starting_state(env)
    return trajectory, cumReward

def generate_episode_DynaQPlus_reward(env, agent, num_steps):
    cumReward = []
    trajectory = np.empty((0, 4))
    done = False
    state = starting_state(env)
    for i in range(0, num_steps):
        if i == 2000:
            change_maze(env, agent)
        action = agent.epsilon_greedy(state)
        new_state, crossed_boundary, crossed_finish, reward = state_transition(env, agent, state, action)
        agent.Q_step(state, action, reward, new_state)
        agent.update_model(state, action, reward, new_state)
        agent.update_last_visited(state, action)
        agent.planning_DynaQPlus_reward()
        current = np.array([state, action, reward, new_state], ndmin = 2)
        trajectory = np.append(trajectory, current, axis = 0)
        cumReward = update_cum_reward(cumReward, reward)
        state = new_state
        done = crossed_finish
        if (done):
            state = starting_state(env)
    return trajectory, cumReward

def generate_episode_DynaQPlus_action(env, agent, num_steps):
    cumReward = []
    trajectory = np.empty((0, 4))
    done = False
    state = starting_state(env)
    for i in range(0, num_steps):
        if i == 2000:
            change_maze(env, agent)
        action = agent.epsilon_greedy_bonus(state)
        new_state, crossed_boundary, crossed_finish, reward = state_transition(env, agent, state, action)
        agent.Q_step(state, action, reward, new_state)
        agent.update_model(state, action, reward, new_state)
        agent.update_last_visited(state, action)
        agent.planning_DynaQPlus_action()
        current = np.array([state, action, reward, new_state], ndmin = 2)
        trajectory = np.append(trajectory, current, axis = 0)
        cumReward = update_cum_reward(cumReward, reward)
        state = new_state
        done = crossed_finish
        if (done):
            state = starting_state(env)
    return trajectory, cumReward

def starting_state(env):
    possible_starts = np.where(env.maze[-1,:] == 500)[0]
    i = np.random.randint(0,len(possible_starts))
    state = (env.maze.shape[0]-1, possible_starts[i])
    return state


def state_transition(env, agent, state, action):
    act = agent.A[action]
    crossed_boundary = False
    crossed_finish = False
    reward = 0
    temp_state = list(state)

    temp_state[0] += act[0]
    temp_state[1] += act[1]

    if (not in_track(env, temp_state)):
        crossed_boundary = True
        return state, crossed_boundary, crossed_finish, reward
    if in_finish_line(env, temp_state):
        reward = 1
        crossed_finish = True
        return tuple(temp_state), crossed_boundary, crossed_finish, reward
    return tuple(temp_state), crossed_boundary, crossed_finish, reward

def in_track(env, state):
    return in_bounds(env, state) and (env.maze[state[0], state[1]] != 0)

def in_bounds(env, state):
    return (state[0] >= 0 and state[0] < env.maze.shape[0]) and (state[1] >= 0 and state[1] < env.maze.shape[1])

def in_finish_line(env, state):
    return env.maze[state[0], state[1]] == 100

def plot_on_track(env, trajectories):
    for i in range(0, trajectories.shape[0]):
        state = trajectories[i, 0]
        env.maze[state[0], state[1]] += 5

def update_cum_reward(reward_list, current_reward):
    number_rewards = len(reward_list)
    if number_rewards == 0:
        reward_list.append(current_reward)
    else:
        last_reward = reward_list[number_rewards - 1]
        reward_list.append(last_reward + current_reward)
    return reward_list

def change_maze(env, agent):
    env.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 100],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0 ,0, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 500, 1, 1, 1, 1, 1]])
    agent.maze = env.maze


if __name__ == '__main__':
    main()
