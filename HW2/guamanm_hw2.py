#Mateo Guaman Castro
#Homework 2
#Tufts University
#Comp 150: Reinforcement Learning
#Exercise 5.8: Racetrack, from Reinforcement Learning: An Introduction, by Sutton and Barto


import numpy as np
import matplotlib.pyplot as plt



class Environment:
    def __init__(self):
        self.track = self.generate_track()
    def generate_track(self):
        track = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        return track

    def generate_track1(self):
        track = np.array([[0, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 1, 1, 2],
                          [1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 0, 0, 0]])
        return track
    def generate_track2(self):
        track = np.array([[1, 2],
                          [1, 2],
                          [1, 0],
                          [1, 0]])
        return track

    def generate_track3(self):
        track = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2]])
        print(track)
        return(track)


class Agent:
    def __init__(self, track, epsilon):
        self.track = track
        print(track)
        self.act = [-1, 0, 1]
        self.A = list(((x,y) for x in self.act for y in self.act))
        print(self.A)
        self.row_pos = [i for i in range(self.track.shape[0])]
        self.col_pos = [i for i in range(self.track.shape[1])]
        self.row_vel = [i for i in range(5)] #Change to 5
        self.col_vel = [i for i in range(5)]
        self.S = list(((v,w,x,y) for v in self.row_pos for w in self.col_pos for x in self.row_vel for y in self.col_vel))
        self.Q, self.Returns = self.initialize_Q_Returns(self.S, self.A)
        self.epsilon = epsilon
        self.pi = self.initialize_pi(self.S, len(self.A), self.epsilon)

    def initialize_Q_Returns(self, state_space, action_space):
        '''
        Initializes Q(s, a) anc C(s, a) where s in Z^4 and a in Z^2
        '''
        state_action_tuples = tuple((x, y) for x in state_space for y in action_space)
        Q = {l:np.random.normal(0,1) for l in state_action_tuples}
        Returns = {l:[] for l in state_action_tuples}
        return Q, Returns

    def initialize_pi(self, state_space, numActions, epsilon):
        pi = {i:np.ones(numActions, dtype=float) *  epsilon/numActions for i in state_space}
        for i in state_space:
            pi[i][np.argmax(pi[i])] += 1 - epsilon
        return pi

    def argmax(self, state):
        #print("Inside argmax")
        highest_Q = self.Q[state, self.A[0]]
        arg = 0
        for i in range(1, len(self.A)):
            if self.Q[state, self.A[i]] > highest_Q:
                highest_Q = self.Q[state, self.A[i]]
                arg = i
        return arg

def main():
    env = Environment()
    epsilon = 0.3
    agent = Agent(env.track, epsilon)
    gamma = 0.9
    noise = 0.1

    optimal_solution = None

    for i in range(0, 10000):
        #agent.epsilon = min(10/(i+1), 1)
        episode_info = generate_episode(env, agent, noise)
        if optimal_solution is None or episode_info.shape[0] <= optimal_solution.shape[0]:
            optimal_solution = episode_info
        if i % 1 == 0:
            print("Episode " + str(i))
            print(episode_info)
            print(episode_info.shape)
         #Credit to Denny Britz (https://github.com/dennybritz) for this Python magic
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode_info])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_occurence_idx = next(i for i,x in enumerate(episode_info) if x[0] == state and x[1] == action)
            G = sum([x[2]*(gamma**i) for i,x in enumerate(episode_info[first_occurence_idx:])])
            #End of Python magic
            sa_pair_descriptive = (state, agent.A[action])
            agent.Returns[sa_pair_descriptive].append(G)
            agent.Q[sa_pair_descriptive] = sum(agent.Returns[sa_pair_descriptive]) / len(agent.Returns[sa_pair_descriptive])
            a_star = agent.argmax(state)
            for i in range(0, len(agent.pi[state])):
                if i == a_star:
                    agent.pi[state][i] =  1 - epsilon + epsilon/len(agent.A)
                else:
                    agent.pi[state][i] = epsilon/len(agent.A)
    print(episode_info)
    print(episode_info.shape)

    episode_info = generate_episode(env, agent, 0)
    plot_on_track(env, optimal_solution)
    plt.imshow(env.track * 5, cmap='hot', interpolation='nearest')
    plt.title(str("Number of steps: " + str(optimal_solution.shape[0])))
    plt.savefig('figure.png', dpi=300, bbox_inches='tight')



def generate_episode(env, agent, noise):
    trajectory = np.empty((0, 4))
    done = False
    state = starting_state(env)
    while (not done):
        rand_val = np.random.random()
        if rand_val < noise:
            action = agent.A.index((0,0))
        else:
            action = np.random.choice(len(agent.A), p = agent.pi[state])
        new_state, crossed_boundary, crossed_finish, reward = update_state(env, agent, state, action)
        current = np.array([state, action, reward, new_state], ndmin = 2)
        trajectory = np.append(trajectory, current, axis = 0)
        state = new_state
        done = crossed_finish
    return trajectory

def starting_state(env):
    possible_starts = np.where(env.track[-1,:] == 1)[0]
    i = np.random.randint(0,len(possible_starts))
    state = (env.track.shape[0]-1, possible_starts[i], 0, 0)
    return state


def update_state(env, agent, state, action):
    act = agent.A[action]
    crossed_boundary = False
    crossed_finish = False
    reward = -1
    temp_state = list(state)

    temp_state[2] = min(max(temp_state[2] + act[0], 0), 4)
    temp_state[3] = min(max(temp_state[3] + act[1], 0), 4)
    if (temp_state[2] == 0 and temp_state[3] == 0):
        temp_state[2] += 1

    for x in range(0, temp_state[2]):
        temp_state[0] -= 1
        if (not in_track(env, temp_state)):
            crossed_boundary = True
            return starting_state(env), crossed_boundary, crossed_finish, reward
        if in_finish_line(env, temp_state):
            reward = 0
            crossed_finish = True
            return tuple(temp_state), crossed_boundary, crossed_finish, reward
    for y in range(0, temp_state[3]):
        temp_state[1] += 1
        if (not in_track(env, temp_state)):
            crossed_boundary = True
            return starting_state(env), crossed_boundary, crossed_finish, reward
        if in_finish_line(env, temp_state):
            reward = 0
            crossed_finish = True
            return tuple(temp_state), crossed_boundary, crossed_finish, reward

    return tuple(temp_state), crossed_boundary, crossed_finish, reward

def in_track(env, state):
    return in_bounds(env, state) and (env.track[state[0], state[1]] != 0)

def in_bounds(env, state):
    return (state[0] >= 0 and state[0] < env.track.shape[0]) and (state[1] >= 0 and state[1] < env.track.shape[1])

def in_finish_line(env, state):
    return env.track[state[0], state[1]] == 2

def plot_on_track(env, trajectories):
    for i in range(0, trajectories.shape[0]):
        state = trajectories[i, 0]
        env.track[state[0], state[1]] = 5

if __name__ == '__main__':
    main()
