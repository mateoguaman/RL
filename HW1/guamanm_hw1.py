"""
K-Armed Bandits in the context of non-stationary problems
Author: Mateo Guaman Castro
Class:  COMP150 Special Topics: Reinforcement Learning
        Professor Jivko Sinapov
        Tufts University
Homework #1

This assignment is based on Exercise 2.5 of Reinforcement Learning: An Introduction, 2nd Edition, by Sutton and Barto.

For more information on this assignment, refer to the PDF writeup at https://github.com/mateoguaman/RL/blob/master/HW1/HW1_kArmedBandits.pdf
"""

import numpy as np
import matplotlib.pyplot as plt


class k_armed_bandit:
    """
    Class for the environment of the agent.

    Inputs:
        k -- Number of arms
        reward_var -- Variance used when generating rewards

    The purpose of this class is to set up a non-stationary environment for the reinforcement learning agent.
    """
    def __init__(self, k, reward_var):
        self.k = k
        self.reward_var = reward_var
        self.q_star = np.zeros((1, k))
        self.R = np.zeros((1, k))
        self.t_step = 0

    def starting_q_star(self):
        """
        Purpose: Generate initial expected action values, which are all the same for all arms
        """
        initial_val = 0
        self.q_star = np.zeros((1,self.k)) + initial_val

    def add_random_walks(self):
        """
        Purpose: Make the expected action values non-stationary
        """
        self.q_star += np.random.normal(0, 0.01, (1,self.k))

    def generate_reward(self, i):
        """
        Purpose: Generate rewards for all actions based on a normal distribution
        """
        self.R = np.random.normal(self.q_star, self.reward_var)
        return self.R[(0,i)]

    def print_status(self):
        """
        Purpose: Display class values for debugging purposes
        """
        print("Status of Environment")
        print("Current q_star: " + str(self.q_star))
        print("Current R vector: " + str(self.R))
        print("Current t_step: " + str(self.t_step))

class Agent:

    """
    Class for the learning agent.

    Inputs:
        bandit -- Instance of class k_armed_bandit
        k -- Number of arms
        epsilon -- Epsilon value for epsilon-greedy
        alpha -- Alpha parameter for recency-weighted methods
        time_steps -- Number of time steps per run

    The purpose of this class is to setup the agent, as well as provide action-selection and action-value-estimation (Q) methods.
    """
    def __init__(self, bandit, k, epsilon, alpha, time_steps):

        self.bandit = bandit
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.time_steps = time_steps
        self.Q = np.zeros((1,k))
        self.N = np.zeros((1,k))
        self.action_choice = None
        self.reward = None

        #Benchmark variables
        self.counter = 0
        self.reward_vec = np.zeros((1,self.time_steps))
        self.optimal_vec = np.zeros((1,self.time_steps))

    def epsilon_greedy(self, avg_method="RUNNING"):
        """
        Epsilon-greedy action-selection method
        Inputs:
            avg_method -- "RUNNING" or "WEIGHTED" to choose between samples average or recency-weighted average, respectively
        """
        rand_val = np.random.random()
        if rand_val >= self.epsilon:
            self.action_choice = np.argmax(self.Q)
        else:
            self.action_choice = np.random.randint(0,self.k)
        #Generate reward
        self.reward = self.bandit.generate_reward(self.action_choice)
        #Increase action count for action selected
        self.N[(0,self.action_choice)] += 1

        if avg_method == "RUNNING":
            self.running_average()
        else:
            self.weighted_average()

    def running_average(self):
        """
        Purpose: Compute running samples average
        """
        self.Q[(0,self.action_choice)] += 1/self.N[(0,self.action_choice)] * (self.reward - self.Q[(0,self.action_choice)])

    def weighted_average(self):
        """
        Purpose: Compute recency-weighted average
        """
        self.Q[(0,self.action_choice)] += self.alpha * (self.reward - self.Q[(0,self.action_choice)])

    def print_status(self):
        """
        Purpose: Print status of class for debugging purposes
        """
        print("Status of Agent")
        print("Epsilon: " + str(self.epsilon))
        print("Current Q: " + str(self.Q))
        print("Current N: " + str(self.N))
        print("Current action_choice: " + str(self.action_choice))
        print("Current reward: " + str(self.reward))

    def benchmark(self):
        """
        Purpose: Benchmark the agent's selections
        """
        self.reward_vec[(0,self.counter)] = self.reward
        self.optimal_vec[(0, self.counter)] = (self.action_choice == np.argmax(self.bandit.q_star))
        self.counter += 1


def experiment(time_steps, epsilon, k, alpha, avg_method, reward_var):
    """
    Do one run of the experiment (k timesteps)
    Inputs:

        time_steps -- Number of time steps per run
        epsilon -- Epsilon value for epsilon-greedy
        k -- Number of arms
        alpha -- Alpha parameter for recency-weighted methods
        avg_method -- "RUNNING" or "WEIGHTED" to choose between samples average or recency-weighted average, respectively
        reward_var -- Variance used when generating rewards

    Outputs:

        agent.reward_vec -- Vector of all obtained rewards over one run
        agent.optimal_vec -- Boolean vector of whether the agent chose the optimal action over one run

    """
    bandit = k_armed_bandit(10, reward_var)
    agent = Agent(bandit, k, epsilon, alpha, time_steps)
    bandit.starting_q_star()
    for i in range(time_steps):

        agent.epsilon_greedy(avg_method)
        agent.benchmark()
        bandit.add_random_walks()
        bandit.t_step += 1

    return agent.reward_vec, agent.optimal_vec


def run_n_plot(time_steps, runs, k, epsilon, alpha, reward_var, avg_type, axis1, axis2):
    """
    Perform "runs" (number of runs) runs and plot

    Inputs:
        time_steps -- Number of time steps per run
        runs -- Number of runs
        k -- Number of arms
        epsilon -- Epsilon value for epsilon-greedy
        alpha -- Alpha parameter for recency-weighted methods
        reward_var -- Variance used when generating rewards
        avg_type -- "RUNNING" or "WEIGHTED" to choose between samples average or recency-weighted average, respectively
        axis1 -- axis for subplot 1
        axis2 -- axis for subplot 2
    """
    total_rewards = np.zeros((1, time_steps))
    total_optimal_vec = np.zeros((1, time_steps))
    t_vec = np.arange(time_steps)
    if avg_type == "WEIGHTED":
        label_str = "Recency-weighted average"
    else:
        label_str = "Normal average"

    for j in range(runs):
        run_rewards, run_optimal_vec = experiment(time_steps, epsilon, k, alpha, avg_type, reward_var)
        total_rewards += run_rewards
        total_optimal_vec += run_optimal_vec
    total_rewards /= runs
    total_optimal_vec /= runs

    axis1.plot(t_vec, total_rewards[0,:], label=label_str)
    axis1.legend()
    axis2.plot(t_vec, total_optimal_vec[0,:], label=label_str)
    axis2.legend()


def experiment_run(time_steps, runs, k, epsilon, alpha, reward_var, filename):
    """
    Runs one whole experiment of a certain number of time steps and a certain number of runs and saves the plots from the experiment.

    Inputs:
        time_steps -- Number of time steps per run
        runs -- Number of runs
        k -- Number of arms
        epsilon -- Epsilon value for epsilon-greedy
        alpha -- Alpha parameter for recency-weighted methods
        reward_var -- Variance used when generating rewards
        filename -- Filename of where the plots will be stored
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
    plt.suptitle(r'$\epsilon$: ' + str(epsilon) + r', $\alpha$: ' + str(alpha) + r', Reward $\sigma^2$: ' + str(reward_var))

    ax1.set_title("Average rewards")
    ax2.set_title("Optimal %")
    ax1.grid()
    ax2.grid()

    run_n_plot(time_steps, runs, k, epsilon, alpha, reward_var, "RUNNING", ax1, ax2)

    print("Done w/ running average!")

    run_n_plot(time_steps, runs, k, epsilon, alpha, reward_var, "WEIGHTED", ax1, ax2)

    print("Done w/ weighted average!")

    #Save to file
    fig.savefig(filename, bbox_inches='tight', dpi=300)

    plt.close(fig)


def main():
    time_steps = 10000
    runs = 2000
    k = 10

    #Experiment 1
    epsilon = 0
    alpha = 0.1
    reward_var = 1
    experiment_run(time_steps, runs, k, epsilon, alpha, reward_var, "experiment1.png")

    #Experiment 2
    epsilon = 0.1
    alpha = 0.1
    reward_var = 1
    experiment_run(time_steps, runs, k, epsilon, alpha, reward_var, "experiment2.png")

    #Experiment 3
    epsilon = 0.1
    alpha = 0.5
    reward_var = 1
    experiment_run(time_steps, runs, k, epsilon, alpha, reward_var, "experiment3.png")

    #Experiment 4
    epsilon = 0.1
    alpha = 0.1
    reward_var = 0.1
    experiment_run(time_steps, runs, k, epsilon, alpha, reward_var, "experiment4.png")

if __name__ == '__main__':
    main()
