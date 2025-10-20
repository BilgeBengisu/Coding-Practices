"""
tutorial used: https://towardsdatascience.com/introduction-to-reinforcement-learning-and-solving-the-multi-armed-bandit-problem-e4ae74904e77/

Author: Bilge Akyol

This module defines a Bandit class to simulate multi-armed bandit problems.

Each Bandit object represents one slot machine (one arm).

mu = the true mean reward (expected payout) of that arm.

sigma = the standard deviation (how noisy or variable the rewards are). the randomness which forces exploration.

Default is 1, meaning each pull gives a normally distributed reward around mu with noise.
"""

from typing import Callable 
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, mu, sigma=1):
        self.mu = mu # true mean reward
        self.sigma = sigma # standard deviation of reward distribution
    
    # pull behavior 
    # returns one random sample (reward) drawn from a Normal(μ, σ) distribution.
    # makes each pull slightly different, modeling uncertainty and noise
    def __call__(self):
        return np.random.normal(self.mu, self.sigma)
    
def initialize_bandits() -> list[Bandit]:
    # passing the mu/true mean values 
    return [
        Bandit(0.2),
        Bandit(-0.8),
        Bandit(1.5),
        Bandit(0.4),
        Bandit(1.1),
        Bandit(-1.5),
        Bandit(-0.1),
        Bandit(1),
        Bandit(0.7),
        Bandit(-0.5),
    ]


def simple_crit(Q: np.ndarray, N: np.ndarray, t: int, eps: float) -> int:
    return (
        int(np.argmax(Q))
        if np.random.rand() < 1 - eps
        else np.random.randint(Q.shape[0])
    )

"""
 Q: estimated values for each bandit arm
 N: number of times each arm has been pulled
 t: current time step/iteration count
 eps: exploration probability, 1-eps is exploitation probability
"""
def epsilon_action(Q: np.ndarray, N:np.ndarray, t:int, eps:float) -> int:
    return(
        int(np.argmax(Q)) # return the greedy action
        if np.random.rand() < 1 - eps # if random number is less than 1-eps so with 1-eps probability
        else np.random.randint(Q.shape[0]) # else return a random action (exploration)
    )
    

"""
 passing a list of Bandit objects
 and the epsilon action function
"""

def bandit_solver(
    bandits: list[Bandit], epsilon_action: Callable[[np.ndarray, np.ndarray, int], int]) -> np.ndarray:

    # set up estimated values and counts
    Q = np.zeros((len(bandits)))
    N = np.zeros((len(bandits)))

    num_steps = 10000

    rewards = []
    for t in range(num_steps):
        arm_index = epsilon_action(Q, N, t)
        reward = bandits[arm_index]()
        rewards.append(reward)
        N[arm_index] = N[arm_index] + 1
        Q[arm_index] = Q[arm_index] + 1 / N[arm_index] * (reward - Q[arm_index]) # q value update formula: old estimate + (1/n)(reward - old estimate)

    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    # np.cumsum(rewards) gives the cumulative total reward up to each time step.
    # Dividing by the time steps 1, 2, 3, … gives the average reward over time.
    # This lets us plot or compare how quickly the algorithm learns.


def main() -> None:
    bandits = initialize_bandits()
    epss = [0, 0.01, 0.1]
    reward_averages = [
        bandit_solver(bandits, lambda q, n, t: epsilon_action(q, n, t, eps)) 
        for eps in epss
    ]

    # plotting results and averages
    colors = ["r-", "b-", "g-"]
    for idx, reward_average in enumerate(reward_averages):
        plt.plot(
            range(len(reward_average)), reward_average, colors[idx], label=epss[idx]
        )
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()