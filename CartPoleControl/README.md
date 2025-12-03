### Problem at hand
CartPole is a small cart that moves in a straight line. A thin pole is attached to the cart. The pole is affected by gravity and falls. Our role is to teach an RL agent to move the cart so that the pole stays straight up.

For the pole to stay vertical, the cart must move left or right. The pole that is swinging on the cart has continuous observations (cart position, speed, pole angle, angular velocity). For manual calculation, these observations have been discretized into 4 states (S0-S3).

### Why Q-Learning
The agent doesn’t need to understand the full environment. It needs to be consistent. With each step, its decisions get a little better. Every time the agent takes an action and receives feedback. After this step, it updates its internal memory – the Q-table. It’s like a person who adjusts their habits after each small success or failure.
I have used tabular Q-Learning which, combined with Cart Pole environment, makes it easier to see how Q-Learning works. However, Tabular Q-Learning requires discretization of the feature space as it can't handle Cart Pole's continuous states.

### Q-Learning Algorithm
<img width="877" height="359" alt="image" src="https://github.com/user-attachments/assets/0781116a-96a5-44b9-8c17-81170456ff2e" />

### Cart Pole Environment
https://gymnasium.farama.org/environments/classic_control/cart_pole/
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

### Starting State
All observations are assigned a uniformly random value in (-0.05, 0.05)

### Episode End
The episode ends if any one of the following occurs:

Termination: Pole Angle is greater than ±12°

Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)

Truncation: Episode length is greater than 500 (200 for v0)

### Action Space
The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.

0: Push cart to the left

1: Push cart to the right

Note: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

## Observation Space
State Space (Continuous, 4 Dimensions)

Each observation is a real-valued vector: [𝑥, 𝑥˙, 𝜃, 𝜃˙]

Cart Position (𝑥)
Horizontal position of the cart on the track.

Cart Velocity (𝑥˙)
Rate of movement of the cart left or right.

Pole Angle (𝜃)
Angle of the pole relative to vertical (0 = perfectly upright).

Pole Angular Velocity (𝜃˙)
Rate at which the pole angle is changing.

## Discretization
Using uniform bins, clipped velocity bounds, and a moderately large number of bins improves state representation enough for convergence
Number of bins = bias vs. variance tradeoff.

Few bins → high bias → underfitting

Many bins → high variance → slow learning, sparse Q-table


## Rewards
Since the goal is to keep the pole upright for as long as possible, by default, a reward of +1 is given for every step taken, including the termination step. The default reward threshold is 500 for v1 (the one we use in this project) and 200 for v0 due to the time limit on the environment.

If sutton_barto_reward=True, then a reward of 0 is awarded for every non-terminating step and -1 for the terminating step. As a result, the reward threshold is 0 for v0 and v1

## Takeaways
cartpole.py has the final solution with test episodes yielding rewards between 150-250. first_implementation.py yielded no rewards higher than 150 and never consistenly, sometimes going as low as 12 rewards an episode after being trained.
The difference between cartpole.py and first_implementation.py are:
* Cartpole.py used np for Q_table whereas first_implementation.py used a dictionary. Benefit: Much faster training → more episodes → better convergence
* First_implementation.py was a manual attempt to discretizing states and normalizing bins. cartpole.py uses np.digitize(). Benefit: makes bins uniform and accurate, better discretization and correct state representation
* Better epsilon decay schedule. Epsilon_decay 0.9995 instead of 0.99995
* Higher number of episodes. 25,000 instead of 10,000
Overall, cartpole.py takes longer but yields a better trained agent.


Resources used:
https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
https://gymnasium.farama.org/environments/classic_control/cart_pole/
https://www.nickjalbert.com/reading/2020/06/15/tabular-q-learning-for-cartpole.html
https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df
https://www.reinforcementlearningpath.com/step-by-step-tutorial-q-learning-example-with-cartpole/
