Final Project is stored in the MountainCar folder which solves the Mountain Car problem from https://gymnasium.farama.org/environments/classic_control/mountain_car/ in 3 different ways:
* tabular epsilon-greedy SARSA with discrete observation space using Gymnasium's MountainCar env.
* semi-gradient 1-step SARSA with continuous observation space using tile coding.
* semi-gradient 8-step SARSA with continuous observation space using tile coding. 
For the last two solutions, I implemented my own environment. Action space is Discrete(3).

resources:
OpenAI Gym Spaces http://deepwiki.com/openai/gym/2.3-spaces
Matplotlib docs: https://matplotlib.org/stable/tutorials/pyplot.html
matplotlib heatmap: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
Registering environments: https://gymnasium.farama.org/api/registry/
numpy: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
numpy bin discretization with digitize(): https://numpy.org/doc/stable/reference/generated/numpy.digitize.html