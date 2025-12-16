this project solves exercise 6.9 from the book by first implementing a simple windy gridworld and later extending it to use King's moves.

The envs folder has the two environments - one with 4 actions, the other with 9 actions.
The file run_windy_gridworld is the solution which makes the environments, runs e_sarsa.py, and saves the plots from the results.

e_sarsa.py runs the tabular epsilon-greedy sarsa algorithm which returns the updated q table and the time steps for plotting purposes.

The plots windy_gridworld and king_windy_gridworld highlight how king's moves implementation performs better, especially in the later episodes, eventually reaching 175 episodes in 6000 time stepse rather than the 7000 time steps of the simpler implementation. This is to indicate that last episodes reach goal state in much fever steps than the first implementation with 4 actions. This makes sense as the extra diagonial moves helps us optimize the path and minimize the number of steps.