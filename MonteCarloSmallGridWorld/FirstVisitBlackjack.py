"""
    Replicate the graphs in Figure 5.1 by implementing First-visit Monte Carlo prediction. 
    Hints:
    Implement the environment as a 2D GridWorld. One dimension is the 10 possible up cards of the dealer. The other dimension has 20 values: 2-11 represent the player's total when the player has a usable ace, and 12-21 represent the player's total when the player has a usable ace. 
    When the environment restarts it should give the player enough cards to have a total of at least 12 or an ace.
"""
"""
The first-visit MC method estimates v⇡ (s) as the average of the returns following
first visits to s, whereas the every-visit MC method averages the returns following all
visits to s.
"""
"""
decision is made based on 3 things
    - current sum (12-21) -- below 12 are forced to hit
    - dealer's one showing card (upcard) (ace-10) ace = 1
    - whether or not he holds a usable ace

state is a tuple of (player_sum, dealer_upcard, usable_ace)
"""
import matplotlib
import numpy as np
import sys

