"""
    Replicate the graphs in Figure 5.1 by implementing First-visit Monte Carlo prediction. 
    Hints:
    Implement the environment as a 2D GridWorld. One dimension is the 10 possible up cards of the dealer. The other dimension has 20 values: 2-11 represent the player's total when the player has a usable ace, and 12-21 represent the player's total when the player has a usable ace. 
    When the environment restarts it should give the player enough cards to have a total of at least 12 or an ace.
"""

def first_visit_blackjack():
    def __init__():
        self._states = (player_sum, dealer_upcard, usable_ace)
        self._actions = 


    def policy(players_hand):
        return 'stick' if sum_hand(player_hand) >= 20 else 'hit'