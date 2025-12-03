'''
blackjack_transitions.py
David Gillman 10/27/2025

Calculates and stores blackjack transition probabilities.

The state space is a 12x22 array.
For d>= 2, p>=2, (d, p) represents a dealer total of d and a player total of
 - p, with a usable ace, if p>=12
 - p+10, with no usable ace, if p<12

This code assumes that the environment has already dealt out enough cards
to the player to generate a total of >=12. At that point,
 - p = total, if there is a usable ace
 - p = total - 10, if there is no usable ace
   (no ace at all or ace + other cards that total at least 11)

On player action Hit, the environment will consult hit_transitions, which
give the probability of going from (d, p) to (d, p'). The environment will
take one Hit step for each Hit action.

On player action Stand, the environment will consult stand_transitions, which
give the probability of going from (d, p) to (d', p). The environment will take
as many Stand transitions it takes for the dealer to stand or bust, for one
Stand action of the player.
'''
import numpy as np

# probabilities of changes to the player's total on Hit action, i.e.
# probabilities of transitions (d, p) -> (d, p') or (d, p) -> Bust
def hit_transitions():
    ht = np.array((12, 22), dtype=object)
    for i in range(2, 12):
        for j in range(2, 22):
            # ht[i, j][(k, l)] is the probability of going from (i, j) to (k, l)
            ht[i, j] = dict()

            for card in range(1, 14):
                # since the player's total is >=12, a new ace = 1.
                value = card if card <=10 else 10 # J = Q = K = 10

                # Transitions:
                # Each transition to state (k, l)
                #  - adds (k, l): 1./13 to ht[i, j], if (k, l) isn't a key yet
                #  - adds 1./13 to ht[i, j][(k, l)] if (k, l) is already a key
                #
                # If there's a usable ace (p>=12), a new card c causes
                #  (d, p) -> (d, p+c-10), if p+c > 21
                #  (d, p) -> (d, p+c), if p+c <= 21
                #
                # If there's no usable ace (p < 12 but the player's total is p+10),
                # a new card c causes
                #  (d, p) -> Bust, if p+c > 11
                #  (d, p) -> (d, p+c), if p+c <= 11

                # Fill in here!

    return ht

def stand_transitions():
    st = np.array((12, 22), dtype=object)
    for i in range(2, 12):
        for j in range(2, 22):
            # st[i, j][(k, l)] is the probability of going from (i, j) to (k, l)
            st[i, j] = dict()

            # if the dealer busts, the transition should be to Win

            # Fill in here!

    return st
                

