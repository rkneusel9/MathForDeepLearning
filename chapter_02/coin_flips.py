#
#  file:  coin_flips.py
#
#  Probability of getting 0,1,2, or 3 heads
#  in three flips of a coin.
#
#  RTK, 05-Jun-2020
#  Last update: 05-Jun-2020
#
################################################################

import numpy as np

N = 1000000
M = 4

heads = np.zeros(M+1)

for i in range(N):
    flips = np.random.randint(0,2,M)
    h, _ = np.bincount(flips, minlength=2)
    heads[h] += 1

prob = heads / N

print()
print("Probabilities: %s" % np.array2string(prob))
print()

