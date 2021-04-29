#
#  file:  birthday.py
#
#  Simulate the birthday paradox
#
#  RTK, 04-Jun-2020
#  Last update:  04-Jun-2020
#
################################################################

import numpy as np

# Simulate picking two people at random, probability of sharing a birthday
N = 100000
match = 0
for i in range(N):
    a = np.random.randint(0,364)
    b = np.random.randint(0,364)
    if (a == b):
        match += 1
print()
print("Probability of a random match = %0.6f" % (match/N,))
print()

# Simulate people in a room, N tests per M
M = 30
N = 100000
for m in range(2,M+1):
    matches = 0
    for n in range(N): 
        match = 0
        b = np.random.randint(0,364,m)
        for i in range(m):
            for j in range(m):
                if (i != j) and (b[i] == b[j]):
                    match += 1
        if (match != 0):
            matches += 1
    print("%2d people: probability of at least one match %0.6f" % (m, matches/N))

