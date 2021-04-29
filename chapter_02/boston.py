import numpy as np
P = 50
B = 4
M = 3
N = 100000

nb = 0

for i in range(N):
    s = np.random.randint(0,P,M)
    fail = False
    for t in range(M):
        if (s[t] < B):
            fail = True
    if (not fail):
        nb += 1

print()
print("Prob no Boston in the fall = %0.4f" % (nb/N,))
print()

