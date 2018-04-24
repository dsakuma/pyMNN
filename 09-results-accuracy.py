# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %%
with open('data_out/accuracy/acc.bin', 'rb') as f:
    obj = pickle.load(f)

# %%
obj.keys()

# %%
nsamples = obj['nsamples']
amounts = obj['amounts']
sims = obj['sims']

# %%
ys = dict()
for i, ns in enumerate(nsamples):
    ys[ns] = np.mean(np.array(sims[i::len(nsamples)]).T, axis=1)

# %%
X = np.array(amounts)
colors = ['k-o', 'k-v', 'k-s']
for (ns, Y), c in zip(ys.items(), colors):
    plt.plot(X, Y, c)

plt.xlim(0, 1)
plt.ylim(0, 1.1)

plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.grid()

plt.xlabel('Amount of noise applied')
plt.ylabel('Output similarity to original image')

plt.legend(['n(samples)={0}'.format(n) for n in ys.keys()])
plt.savefig('data_out/recall-accuracy.png', dpi=300)
plt.show()
