import numpy as np

base=np.array([[1,2],[3,4],[5,6]])
nsample=3
tmp = []
for i in range(nsample):
    id_pick = np.random.choice(np.shape(base)[0], size=(np.shape(base)[0]))
    boot1=base[id_pick,:]
    tmp.append(boot1)
tmp = np.stack(tmp, axis=-1)
print(tmp)
