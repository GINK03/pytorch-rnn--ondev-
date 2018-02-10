
bs = []
for i in range(100):
  a = list(range(0,100))
  bs.append(a)

import pickle

pickle.dump(bs, open('data.pkl','wb'))
