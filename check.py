import numpy as np
from api.db import VectorDB

rng = np.random.default_rng(42)
vectors = rng.normal(size=(1000, 128))

db = VectorDB(dim=128, M=8, k=16)
db.fit(vectors)
db.insert_batch(list(range(1000)), vectors)
results = db.query(vectors[87], top_k=5)

print('Top 5 results for vector 42:')
for vid, dist in results:
  print(f'  id={vid}, distance={dist:.4f}')
db.save("./my_index")
