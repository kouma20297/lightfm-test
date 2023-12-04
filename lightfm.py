import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

data = fetch_movielens(min_rating=5.0)


print(repr(data["train"]))
print(repr(data["test"]))

model = LightFM(loss="warp-kos")
start_time = time.time()
model.fit(data["train"], epochs=30, num_threads=2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
