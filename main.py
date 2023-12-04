import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import time
import resource

data = fetch_movielens(min_rating=5.0)


print(repr(data["train"]))
print(repr(data["test"]))


model = LightFM(loss="warp-kos")
model.fit(data["train"], epochs=30, num_threads=2)


# Start time
start_time = time.time()

# User CPU time and System CPU time at the start
start_resources = resource.getrusage(resource.RUSAGE_SELF)
start_user_cpu_time = start_resources.ru_utime
start_system_cpu_time = start_resources.ru_stime

# End time
end_time = time.time()

# User CPU time and System CPU time at the end
end_resources = resource.getrusage(resource.RUSAGE_SELF)
end_user_cpu_time = end_resources.ru_utime
end_system_cpu_time = end_resources.ru_stime

# Calculate elapsed time and CPU times
elapsed_time = end_time - start_time
user_cpu_time = end_user_cpu_time - start_user_cpu_time
system_cpu_time = end_system_cpu_time - start_system_cpu_time

print(
    f"CPU時間: User {user_cpu_time} seconds, システム: {system_cpu_time} seconds, 合計時間: {user_cpu_time + system_cpu_time} seconds"
)


print("Train precision: %.2f" % precision_at_k(model, data["train"], k=5).mean())
print("Test precision: %.2f" % precision_at_k(model, data["test"], k=5).mean())


def sample_recommendation(model, data, user_ids):
    n_users, n_items = data["train"].shape

    for user_id in user_ids:
        known_positives = data["item_labels"][data["train"].tocsr()[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items))
        top_items = data["item_labels"][np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


sample_recommendation(model, data, [3, 25, 450])
