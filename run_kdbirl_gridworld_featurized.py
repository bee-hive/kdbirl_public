'''
10x10 Featurized Gridworld
'''
import pystan
import pandas as pd
import pickle
import numpy as np
from hashlib import md5
from os.path import join as pjoin
import os
import tqdm
from src.utils import *

# Load the stan model
with open("./stan_files/gridworld_featurized.stan", "r") as file:
    model_code = file.read()
code_hash = md5(model_code.encode("ascii")).hexdigest()
cache_fn = pjoin("cached_models", "cached-model-{}.pkl".format(code_hash))

if os.path.isfile(cache_fn):
    print("Loading cached model...")
    sm = pickle.load(open(cache_fn, "rb"))
else:
    print("Saving model to cache...")
    sm = pystan.StanModel(model_code=model_code)
    with open(cache_fn, "wb") as f:
        pickle.dump(sm, f)
# Generate expert demonstrations
true_reward = [-1, 1]
print("True Weights: ", true_reward)
save = True
gridsize = 10
n = 500
m = 500
h = 0.1
h_prime = 0.4
n_iter = 10000
n_posterior_samples = 8000
behavior_opt = runLinearFQI(reward_weights=true_reward, gridsize=gridsize, num_rollouts=200)
behavior_points = []
for b in behavior_opt:
    behavior_points.append(b[0])

# Generate training dataset
observations_points = []
observations_rewards = []
for aa, i in enumerate(tqdm.tqdm(range(-110, 110, 20))):
    for j in range(-110, 110, 20):
        r = [i / 100, j / 100]
        behavior_opt = runLinearFQI(reward_weights=r, num_rollouts=1000, gridsize=gridsize)
        for sample in behavior_opt:
            observations_points.append(sample[0])
            observations_rewards.append(r)

# Run Stan
idx = [i for i in range(len(behavior_points))]
size_obs = min(n, len(behavior_points))
new_idx = np.random.choice(idx, size=size_obs, replace=False)
behavior_points = np.asarray(behavior_points)[new_idx]
print("Len behavior: ", behavior_points.shape[0])

idx = [i for i in range(len(observations_points))]
size_obs = min(m, len(observations_points))
new_idx = np.random.choice(idx, size=size_obs, replace=False)
observations_points = np.asarray(observations_points)[new_idx]
observations_rewards = np.asarray(observations_rewards)[new_idx]
print("len observations: ", observations_points.shape[0])

# Fit and check if chains mixed
rhat_failed = True
kdbirl_data = {"J": 2, "n": n, "m": m, "training_points":
    observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}

while rhat_failed:
    fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1, control={"adapt_delta": 0.9})
    rhat_vals = fit.summary()["summary"][:, -1]
    print("RHAT: ", rhat_vals)
    rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)

sample_reward = np.squeeze(fit.extract()["sample_reward"])
cols = [str(i) for i in range(len(true_reward))]
sample_df = pd.DataFrame(np.vstack(sample_reward), columns=cols)
print(str(fit))
if save:
    sample_df.to_csv("./results/sanity/m=" + str(m) + "_n=" + str(n) +
                     "_fn=" + str(true_reward) + "_h=" + str(h) + "_h_prime=" + str(h_prime) + "_iter=" + str(n_iter) + ".csv")
