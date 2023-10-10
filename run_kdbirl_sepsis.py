'''
Sample from the KD-BIRL posterior for the sepsis management task.
'''
from hashlib import md5
from os.path import join as pjoin
import os
import pickle
import numpy as np
import tqdm
import pystan
import scipy
import pandas as pd

with open("../kdbirl_sepsis.stan", "r") as file:
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

def distance_r_euclidean(r, r_prime):
    h_prime = 1
    dist = scipy.spatial.distance.euclidean(r, r_prime)
    # Euclidean distance
    return np.exp(-(np.power(dist, 2) / (2 * h_prime)))

def distance_points(p1, p2):
    h = 1
    dist = scipy.spatial.distance.euclidean(p1, p2)
    return np.exp(-np.power(dist, 2) / (2 * h))

def find_hprime(obs_rewards):
    # Distance between rewards, h_prime
    all_distances_r = []
    for ii, o in enumerate(tqdm.tqdm(obs_rewards)):
        for o_prime in obs_rewards:
            all_distances_r.append(distance_r_euclidean(o, o_prime))
    h_prime = np.std(all_distances_r) * np.std(all_distances_r)
    return h_prime
def find_h(training_obs):
    all_distances_p = []
    for o in training_obs:
        for o_prime in training_obs:
            all_distances_p.append(distance_points(o, o_prime))
    h = np.std(all_distances_p) * np.std(all_distances_p)
    return h

def sample_kdbirl(reward, n_iter, n_posterior_samples, n, m, h, h_prime, fname=None):
    input_dir = "./observations"
    observations_points = np.load(input_dir + "observations_phi_sa.npy").tolist()
    observations_rewards = np.load(input_dir + "observations_r.npy").tolist()
    behavior_points = np.load(input_dir + "behavior_" + str(reward) + ".npy").tolist()

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
    print("reward: ", reward)

    # Fit and check if chains mixed
    rhat_failed = True
    kdbirl_data = {"J": 3, "n": n, "m": m, "training_points":
        observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h, "h_prime": h_prime}

    while rhat_failed:
        fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1, control={"adapt_delta": 0.9, "stepsize":0.05},
                          init="random")
        rhat_vals = fit.summary()["summary"][:, -1]
        print("RHAT: ", rhat_vals)
        rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)
    print(str(fit))
    sample_reward = np.squeeze(fit.extract()["sample_reward"])
    cols = [str(i) for i in range(3)]
    sample_df = pd.DataFrame(np.vstack(sample_reward), columns=cols)
    if fname is not None:
        sample_df.to_csv(fname)


reward = [0.8, 0.6, 0.4]
m = 500
fdir = "./results"
n = 500
niter= 10000
h = 0.4
h_prime = 0.5
fname = fdir + "n=" + str(n) + "_m=" + str(m) + "_h=" + str(h) + "_hprime=" + str(h_prime) + "_iter=" + str(niter) + "_w=" + str(reward) + "_prior=uniform.csv"
sample_kdbirl(reward=reward, n_iter=niter, n_posterior_samples=niter-2000, n=n, m=m, h=h, h_prime=h_prime, fname=fname)