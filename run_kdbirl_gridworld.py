from src.utils import *
import numpy as np
import md5
import tqdm
import pandas as pd
import pystan
import os
import pickle
from os.path import join as pjoin


print("Sampling grid file")
with open("src/stan_files/gridworld.stan", "r") as file:
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

def sample_kdbirl(n_iter, n_posterior_samples, gridsize, target_state, n, m, h, h_prime, step, reward_fn=None,
                  save=True, num_rl=1000):
    if step:
        observations_points = []
        observations_rewards = []
        observations = []
        init_experience = 200
        target_states = []
        for i in range(0, gridsize):
            for j in range(0, gridsize):
                target_states.append([i, j])

        for kk, r in enumerate(tqdm.tqdm(target_states)):
            behavior_opt = runNonlinearFQI(init_experience=init_experience, behavior=True, target_state=r, n=gridsize,
                                           num_rollouts=1000)
            for sample in behavior_opt:
                observations.append((sample[0], targetstate_to_rewardvec(r, gridsize)))
                s_a = feature_to_oh(list(sample[0]), gridsize=gridsize)
                action = action_to_index(sample[1])
                s_a = s_a + action
                observations_points.append(s_a)
                observations_rewards.append(targetstate_to_rewardvec(r, gridsize))

        behavior_opt = runNonlinearFQI(init_experience=init_experience, behavior=True, target_state=target_state,
                                       n=gridsize, num_rollouts=1000)

        behavior_points = []
        for b in behavior_opt:
            s_a = feature_to_oh(list(b[0]), gridsize=gridsize)
            action = action_to_index(b[1])
            s_a = s_a + action
            behavior_points.append(s_a)

    else:  # Nonzero reward
        observations_points = []
        observations_rewards = []
        observations = []

        reward_vecs = []
        for i in range(40):
            reward_vecs.append(np.random.uniform(low=0.0, high=1.0, size=gridsize * gridsize))

        idx = [i for i in range(len(reward_vecs))]
        size_rl = min(num_rl, len(reward_vecs))
        new_idx = np.random.choice(idx, size=size_rl, replace=False)
        for i, r in enumerate(reward_vecs):
            if i in new_idx:
                reward_vec = reward_vecs[i]
                policy_rollouts = runNonparametricFQI(reward_vec, gridsize, num_rollouts=1000)
                for sample in policy_rollouts:
                    s_a = feature_to_oh(list(sample[0]), gridsize=gridsize)
                    action = action_to_index(sample[1])
                    s_a = s_a + action
                    observations_points.append(s_a)
                    observations_rewards.append(reward_vec)
                    observations.append((sample[0], reward_vec))

        behavior_opt = runNonparametricFQI(reward_fn, gridsize, num_rollouts=10000)
        behavior_points = []
        for b in behavior_opt:
            s_a = feature_to_oh(list(b[0]), gridsize=gridsize)
            action = action_to_index(b[1])
            s_a = s_a + action
            behavior_points.append(s_a)

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
    # print(str(find_optimal_bandwidth(observations, gridsize, metric_r='euclidean')))

    # Fit and check if chains mixed
    rhat_failed = True
    kdbirl_data = {"J": gridsize * gridsize, "n": n, "m": m, "training_points":
        observations_points, "training_rewards": observations_rewards, "behavior_points": behavior_points, "h": h,
                   "h_prime": h_prime}

    while rhat_failed:
        fit = sm.sampling(data=kdbirl_data, iter=n_iter, warmup=n_iter - n_posterior_samples, chains=1,
                          control={"adapt_delta": 0.95})
        rhat_vals = fit.summary()["summary"][:, -1]
        print("RHAT: ", rhat_vals)
        rhat_failed = np.sum(rhat_vals < 0.9) or np.sum(rhat_vals > 1.1)

    # Get samples, it's an 800 by 4 numpy array
    sample_reward = np.squeeze(fit.extract()["sample_reward"])
    cols = [str(i) for i in range(gridsize * gridsize)]
    sample_df = pd.DataFrame(np.vstack(sample_reward), columns=cols)
    print(str(fit))
    if save:
        sample_df.to_csv("fdir/m=" + str(m) + "_n=" + str(n) + "_fn=" + str(reward_fn) + "_num_rl=" + str(num_rl) + ".csv")

# Fit the 2x2 Gridworld model
gridsize = 2
target_state = [1, 1]
m = 200
n = 200
reward_fn = [0, 0, 0, 1]
sample_kdbirl(5000, 4000, 2, target_state, n, m, 0.5, 0.4, step=True, reward_fn=reward_fn, save=True)

# Fit the 4x4 Gridworld model
m = 200
n = 200
reward_fn = [0]*16
reward_fn[-1] = 1
sample_kdbirl(n_iter=10000, n_posterior_samples=8000, gridsize=4, target_state=[3, 3], n=200, m=200, h=0.1, h_prime=0.4, step=True, reward_fn=reward_fn, save=True)


# Fit the 5x5 Gridworld model
m = 300
n = 300
reward_fn = [0]*25
reward_fn[-1] = 1
sample_kdbirl(n_iter=10000, n_posterior_samples=8000, gridsize=5, target_state=[4, 4], n=n, m=m, h=0.1, h_prime=0.4, step=True, reward_fn=reward_fn, save=True)

# TODO: Fit the 10x10 Gridworld with a featurized reward function

