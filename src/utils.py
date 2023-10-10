from environments.gridworld import Gridworld
from sklearn.linear_model import LinearRegression
import torch
import numpy as np

def runLinearFQI(reward_weights, gridsize, num_rollouts):
    if reward_weights[0] > 0 and reward_weights[1] > 0:
        target_state = [gridsize-1, gridsize-1]
    elif reward_weights[0] < 0 and reward_weights[1] < 0:
        target_state = [0, 0]
    elif reward_weights[0] < 0 and reward_weights[1] > 0:
        target_state = [0, gridsize-1]
    else:
        target_state = [gridsize-1, 0]
    rollouts = []
    for _ in range(num_rollouts):
        rollout, done = generate_policy_rollout_linear(reward_weights, target_state=target_state, gridsize=gridsize)
        if done:
            rollouts.extend(rollout)
    policy_rollouts = rollouts
    return policy_rollouts

def linearreward_to_rewardvec(linearreward, gridsize):
    gridsize = int(gridsize)
    reward_vec = np.zeros((gridsize, gridsize))
    for x in range(gridsize):
        for y in range(gridsize):
            reward_vec[x, y] = np.dot(linearreward, [x, y])
    return reward_vec.flatten()

def generate_rollout(reg, group=0, target_state=None, gridsize=4):
    rollout = []
    obs = [np.random.choice(4), np.random.choice(4)]
    time_limit = 15
    it = 0
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)

        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            reward = reg.predict([[nxt_agent_state[0], nxt_agent_state[1]]])
            curr_rewards[ii] = reward + np.random.normal(scale=0.05)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        done = False
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, group))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, 0, max_reward, target_state, done, group))
            break
    return rollout

def runNonlinearFQI(behavior=True, group=0, target_state=None, init_experience=100, n=4, num_rollouts=100):
    train_env = Gridworld(group=group, target_state=target_state,
                          gridsize=n, nonlinear=True)

    rollouts = []
    if init_experience > 0:
        for _ in range(init_experience):
            rollout, episode_cost = train_env.generate_rollout(None, render=False, group=group, random_start=True)
            rollouts.extend(rollout)
    all_rollouts = rollouts.copy()

    if not behavior:
        state_b, action_b, cost_b, next_state_b, done_b, group_b = zip(*all_rollouts)
        cost_b = torch.FloatTensor(cost_b)
        reg = LinearRegression().fit(next_state_b, cost_b)
        rollouts = []
        if init_experience > 0:
            for _ in range(int(init_experience / 2)):
                rollout = generate_rollout(reg, group=group, target_state=target_state)
                rollouts.extend(rollout)
        policy_rollouts = rollouts
    else:
        rollouts = []
        if init_experience > 0:
            for _ in range(num_rollouts):
                rollout, done = generate_policy_rollout_nonlinear(group=group, target_state=target_state, gridsize=n)
                if done:
                    rollouts.extend(rollout)
            policy_rollouts = rollouts
    return policy_rollouts


def generate_policy_rollout_nonlinear(group=0, target_state=None, gridsize=10):
    rollout = []
    seen_states = []
    obs = [np.random.choice(gridsize), np.random.choice(gridsize)]
    # obs = [0, 0]
    time_limit = 7
    it = 0
    done = False
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)

        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            if nxt_agent_state[0] == target_state[0] and nxt_agent_state[1] == target_state[1]:
                reward = 1.0
            else:
                reward = 0.0
            if list(nxt_agent_state) not in seen_states:
                curr_rewards[ii] = reward + np.random.normal(scale=0.0001)
            if list(nxt_agent_state) in seen_states:
                curr_rewards[ii] = -np.Inf
        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = (obs[0] + action[0], obs[1] + action[1])
        cost = max_reward
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, group))
        seen_states.append(list(obs))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, [0, 0], 1.0, target_state, True, group))
            break
    return rollout, done


def runNonparametricFQI(reward_vec, gridsize=2, num_rollouts=100):
    # Find the target state
    max_reward_idx = np.argmax(reward_vec)
    target_state = convert_state_idx_to_state_coord(max_reward_idx, gridsize)
    target_state = target_state.tolist()
    rollouts = []
    for _ in range(num_rollouts):
        rollout, done = generate_policy_rollout_nonparametric(reward_vec, target_state=target_state, gridsize=gridsize)
        if done:
            rollouts.extend(rollout)
    policy_rollouts = rollouts
    return policy_rollouts


def generate_policy_rollout_nonparametric(reward_vec, target_state, gridsize=10, time_limit=30):
    rollout = []
    obs = [0, 0]
    it = 0
    done = False
    while it < time_limit:
        valid_actions, action_pos_dict = find_valid_actions(obs, gridsize=gridsize)
        curr_rewards = np.zeros(len(valid_actions))
        for ii, a in enumerate(valid_actions):
            nxt_agent_state = (obs[0] + action_pos_dict[a][0], obs[1] + action_pos_dict[a][1])
            nxt_state_idx = convert_state_to_state_idx(nxt_agent_state, gridsize)
            reward = reward_vec[nxt_state_idx]
            curr_rewards[ii] = reward + np.random.normal(scale=0.01)

        # Choose best action, randomly breaking ties
        best_action_idx = np.random.choice(np.flatnonzero(curr_rewards == curr_rewards.max()))
        max_reward = curr_rewards[best_action_idx]
        action = action_pos_dict[valid_actions[best_action_idx]]
        next_obs = [obs[0] + action[0], obs[1] + action[1]]
        cost = max_reward
        if next_obs[0] == target_state[0] and next_obs[1] == target_state[1]:
            done = True
        rollout.append((obs, action, cost, next_obs, done, 0))
        obs = next_obs
        it += 1
        if done:
            rollout.append((target_state, [0, 0], 1.0, target_state, True, 0))
            break
    return rollout, done


def convert_state_to_state_idx(state_coord, grid_dimension):
    return grid_dimension * state_coord[0] + state_coord[1]


def convert_state_idx_to_state_coord(state_idx, grid_dimension):
    return np.array([state_idx // grid_dimension, state_idx % grid_dimension])


def feature_to_oh(xy, gridsize):
    oh = [0] * gridsize * gridsize
    oh[xy[0] * gridsize + xy[1]] = 1
    return oh


def find_valid_actions(pos, gridsize=4):
    valid_actions = []
    action_pos_dict = {1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
    for k in action_pos_dict:
        nxt_agent_state = (
            pos[0] + action_pos_dict[k][0],
            pos[1] + action_pos_dict[k][1],
        )
        if (
                nxt_agent_state[0] >= 0
                and nxt_agent_state[0] < gridsize
                and 0 <= nxt_agent_state[1] < gridsize
        ):
            valid_actions.append(k)
    return valid_actions, action_pos_dict

def targetstate_to_rewardvec(target_state, gridsize):
    reward_vec = np.zeros(int(gridsize * gridsize))
    idx = target_state[0] * gridsize + target_state[1]
    reward_vec[int(idx)] = 1
    return reward_vec


def action_to_index(a):
    action_vec = [0] * 5
    if a == [-1, 0]:
        action_vec[0] = 1
    elif a == [1, 0]:
        action_vec[1] = 1
    elif a == [0, -1]:
        action_vec[2] = 1
    elif a == [0, 1]:
        action_vec[3] = 1
    elif a == [0, 0]:
        action_vec[4] = 1
    return action_vec

