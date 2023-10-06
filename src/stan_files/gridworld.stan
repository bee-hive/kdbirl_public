data {
  int<lower=0> J;          // length of the reward vector (should be l^2 where l is the dimension of the gridworld)
  int<lower=0> n;          // number of expert samples
  int<lower=0> m;          // number of training samples
  real<lower=0.000001> h;  // bandwidth for point
  real<lower=0.000001> h_prime; // bandwidth for reward
  matrix[m, J + 5] training_points;    // training dataset + oh actions
  matrix[m, J] training_rewards;    // training dataset
  matrix[n, J + 5] behavior_points;          // expert demonstrations + oh actions
}
parameters { // Correct
  vector<lower=0.0, upper=1.0>[J] sample_reward;
}
transformed parameters { // calculate all the distance metrics.
  real dist_rewards = 0;
  real likelihood = 0;
  real inner_sum = 0;
  //print("sampled reward: ", sample_reward)
  for (ii in 1:m) {
    dist_rewards += exp(-square(distance(sample_reward, training_rewards[ii]))/(2*h_prime));
  }
  //print("behavior points: ", behavior_points);
  for (ii in 1:n) {
    inner_sum = 0;
    for (jj in 1:m) {
        inner_sum += (exp(-square(distance(training_points[jj], behavior_points[ii]))/(2*h)) * exp(-square(distance(sample_reward, training_rewards[jj]))/(2*h_prime)))/dist_rewards;
    }

    likelihood += log(inner_sum);
  }
  //print("likelihood: " , likelihood)
  //likelihood += normal_lpdf(sample_reward | 0, 0.0000001); // Add a prior on the reward
}
model {
  target += likelihood;
}