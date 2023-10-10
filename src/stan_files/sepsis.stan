data {
  int<lower=2> J;          // length of the reward vector weights
  int<lower=100> n;          // number of expert samples
  int<lower=100> m;          // number of training samples
  real<lower=0.01> h;  // bandwidth for point
  real<lower=0.01> h_prime; // bandwidth for reward
  matrix[m, 3] training_points;    // training dataset
  matrix[m, J] training_rewards;    // training dataset
  matrix[n, 3] behavior_points;          // expert demonstrations
}
parameters {
  vector<lower=0.0, upper=1.0>[J] sample_reward;
}
transformed parameters { // calculate all the distance metrics.
  real likelihood = 0;
  real inner_sum = 0;
  real dist_rewards = 0;
  for (ii in 1:n) {
    dist_rewards = 0;
    for (jj in 1:m) {
        dist_rewards += exp(-square(distance(sample_reward,training_rewards[jj]))/(2*h_prime));
    }
    inner_sum = 0;
    for (jj in 1:m) {
        inner_sum += (exp(-square(distance(behavior_points[ii], training_points[jj]))/(2*h)) * exp(-square(distance(sample_reward,training_rewards[jj]))/(2*h_prime)))/dist_rewards;
    }
    likelihood += log(inner_sum);
  }
 likelihood += normal_lpdf(sample_reward | 0, 1);
}
model {
  target += likelihood;
}