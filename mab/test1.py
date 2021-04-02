from envs.env import GaussianEnv, GaussianMixtureEnv
from mab.bandits.bgm_bandit import BayesianGaussianMixtureBandit
from mab.bandits.gaussian_bandit import GaussianBandit
from mab.bandits.gaussian_mixture_bandit import GaussianMixtureBandit
from mab.sampling.bfts import BFTS
from mab.sampling.thompson_sampling import ThompsonSampling

if __name__ == '__main__':

    s = 0  # Integer (or None for random seed configuration)
    n_arms = 10
    steps = 1000
    # The type of environment
    # env = GaussianEnv(n_arms, seed=s)
    env = GaussianMixtureEnv(n_arms, mixtures_per_arm=2, seed=s)
    env.print_info()
    env.plot_rewards()

    # The sampling method
    sampling_method = ThompsonSampling
    # sampling_method = BFTS.new(top_m=2)

    # Gaussian posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Nonparametric Gaussian Mixture posterior bandit
    # bandit = GaussianMixtureBandit(n_arms, env, k=2, sampling_method=sampling_method, t_max=steps, seed=s)
    # Sklearn-based implementation
    bandit = BayesianGaussianMixtureBandit(n_arms, env, sampling_method, k=2, seed=s)

    # Let the bandit run for the given number of steps
    bandit.play_bandit(steps=steps, initialise_arms=3)
