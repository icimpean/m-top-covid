from envs.env import GaussianEnv
from mab.bandits.gaussian_bandit import GaussianBandit
from mab.sampling.bfts import BFTS
from mab.sampling.thompson_sampling import ThompsonSampling

if __name__ == '__main__':

    s = 0
    n_arms = 10
    env = GaussianEnv(n_arms)
    env.print_info()
    env.plot_rewards()

    # The sampling method
    # sampling_method = ThompsonSampling
    sampling_method = BFTS.new(top_m=2)

    # Gaussian posterior bandit
    bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)
    # Nonparametric Gaussian Mixture posterior bandit
    # bandit = GaussianBandit(n_arms, env, sampling_method, seed=s)

    # Let the bandit run for the given number of steps
    bandit.play_bandit(steps=1000, initialise_arms=1)
