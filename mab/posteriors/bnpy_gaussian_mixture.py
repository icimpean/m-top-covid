import random
from pathlib import Path

import numpy as np
import bnpy
import pickle

from mab.posteriors import Posterior, SinglePosteriors


class BNPYGaussianMixturePosterior(Posterior):
    """A gaussian mixture posterior of a bandit's arms, based on the BNPy project:

    github: https://github.com/bnpy/bnpy
    installation: https://bnpy.readthedocs.io/en/latest/installation.html
    """

    def __init__(self, k, tol, max_iter, num=0, seed=None, log_dir="./rl-tmp"):
        # Super call
        super(BNPYGaussianMixturePosterior, self).__init__(seed)
        # The internal mixture distribution
        self._model_type = "DPMixtureModel"
        self._allocModelName = "Gauss"
        self._obsModelName = "memoVB"
        path = log_dir
        self._output_path = Path(f"{path}/Posteriors/posterior_{num}/").absolute()

        self._nLap = max_iter
        self._sF = 0.1
        self._ECovMat = "eye"
        self._initname = 'randexamples'
        self._K = k
        self._tol = tol

        self.mean_ = 0
        self.var_ = 0
        self.std_ = 0

        self._is_initialised = False
        # force update with no new data to sample uninitialised posteriors
        _init_data = bnpy.data.XData(np.random.default_rng(seed=seed).random(size=(self._K, 1)))
        self._model, self.info = bnpy.run(_init_data, self._model_type, self._allocModelName, self._obsModelName,
                                          doWriteStdOut=False, doSaveToDisk=True, taskID=-1,
                                          output_path=self._output_path, initname=self._initname, K=1,
                                          nLap=self._nLap, sF=self._sF, ECovMat=self._ECovMat, convergeThr=self._tol)
        # Don't save output path of this trained posterior,
        # It only serves as a temporary reward distribution until real data is used for training

    @staticmethod
    def new(seed, k, tol, max_iter, num=0, log_dir="/Users/alexandracimpean/Documents/VUB/PhD/COVID19/Code/rl-tmp/"):
        return BNPYGaussianMixturePosterior(k, tol, max_iter, num, seed, log_dir)

    def update(self, reward, t):
        self.rewards.append(reward)
        X = np.array(self.rewards)
        # Data contains a single feature (reward)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        dataset = bnpy.data.XData(X)
        # Run the algorithm
        self._model, self.info = self._update(dataset, t)
        self._is_initialised = True
        # Update statistics
        self.mean_ = self.mean(t)
        self.var_ = self.get_cov()          # NOT the variance, but the covariance matrix for the mixture components
        self.std_ = self.get_std_dev()      # NOT the std dev, but uncertainty on mean for each mixture component

    def _update(self, dataset, t):
        model, info_dict = bnpy.run(dataset, self._model_type, self._allocModelName, self._obsModelName,
                                    doSaveToDisk=True, taskID=t, output_path=self._output_path, nLap=self._nLap,
                                    sF=self._sF, ECovMat=self._ECovMat, convergeThr=self._tol, doWriteStdOut=True,
                                    K=1 if self._initname == 'randexamples' else self._K, initname=self._initname,
                                    nTask=1, nBatch=1,
                                    moves='birth,merge,shuffle', m_startLap=3, b_startLap=0, b_Kfresh=2
                                    )
        self._initname = info_dict['task_output_path']
        return model, info_dict

    def get_means(self):
        """Extract posterior means from bnpy model"""
        if self._is_initialised:
            return self._model.obsModel.Post.m
        else:
            return self._model.obsModel.Prior.m

    def get_cov(self, k=None):
        """Extract posterior covariance matrix from bnpy model"""
        P = self._model.obsModel.Post if self._is_initialised else self._model.obsModel.Prior
        B = P.B
        nu = P.nu
        if k is not None:
            B = B[k]
            nu = nu[k]
        covar = B / (nu - self._model.obsModel.D - 1)
        return covar

    def get_weights(self):
        """Extract posterior mixture weights/proportions from bnpy model"""
        # Special case of 1 component => weight should always be 1, but isn't
        weights = self._model.allocModel.get_active_comp_probs()
        if self.get_K() == 1:
            weights = np.array([1])
        return weights

    def mean_precisions(self):
        """bnpy scalar precision on parameter mu defined in GaussObsModel"""
        return self._model.obsModel.Post.kappa

    def get_std_dev(self):
        """Get std dev from mean precision on each component"""
        prec = self.mean_precisions()
        var = 1 / prec
        std = np.sqrt(var)
        return std

    def get_K(self):
        return self._model.allocModel.K

    def sample(self, t):
        """Sample the means"""
        weights = self.get_weights()
        means = self.get_means()
        precisions = self.mean_precisions()
        overall_mean_sample = 0
        # print("SAMPLE")
        for k in range(self._model.allocModel.K):
            w = weights[k]
            mu = means[k][0]
            # precision = 1 / variance; variance = std ** 2
            prec = precisions[k]
            std = np.sqrt(1 / prec)
            # Sample mean distribution
            m_sample = self.rng.normal(loc=mu, scale=std)
            # print("\tk:", k, "sample:", m_sample, "w:", w, "mu:", mu, "p:", prec, "std:", std)
            # Weighted sample for all mixtures:
            overall_mean_sample += (m_sample * w)
        # print("\tOverall mean:", overall_mean_sample)
        return overall_mean_sample

    def save(self, path):
        with open(path, "wb") as file:
            data = (self.rng, self._model, self._initname, self.rewards, self.mean_, self.var_, self.std_)
            pickle.dump(data, file)

    def load(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
            self.rng, self._model, self._initname, self.rewards, self.mean_, self.var_, self.std_ = data
            self._is_initialised = True

    def mean(self, t):
        """The mean of the gaussian mixture distribution."""
        mean = np.sum(self.get_means() * self.get_weights())
        return mean


class BNPYBGMPosteriors(SinglePosteriors):
    """A Bayesian Gaussian Mixture Posterior for a given number of bandit arms."""

    def __init__(self, nr_arms, seed=None, k=2, tol=0.001, max_iter=100, log_dir="./"):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.log_dir = log_dir
        super(BNPYBGMPosteriors, self).__init__(nr_arms, BNPYGaussianMixturePosterior, seed)

    def _create_posteriors(self, seed, posterior_type):
        return [posterior_type.new(seed + i, self.k, self.tol, self.max_iter, i, self.log_dir) for i in range(self.nr_arms)]
