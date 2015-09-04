# Factorized Asymptotic Bayesian Hidden Markov Models
#
# Author: Taikai Takeda <297.1951@gmail.com>

import numpy as np
from .base import ConvergenceMonitor
from .hmm import MultinomialHMM
from .utils import iter_from_X_lengths, normalize
from sklearn.utils import check_array
from .utils import normalize, logsumexp, iter_from_X_lengths


class MultinomialFABHMM(MultinomialHMM):
    def __init__(self, n_components=10,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="sted", init_params="sted"):
        MultinomialHMM.__init__(self, n_components,
                                startprob_prior=startprob_prior,
                                transmat_prior=transmat_prior,
                                algorithm=algorithm,
                                random_state=random_state,
                                n_iter=n_iter, tol=tol, verbose=verbose,
                                params=params, init_params=init_params)

    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self._init(X, lengths=lengths, params=self.init_params)
        self._check()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

        n_samples = X.shape[0]
        self.posteriors = np.ones((n_samples,
                                  self.n_components)) / self.n_components

        count = 0
        n_seqs = 1 if lengths is None else len(lengths)

        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0

            shrinker, sh_reg = self._compute_shrinker(self.posteriors)

            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                shrinked_framelogprob = \
                    self._compute_shrinked_logprob(framelogprob, shrinker)
                logprob, fwdlattice = \
                    self._do_forward_pass(shrinked_framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(shrinked_framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self.posteriors[i:j] = posteriors

                self._accumulate_sufficient_statistics(
                    stats, X[i:j], shrinked_framelogprob,
                    posteriors, fwdlattice,
                    bwdlattice, self.params)

            self._do_mstep(stats, self.params)

            old_count = count
            count = self._eliminate_components()

            ficlb = self._compute_ficlb(curr_logprob, sh_reg,
                                        n_samples, n_seqs)

            self.monitor_.report(ficlb)
            if self.monitor_.converged and count + old_count is 0:
                break

        return self

    def _compute_ficlb(self, logprob, sh_reg, n_samples, n_seqs):
        ficlb = logprob
        ficlb += sh_reg[0] * (n_samples - n_seqs) + sh_reg[1] * n_seqs
        ficlb += - self.n_components / 2.0 * np.log(n_seqs)
        _sum_post = np.sum(self.posteriors[:-1], axis=0)
        sum_post = _sum_post + self.posteriors[-1]
        term_beta = self.n_components / 2.0 * (np.log(_sum_post) - 1)
        term_phi = self.dim_emission / 2.0 * (np.log(sum_post) - 1)
        ficlb -= np.sum(term_beta + term_phi)

        return ficlb

    def _init(self, X, lengths=None, params='sted'):
        super(MultinomialFABHMM, self)._init(X, lengths=lengths, params=params)

        if 'd' in params:
            self.dim_emission = self.n_features

    def _compute_shrinker(self, posteriors):
        n_samples, n_components = posteriors.shape
        sum_post_minus = np.sum(posteriors[:-1], axis=0)
        sum_post_minus[sum_post_minus == 0] = np.finfo(float).eps
        sum_post = sum_post_minus + posteriors[-1]

        term_beta = - n_components / (2 * sum_post_minus)

        term_phi = - self.dim_emission / (2 * sum_post)

        _logshrinker = np.zeros((2, self.n_components))
        _logshrinker[0] = term_beta + term_phi
        _logshrinker[1] = term_phi

        reg = logsumexp(_logshrinker, axis=1)

        logshrinker = _logshrinker - reg[:, np.newaxis]
        return logshrinker, reg

    def _compute_shrinked_logprob(self, framelogprob, logshrinker):
        n_observations, n_components = framelogprob.shape

        shrinked_logprob = np.zeros((n_observations, n_components))
        shrinked_logprob[:-1] = \
            framelogprob[:-1, :] + logshrinker[np.newaxis, 0]
        shrinked_logprob[-1] = framelogprob[-1, :] + logshrinker[1]

        return shrinked_logprob

    def _eliminate_components(self):
        pst = self.posteriors
        pst[pst < 1e-5] = 0
        sum_post = np.sum(pst, axis=0)
        idx = sum_post > 1
        count = np.count_nonzero(idx == 0)
        self.n_components -= count
        self.posteriors = self.posteriors[:, idx]
        idx_2d = np.logical_and(idx[:, np.newaxis],
                                idx[np.newaxis, :])
        self.transmat_ = self.transmat_[idx_2d]\
                             .reshape(self.n_components, self.n_components)
        self.startprob_ = self.startprob_[idx]
        self.emissionprob_ = self.emissionprob_[idx, :]\
                                 .reshape(self.n_components, self.n_features)

        normalize(self.startprob_)
        normalize(self.startprob_)

        return count
