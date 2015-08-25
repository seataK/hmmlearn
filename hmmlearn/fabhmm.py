# Hidden Markov Models
#
# Author: Taikai Takeda <297.1951@gmail.com>

import numpy as np
from .base import ConvergenceMonitor
from .hmm import MultinomialHMM
from .utils import iter_from_X_lengths, normalize
from sklearn.utils import check_array
from .utils import normalize, logsumexp, iter_from_X_lengths


class MultinomialFABHMM(MultinomialHMM):
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

        n_seqs = 1 if lengths is None else len(lengths)
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            ficlb = 0

            shrinker, sh_reg = self._compute_shrinker(self.posteriors)

            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                shrinked_framelogprob = \
                    self._compute_shrinked_logprob(framelogprob, shrinker)
                logprob, fwdlattice = \
                    self._do_forward_pass(shrinked_framelogprob)
                ficlb += logprob
                ficlb += sh_reg[1]
                ficlb += sh_reg[0] * (j - i - 1)
                bwdlattice = self._do_backward_pass(shrinked_framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self.posteriors[i:j] = posteriors

                self._accumulate_sufficient_statistics(
                    stats, X[i:j], shrinked_framelogprob,
                    posteriors, fwdlattice,
                    bwdlattice, self.params)

            D_beta = self.n_components
            D_phi = self.n_features

            self._do_mstep(stats, self.params)

            count = self._eliminate_components()

            ficlb -= D_beta / 2.0 * np.log(n_seqs)
            sum_post_minus = np.sum(self.posteriors[:-1], axis=0)
            sum_post = sum_post_minus + self.posteriors[-1]
            term_beta = D_beta / 2.0 * (np.log(sum_post_minus) - 1)
            term_phi = D_phi / 2.0 * (np.log(sum_post) - 1)
            ficlb -= np.sum(term_beta + term_phi)

            self.monitor_.report(ficlb)
            if self.monitor_.converged and count is 0:
                break

        return self

    def _compute_shrinker(self, posteriors):
        n_samples, n_components = posteriors.shape
        sum_post_minus = np.sum(posteriors[:-1], axis=0)
        sum_post_minus[sum_post_minus == 0] = np.finfo(float).eps
        sum_post = sum_post_minus + posteriors[-1]

        term_beta = - n_components / (2 * sum_post_minus)
        # n_components should be n_demension of emmision parameter
        term_phi = - self.n_features / (2 * sum_post)

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

        print(self.n_components)
        return count
