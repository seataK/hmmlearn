# Pairwise Hidden Markov Models
#
# Author: Taikai Takeda <297.1951@gmail.com>

import numpy as np
import itertools as it
from sklearn.utils import check_array
from .utils import normalize, logsumexp, iter_from_X_lengths

from .base import ConvergenceMonitor
from .hmm import MultinomialHMM
from .utils import iter_from_X_lengths, normalize
from . import _hmmc


def iter_from_XY_lengths(X, Y, x_lengths, y_lengths):
    if x_lengths is None and y_lengths is None:
        yield 0, len(X), 0, len(Y)
    else:
        n_samples = X.shape[0]
        x_end = np.cumsum(x_lengths).astype(np.int32)
        x_start = x_end - x_lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


class MultinomialPairHMM(_BaseHMM):

    def __init__(self, n_components=10,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseHMM.__init__(self, n_components,
                                startprob_prior=startprob_prior,
                                transmat_prior=transmat_prior,
                                algorithm=algorithm,
                                random_state=random_state,
                                n_iter=n_iter, tol=tol, verbose=verbose,
                                params=params, init_params=init_params)

    def fit(self, X, Y, x_lens, y_lens):
        X = check_array(X)
        Y = check_array(Y)

        self._init()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            delta = np.zeros(self.n_components)
            for ix, jx, iy, jy in iter_from_XY_lengths(X, Y, x_lens, y_lens):
                len_x = jx - ix
                delta[0] = len_x + 1  # match
                delta[1] = 1  # ins x
                delta[2] = len_x  # ins y

                framelogprob = self._compute_log_likelihood(X[ix:jx], Y[iy:jy])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, None, framelogprob, posteriors, fwdlattice,
                    bwdlattice, self.params)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

            self._do_mstep(stats, self.params)

        return self

    def _compute_log_likelihood(self, X, Y):
        xx, yy = np.meshgrid(X, Y)
        return np.log(self.emissionprob_)[:, xx, yy].T

    def _do_forward_pass(self, framelogprob):
        len_x, len_y, n_components = framelogprob.shape
        n_observations = len_x * len_y
        fwdlattice = np.zeros((n_observations, n_components))
        _hmmc._pair_forward(n_observations,
                            n_components,
                            np.log(self.startprob_),
                            np.log(self.transmat_),
                            framelogprob,
                            fwdlattice,
                            self.delta)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        len_x, len_y, n_components = framelogprob.shape
        n_observations = len_x * len_y
        bwdlattice = np.zeros((n_observations, n_components))
        _hmmc._backward(n_observations, n_components,
                        np.log(self.startprob_), np.log(self.transmat_),
                        framelogprob.reshape((n_observations, n_components)),
                        bwdlattice, self.delta)
        return bwdlattice

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        log_gamma = fwdlattice + bwdlattice
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma += np.finfo(float).eps
        log_gamma -= logsumexp(log_gamma, axis=1)[:, np.newaxis]
        out = np.exp(log_gamma)
        normalize(out, axis=1)
        return out

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(MultinomialPairHMM, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)
        if 'e' in params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]


    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _count_features(self, X, lengths):
        symbols = set()
        for i, j in iter_from_X_lengths(X, lengths):
            symbols |= set(X[i:j].flatten())
        return len(symbols)

    def _init(self, X, Y, x_lengths=None, y_lenghts=None, params='step'):
        if not self._check_input_symbols(X, Y):
            raise ValueError("expected a sample from "
                             "a Multinomial distribution.")

        super(MultinomialPairHMM, self)\
            ._init(None, lengths=None, params=params)
        self.random_state = check_random_state(self.random_state)
        if 'p' in params:

            if not hasattr(self, "n_features"):
                self.nx_features = self._count_features(X, x_lengths)
                self.ny_features = self._count_features(Y, y_lenghts)
                self.n_features = self.nx_features * self.ny_features

            self.emissionprob_ = self.random_state \
                .rand(self.n_components, self.nx_features, self.ny_features)

            normalize(self.emissionprob_
                      .reshape(self.n_components, self.n_features), axis=1)

    def _check(self):
        super(MultinomialPairHMM, self)._check()

        self.emissionprob_ = np.atleast_3d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _initialize_sufficient_statistics(self):
        stats = super(MultinomialPairHMM, self)\
            ._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components,
                                 self.nx_features,
                                 self.ny_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, Y, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        if 's' in parames:
            stats['start'] += posteriors[0]

        if 't' in params:
            len_x, len_y, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            n_observations = len_x * len_y
            if n_observations <= 1:
                return

            lneta = np.zeros((n))

        if 'e' in params:
            for t, (x_symbol, y_symbol) in enumerate(it.product(X, Y)):
                stats['obs'][:, x_symbol, y_symbol] += posteriors[t]

    def _do_mstep(self, stats, params):

        if 'e' in params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(axis=(1, 2))[:, np.newaxis, np.newaxis])


    def _check_input_symbols(self, X, Y):
        """Check if ``X`` is a sample from a Multinomial distribution.

        That is ``X`` should be an array of non-negative integers from
        range ``[min(X), max(X)]``, such that each integer from the range
        occurs in ``X`` at least once.

        For example ``[0, 0, 2, 1, 3, 1, 1]`` is a valid sample from a
        Multinomial distribution, while ``[0, 0, 3, 5, 10]`` is not.
        """
        return MultinomialHMM._check_input_symbols(X) and \
            MultinomialHMM._check_input_symbols(Y)



