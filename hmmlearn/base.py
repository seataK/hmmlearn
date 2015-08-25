from __future__ import print_function

import string
import sys
from collections import deque

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from . import _hmmc
from .utils import normalize, logsumexp, iter_from_X_lengths


DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class ConvergenceMonitor(object):
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.
    """
    fmt = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 1

    def report(self, logprob):
        if self.history and self.verbose:
            delta = logprob - self.history[-1]
            message = self.fmt.format(
                iter=self.iter, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        return (self.iter == self.n_iter or
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.tol))


class _BaseHMM(BaseEstimator):
    """Hidden Markov Model base class.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string, one of the ``DECODER_ALGORITHMS```
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    """

    # This class implements the public interface to all HMMs that
    # derive from it, including all of the machinery for the
    # forward-backward and Viterbi algorithms.  Subclasses need only
    # implement _generate_sample_from_state(), _compute_log_likelihood(),
    # _init(), _initialize_sufficient_statistics(),
    # _accumulate_sufficient_statistics(), and _do_mstep(), all of
    # which depend on the specific emission distribution.
    #
    # Subclasses will probably also want to implement properties for
    # the emission distribution parameters to expose them publicly.

    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 model_select=False):
        self.n_components = n_components
        self.params = params
        self.init_params = init_params
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.model_select = model_select

    def score_samples(self, X, lengths=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        posteriors = np.zeros((n_samples, self.n_components))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij

            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    def score(self, X, lengths=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij
        return logprob

    def _decode_viterbi(self, X):
        framelogprob = self._compute_log_likelihood(X)
        return self._do_viterbi_pass(framelogprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        algorithm : string, one of the ``DECODER_ALGORITHMS``
            decoder algorithm to be used

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {0!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j])
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, lengths=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, lengths)
        return state_sequence

    def predict_proba(self, X, lengths=None):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If ``None``, the object's
            random_state is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        check_is_fitted(self, "startprob_")

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

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
        l_logprob = []
        l_ficlb = []
        l_n_components = []
        l_delta = []
        l_bp = []
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
                l_logprob.append(logprob)
                ficlb += sh_reg[1]
                ficlb += sh_reg[0] * (j - i - 1)
                l_delta.append(sh_reg[1] + sh_reg[0] * (j - i - 1))
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

            beta_phi = -np.sum(term_beta + term_phi)
            l_bp.append(beta_phi)
            l_ficlb.append(ficlb)

            l_n_components.append(self.n_components)

            self.monitor_.report(ficlb)
            if self.monitor_.converged and count is 0:
                import pylab as plt
                a_ficlb = np.array(l_ficlb)
                a_logprob = np.array(l_logprob)
                a_n_components = np.array(l_n_components)
                a_delta = np.array(l_delta)
                a_bp = np.array(l_bp)
                plt.subplot(2,2,1)
                plt.plot(np.arange(a_ficlb.shape[0]), a_ficlb, label='ficlb')
                plt.plot(np.arange(a_logprob.shape[0]), a_logprob, label='lobprob')
                plt.legend(fontsize=8)
                plt.subplot(2,2,2)
                plt.plot(np.arange(a_delta.shape[0]), a_delta, label='delta')
                plt.legend(fontsize=8)
                plt.subplot(2,2,3)
                plt.plot(np.arange(a_n_components.shape[0]), a_n_components)
                plt.subplot(2,2,4)
                plt.plot(np.arange(a_bp.shape[0]), a_bp)
                plt.savefig('fig.png')
                break

        return self

    def _do_viterbi_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_observations, n_components, np.log(self.startprob_),
            np.log(self.transmat_), framelogprob)
        return logprob, state_sequence

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

        return count

    def _do_forward_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_observations, n_components))
        _hmmc._forward(n_observations, n_components, np.log(self.startprob_),
                       np.log(self.transmat_), framelogprob, fwdlattice)
        return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_observations, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_observations, n_components))
        _hmmc._backward(n_observations, n_components, np.log(self.startprob_),
                        np.log(self.transmat_), framelogprob, bwdlattice)
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

    def _compute_log_likelihood(self, X):
        pass

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _init(self, X, lengths, params):
        init = 1. / self.n_components
        if 's' in params or not hasattr(self, "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if 't' in params or not hasattr(self, "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components),
                                     init)

    def _check(self):
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {0:.4f})"
                             .format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {0})"
                             .format(self.transmat_.sum(axis=1)))

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, seq, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            n_observations, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_observations <= 1:
                return

            lneta = np.zeros((n_observations - 1, n_components, n_components))
            _hmmc._compute_lneta(n_observations, n_components, fwdlattice,
                                 np.log(self.transmat_),
                                 bwdlattice, framelogprob, lneta)
            stats['trans'] += np.exp(logsumexp(lneta, axis=0))

    def _do_mstep(self, stats, params):
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        if 's' in params:
            self.startprob_ = self.startprob_prior - 1.0 + stats['start']
            normalize(self.startprob_)
        if 't' in params:
            self.transmat_ = self.transmat_prior - 1.0 + stats['trans']
            normalize(self.transmat_, axis=1)
