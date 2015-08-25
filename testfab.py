import numpy as np
from hmmlearn.hmm import MultinomialHMM
from hmmlearn.fabhmm import MultinomialFABHMM


def sample(n_samples):
    n_components = 4
    startprob_ = np.array([1, 0, 0, 0])
    transmat_ = np.array([[0, 1, 1, 0],
                          [0, 0, 1, 1],
                          [1, 0, 0, 1],
                          [1, 1, 0, 0]]) / 2

    emissionprob_ = np.array([[1, 0, 0, 0, 0, 0, 1, 1],
                              [1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 1, 0]]) / 3

    model = MultinomialHMM(n_components=n_components)
    model.startprob_ = startprob_
    model.transmat_ = transmat_
    model.emissionprob_ = emissionprob_

    return model.sample(n_samples=n_samples)


def sample_rand(n_samples, n_components, n_features):
    startprob_ = np.random.rand(n_components)
    startprob_ = startprob_ / startprob_.sum()
    transmat_ = np.random.rand(n_components, n_components)
    transmat_ = transmat_ / transmat_.sum(1)[:, np.newaxis]

    emissionprob_ = np.random.rand(n_components, n_features)
    emissionprob_ = emissionprob_ / emissionprob_.sum(1)[:, np.newaxis]

    model = MultinomialHMM(n_components=n_components)
    model.startprob_ = startprob_
    model.transmat_ = transmat_
    model.emissionprob_ = emissionprob_

    return model.sample(n_samples=n_samples)


def main():
    N = 1
    X, state_sequence = sample(n_samples=500)

    acm = 0
    for _ in range(N):
        model = MultinomialFABHMM(n_components=20, n_iter=10000, tol=1e-5)
        model.fit(X)
        acm += model.n_components

        print(model.n_components)
    print(acm / N)
if __name__ == "__main__":
    main()
