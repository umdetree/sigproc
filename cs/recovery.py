import numpy as np
import scipy.special as special


def somp_hybrid(A: np.ndarray, Y: np.ndarray, thresh: float, s: int) -> set[int]:
    """
    Consider the following linear equation: Y = AX

    Using Simultaneous Orthogonal Matching Pursuit algorithm to get the support
    of `X`, where parameter `thresh` is the threshold to stop iteration, `s` is
    the maximum sparsity
    """
    if thresh <= 0.0 or thresh > 1.0:
        raise ValueError("threshold should be in range (0, 1]")

    (row, col) = A.shape

    res = np.matrix(Y)
    support = set()
    supportBar = set(np.arange(0, col, 1))
    ASelected = np.matrix(np.zeros((row, 1), dtype=np.cdouble))

    k = 0
    e0 = np.linalg.norm(res)
    while np.linalg.norm(res) / e0 > thresh and k < s:
        norms = np.linalg.norm(res.H * np.matrix(A[:, list(supportBar)]), axis=0, ord=2)
        index = list(supportBar)[np.argmax(norms)]

        supportBar.remove(index)
        support.add(index)

        if k == 0:
            ASelected = A[:, index]
        else:
            ASelected = np.matrix(np.hstack((ASelected, A[:, index])))

        P = ASelected * np.linalg.inv(ASelected.H * ASelected) * ASelected.H
        res = np.matrix(Y - P * Y)

        k += 1

    return support


def somp_sparsity(A: np.ndarray, Y: np.ndarray, s: int) -> set:
    """
    Consider following linear equation: Y = AX

    Using Simultaneous Orthogonal Matching Pursuit algorithm to get the support
    of `X`, where parameter `s` is the sparsity of signal `X`
    """
    (_, col) = A.shape
    if s > col:
        raise ValueError("sparsity should not be greater than X rows")

    res = np.matrix(Y)
    support = set()
    supportBar = set(np.arange(0, col, 1))
    ASelected = np.matrix(np.zeros((1, col), dtype=np.cdouble))

    for k in range(s):
        norms = np.linalg.norm(res.H @ A[:, list(supportBar)], axis=0, ord=2)
        index = list(supportBar)[np.argmax(norms)]

        supportBar.remove(index)
        support.add(index)

        if k == 0:
            ASelected = np.matrix(A[:, index])
        else:
            ASelected = np.matrix(np.hstack((ASelected, A[:, index])))

        P = ASelected @ np.linalg.inv(ASelected.H @ ASelected) @ ASelected.H
        res = np.matrix(Y - P @ Y)

    return support


def recover_with_support(Y: np.ndarray, A: np.ndarray, support: set[int]):
    """
    Use Moore-Penrose pseudo inversion to recover signal

    #### parameters

    - `Y`: Measured signals
    - `A`: Measurement matrix

    #### returns

    The recovered original signals `X`
    """
    (M, N) = A.shape
    if M > N:
        raise ValueError("A's rows are greater than cols, there is not need for CS")
    P = Y.shape[1]

    supp = list(support)
    As = np.matrix(A[:, supp])

    xs = np.linalg.lstsq(As, Y, rcond=None)[0]
    Xr = np.matrix(np.zeros((N, P), dtype=np.cdouble))
    Xr[supp, :] = xs

    return Xr


def recover(Y: np.ndarray, A: np.ndarray, sparsity=None):
    """
    Use eBIC and SOMP to recover signal

    #### parameters

    - `Y`: Measured signals
    - `A`: Measurement matrix

    #### returns

    The recovered original signals `X`
    """

    A = np.matrix(A)
    Y = np.matrix(Y)
    (M, N) = A.shape
    if M > N:
        raise ValueError("A's rows are greater than cols, there is not need for CS")

    s, _, Reduced = eBIC(Y)
    if sparsity is not None:
        s = sparsity

    support = somp_sparsity(A, Reduced, s)
    Xr = recover_with_support(Y, A, support)

    return Xr, support


def eBIC(Y: np.matrix, s: int = 0):
    """
    Use enhanced Bayesian-Information Criterion to return the approximated
    sparsity and the reduced Y

    Only do matrix reduction if `s` is set to positive integer

    #### Returns:

    - approximated sparsity
    - eBIC np.ndarray of each k
    - the reduced Y
    """
    (M, P) = Y.shape
    Ry = 1 / P * Y * Y.H
    (eigVals, eigVecs) = np.linalg.eigh(Ry)
    eigVals = np.abs(eigVals)
    sortedIdx = eigVals.argsort()[::-1]
    eigVals = eigVals[sortedIdx]
    eigVecs = eigVecs[:, sortedIdx]

    ebics = np.zeros((M - 1))
    if s == 0:
        logEigVals = np.log(eigVals)
        logGammas = special.gammaln(np.arange(1, M + 1, 1))
        cumSumLogEigVals = np.cumsum(logEigVals)
        cumSumLogEigValsRev = np.cumsum(logEigVals[::-1])
        cumSumLogGammas = np.cumsum(logGammas)

        diffs = np.subtract.outer(eigVals, eigVals)
        for k in range(M - 1):
            sigma2 = np.mean(eigVals[k + 1 :])
            term1 = np.sum((eigVals[k + 1 : M] / sigma2 - 1) ** 2)

            term2 = np.log2(
                np.abs(diffs[k + 1 :, k + 1 :][np.triu_indices(M - k - 1, 1)])
            ).sum()

            ebicK = (
                2 * (M - k - 1) * (P + M - k - 1) * np.log(sigma2)
                - 2 * P * cumSumLogEigValsRev[-k - 2]
                + 2 * cumSumLogEigVals[k]
                + 2 * cumSumLogGammas[M - k - 2]
                + P * term1
                - 4 * term2
                + (4 * M * (k + 1) - 2 * (k + 1) ** 2 - k - 1) * np.log(P)
            )

            ebics[k] = ebicK

        s = np.argmin(ebics) + 1

    Reduced = np.matrix(eigVecs[:, :s] @ np.diag(np.sqrt(eigVals[:s])))

    return s, ebics, Reduced
