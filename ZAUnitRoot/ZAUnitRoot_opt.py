from __future__ import division

import sys
import os
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import time
import ZACriticalValues as zacrit


def _quick_ols(endog, exog):
    """Minimal implementation of LS estimator for internal use"""
    xpxi = np.linalg.inv(exog.T @ exog)
    xpy = exog.T @ endog
    nobs, k_exog = exog.shape
    b = xpxi @ xpy
    e = endog - exog @ b
    sigma2 = e.T @ e / (nobs - k_exog)
    return b / np.sqrt(np.diag(sigma2 * xpxi))


def za(x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
    """
    Zivot-Andrews structural-break unit-root test

    The Zivot-Andrews test can be used to test for a unit root in a
    univariate process in the presence of serial correlation and a
    single structural break.

    Parameters
    ----------
    x : array_like
        data series
    trim : float
        percentage of series at begin/end to exclude from break-period
        calculation in range [0, 0.333] (default=0.15)
    maxlag : int
        maximum lag which is included in test, default=12*(nobs/100)^{1/4}
        (Schwert, 1989)
    regression : {'c','t','ct'}
        Constant and trend order to include in regression
        * 'c' : constant only (default)
        * 't' : trend only
        * 'ct' : constant and trend
    autolag : {'AIC', 'BIC', 't-stat', None}
        * if None, then maxlag lags are used
        * if 'AIC' (default) or 'BIC', then the number of lags is chosen
          to minimize the corresponding information criterion
        * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test

    Returns
    -------
    zastat : float
        test statistic
    pvalue : float
        based on MC-derived critical values
    cvdict : dict
        critical values for the test statistic at the 1%, 5%, and 10% levels
    bpidx : int
        index of x corresponding to endogenously calculated break period
    baselag : int
        number of lags used for period regressions

    Notes
    -----
    H0 = unit root with a single structural break

    Algorithm follows Baum (2004/2015) approximation to original Zivot-Andrews
    method. Rather than performing an autolag regression at each candidate
    break period (as per the original paper), a single autolag regression is
    run up-front on the base model (constant + trend with no dummies) to
    determine the best lag length. This lag length is then used for all
    subsequent break-period regressions. This results in significant run time
    reduction but also slightly more pessimistic test statistics than the
    original Zivot-Andrews method, although no attempt has been made to
    characterize the size/power tradeoff.

    References
    ----------
    Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews unit
    root test in presence of structural break," Statistical Software Components
    S437301, Boston College Department of Economics, revised 2015.

    Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo investigation.
    Journal of Business & Economic Statistics, 7: 147-159.

    Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great crash,
    the oil-price shock, and the unit-root hypothesis. Journal of Business &
    Economic Studies, 10: 251-270.
    """

    if regression not in ['c', 't', 'ct']:
        raise ValueError('ZA: regression option \'%s\' not understood' % regression)
    if not isinstance(trim, float) or trim < 0 or trim > (1. / 3.):
        raise ValueError('ZA: trim value must be a float in range [0, 0.333]')
    x = np.asarray(x)
    if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
        raise ValueError('x must be a 1d array or an 2d array with a single column')
    x = np.reshape(x, (-1, 1))
    nobs = x.shape[0]
    if autolag:
        baselags = tsa.adfuller(x[:, 0], maxlag=maxlag, regression='ct', autolag=autolag)[2]
    elif maxlag:
        baselags = maxlag
    else:
        baselags = int(12. * np.power(nobs / 100., 1 / 4.))
    trimcnt = int(nobs * trim)
    start_period = trimcnt
    end_period = nobs - trimcnt
    if regression == 'ct':
        basecols = 5
    else:
        basecols = 4
    # first-diff y
    dy = np.diff(x, axis=0)[:, 0]
    # Standardize for numerical stability in long time series
    dy /= np.sqrt(dy.T.dot(dy))
    x = x / np.sqrt(x.T.dot(x))
    # reserve exog space
    exog = np.zeros((dy[baselags:].shape[0], basecols + baselags))
    # constant
    c_const = 1 / np.sqrt(nobs)  # Normalize for stability in long time series
    exog[:, 0] = c_const
    # lagged y
    exog[:, basecols - 1] = x[baselags:(nobs - 1), 0]
    # lagged dy
    exog[:, basecols:] = tsa.lagmat(dy, baselags, trim='none')[baselags:exog.shape[0] + baselags]
    # Better time trend, t_const @ t_const = 1 for large nobs
    t_const = np.arange(1.0, nobs + 2)
    t_const *= np.sqrt(3) / nobs ** (3 / 2)

    stats = np.full(end_period + 1, np.inf)
    for bp in range(start_period + 1, end_period + 1):
        # intercept dummy / trend / trend dummy
        cutoff = (bp - (baselags + 1))
        if regression != 't':
            exog[:cutoff, 1] = 0
            exog[cutoff:, 1] = c_const
            exog[:, 2] = t_const[(baselags + 2):(nobs + 1)]
            if regression == 'ct':
                exog[:cutoff, 3] = 0
                exog[cutoff:, 3] = t_const[1:nobs - bp + 1]
        else:
            exog[:, 1] = t_const[baselags + 2: nobs + 1]
            exog[:cutoff, 2] = 0
            exog[cutoff:, 2] = t_const[1:nobs - bp + 1]
        stats[bp] = _quick_ols(dy[baselags:], exog)[basecols - 1]

    zastat = np.min(stats)
    bpidx = np.argmin(stats) - 1
    crit = zacrit.za_crit(zastat, regression)
    pval = crit[0]
    cvdict = crit[1]

    return zastat, pval, cvdict, baselags, bpidx


# rgnp.csv: zastat = -5.57615  pval = 0.00312  lags = 8  break_idx = 20
# gnpdef.csv: zastat = -4.12155  pval = 0.28024  lags = 5  break_idx = 40
# stkprc.csv: zastat = -5.60689  pval = 0.00894  lags = 1  break_idx = 65
# rgnpq.csv: zastat = -3.02761  pval = 0.63993  lags = 12  break_idx = 101
# rand10000.csv: zastat = -3.48223  pval = 0.69111  lags = 25  break_idx = 7071
def main():
    print("Zivot-Andrews structural-break unit-root test")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "..\\results\\")
    files = ['rgnp.csv', 'gnpdef.csv', 'stkprc.csv', 'rgnpq.csv', 'rand10000.csv']
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'rgnp.csv':
            res = za(mdl, maxlag=8, regression='c', autolag=None)
        elif file == 'gnpdef.csv':
            res = za(mdl, maxlag=8, regression='c', autolag='t-stat')
        elif file == 'stkprc.csv':
            res = za(mdl, maxlag=8, regression='ct', autolag='t-stat')
        elif file == 'rgnpq.csv':
            res = za(mdl, maxlag=12, regression='t', autolag='t-stat')
        else:
            res = za(mdl, regression='c', autolag='t-stat')
        print("  zastat =", "{0:0.5f}".format(res[0]), " pval =", "{0:0.5f}".format(res[1]))
        print("    cvdict =", res[2])
        print("    lags =", res[3], " break_idx =", res[4], " time =",
              "{0:0.5f}".format(time.time() - st))


if __name__ == "__main__":
    sys.exit(int(main() or 0))
