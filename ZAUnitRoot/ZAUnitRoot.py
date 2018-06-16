import numpy as np
import pandas as pd
import sys
import os
import statsmodels.tsa.stattools as tsa
import statsmodels.tools.tools as tools
import statsmodels.regression.linear_model as lm

def za(x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
    """
    Zivot-Andrews structural-break unit-root test

    The Zivot-Andrews test can be used to test for a unit root in a
    univariate process in the presence of serial correlation and a
    single structural break.

    Parameters
    ----------
    x : array_like, 1d
        data series
    maxlag : int
        maximum lag which is included in test, default=12*(nobs/100)^{1/4} (Schwert, 1989)
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
    pvalue : float (TODO)
        based on MC-derived critical values
    bpidx : int
        index of x corresponding to endogenously calculated break period
    baselag : int
        number of lags used for period regressions

    Notes
    -----
    H0 = unit root with a single structural break

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
    x = np.asarray(x)
    nobs = x.shape[0]
    if autolag:
        baselags = tsa.adfuller(x[:,0], maxlag=maxlag, regression='ct', autolag=autolag)[2]
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
    dy = np.diff(x, axis=0)[:,0]
    zastat = bpidx = np.inf
    for bp in range(start_period, end_period+1):
        # reserve exog space
        exog = np.zeros((dy[baselags:].shape[0], basecols+baselags))
        # constant
        exog[:,0] = 1
        # intercept dummy / trend / trend dummy
        if regression != 't':
            exog[(bp-(baselags+1)):,1] = 1
            exog[:,2] = np.arange(baselags+2, nobs+1)
            if regression == 'ct':
                exog[(bp-(baselags+1)):,3] = np.arange(1, nobs-bp+1)
        else:
            exog[:,1] = np.arange(baselags+2, nobs+1)
            exog[(bp-(baselags+1)):,2] = np.arange(1, nobs-bp+1)
        # lagged y
        exog[:,basecols-1] = x[baselags:(nobs-1),0]
        # lagged dy
        exog[:,basecols:] = tsa.lagmat(dy, baselags, trim='none')[baselags:exog.shape[0]+baselags]
        stat = lm.OLS(dy[baselags:], exog).fit().tvalues[basecols-1]
        if stat < zastat:
            zastat = stat
            bpidx = bp - 1
    return zastat, baselags, bpidx

def main():
    print("Zivot-Andrews unit-root test")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "..\\results\\")
    files = ['gnpdef.csv', 'rgnp.csv', 'rgnpq.csv', 'stkprc.csv']
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        if file == 'gnpdef.csv' or file == 'rgnp.csv':
            res = za(mdl, regression='c')
        elif file == 'rgnpq.csv':
            res = za(mdl, regression='t')
        elif file == 'stkprc.csv':
            res = za(mdl, regression='ct')
        print("  zastat =", res[0], " lags =", res[1], " break_idx =", res[2])

if __name__ == "__main__":
    sys.exit(int(main() or 0))