import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import statsmodels.regression.linear_model as lm
import multiprocessing as mp
import time
import ZACriticalValues as zacrit

def _za_thread(x, regression, start, end, nobs, basecols, baselags, res, residx):
    # first-diff y
    dy = np.diff(x, axis=0)[:,0]
    zastat = bpidx = np.inf
    for bp in range(start, end):
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
            crit = zacrit.za_crit(zastat, regression)
            pval = crit[0]
            cvdict = crit[1]
    res[residx] = [zastat, pval, cvdict, bpidx]

def zaMP(x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
    """
    Zivot-Andrews structural-break unit-root test (multi-processing)

    The Zivot-Andrews test can be used to test for a unit root in a
    univariate process in the presence of serial correlation and a
    single structural break.

    Parameters
    ----------
    x : array_like
        data series
    trim : float
        percentage of series at begin/end to exclude from break-period
        calculation in range [0, 0.25] (default=0.15)
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
    original Zivot-Andrews method although no attempt has been made to
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
    if not isinstance(trim, float) or trim < 0 or trim > 0.25:
        raise ValueError('ZA: trim value must be a float in range [0, 0.25]')
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
    end_period = nobs - trimcnt + 1
    if regression == 'ct':
        basecols = 5
    else:
        basecols = 4
    # set up multi-processing - use 3/4 available CPUs/cores
    print(" Number of available CPUs = ", mp.cpu_count())
    numprocs = int(3 * mp.cpu_count() / 4)
    if numprocs < 1:
        numprocs = 1
    print("  Number of CPUs utilized = ", numprocs)
    numperproc = (end_period - start_period) / numprocs
    rem = (end_period - start_period) % numprocs
    procs = [None] * numprocs
    manager = mp.Manager()
    res = manager.dict()
    startper = start_period
    if numprocs > 1:
        endper = startper + int(numperproc)
    else:
        endper = end_period
    tot = startper + numperproc
    for procidx in range(len(procs)):
        print("   Launching process: procidx =", procidx, "startper =", startper, "endper =", endper)
        procs[procidx] = mp.Process(target=_za_thread, args=(x, regression, startper, endper, nobs, basecols, baselags, res, procidx))
        procs[procidx].start()
        startper = endper
        endper = startper + int(numperproc)
        tot += numperproc
        if tot - endper >= 1:
            endper += 1
    for proc in range(len(procs)):
        procs[proc].join()
    rv = np.asarray(res.values())
    rvmin = rv[np.argmin(rv[:,0]), :]
    return rvmin[0], rvmin[1], rvmin[2], baselags, int(rvmin[3])

# rgnp.csv: zastat = -5.57615  pval = 0.00312  lags = 8  break_idx = 20
# gnpdef.csv: zastat = -4.12155  pval = 0.28024  lags = 5  break_idx = 40
# stkprc.csv: zastat = -5.60689  pval = 0.00894  lags = 1  break_idx = 65
# rgnpq.csv: zastat = -3.02761  pval = 0.63993  lags = 12  break_idx = 101
# rand10000.csv: zastat = -3.48223  pval = 0.69111  lags = 25  break_idx = 7071
def main():
    print("Zivot-Andrews structural-break unit-root test (MP version)")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "..\\results\\")
    files = ['rgnp.csv', 'gnpdef.csv', 'stkprc.csv', 'rgnpq.csv']
    files = ['rand10000.csv']
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'rgnp.csv':
            res = zaMP(mdl, maxlag=8, regression='c', autolag=None)
        elif file == 'gnpdef.csv':
            res = zaMP(mdl, maxlag=8, regression='c', autolag='t-stat')
        elif file == 'stkprc.csv':
            res = zaMP(mdl, maxlag=8, regression='ct', autolag='t-stat')
        elif file == 'rgnpq.csv':
            res = zaMP(mdl, maxlag=12, regression='t', autolag='t-stat')
        else:
            res = zaMP(mdl, regression='c', autolag='t-stat')
        print("  zastat =", "{0:0.5f}".format(res[0]), " pval = ", "{0:0.5f}".format(res[1]))
        print("    cvdict =", res[2])
        print("    lags =", res[3], " break_idx =", res[4], " time =", "{0:0.5f}".format(time.time()-st))

if __name__ == "__main__":
    sys.exit(int(main() or 0))
