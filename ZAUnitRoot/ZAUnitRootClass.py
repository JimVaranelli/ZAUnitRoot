import sys
import os
import time
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as tsa
import statsmodels.regression.linear_model as lm
from numpy.testing import (assert_almost_equal, assert_equal)

class ur_za(object):
    """
    Class wrapper for Zivot-Andrews structural-break unit-root test
    """
    def __init__(self):
        """
        Critical values for the three different models specified for the
        Zivot-Andrews unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using 100,000
        replications and 2000 data points.
        """
        self.__za_critical_values = {}
        # constant-only model
        self.__c = ((0.001, -6.78442), (0.100, -5.83192), (0.200, -5.68139),
             (0.300, -5.58461), (0.400, -5.51308), (0.500, -5.45043),
             (0.600, -5.39924), (0.700, -5.36023), (0.800, -5.33219),
             (0.900, -5.30294), (1.000, -5.27644), (2.500, -5.03340),
             (5.000, -4.81067), (7.500, -4.67636), (10.000, -4.56618),
             (12.500, -4.48130), (15.000, -4.40507), (17.500, -4.33947),
             (20.000, -4.28155), (22.500, -4.22683), (25.000, -4.17830),
             (27.500, -4.13101), (30.000, -4.08586), (32.500, -4.04455),
             (35.000, -4.00380), (37.500, -3.96144), (40.000, -3.92078),
             (42.500, -3.88178), (45.000, -3.84503), (47.500, -3.80549),
             (50.000, -3.77031), (52.500, -3.73209), (55.000, -3.69600),
             (57.500, -3.65985), (60.000, -3.62126), (65.000, -3.54580),
             (70.000, -3.46848), (75.000, -3.38533), (80.000, -3.29112),
             (85.000, -3.17832), (90.000, -3.04165), (92.500, -2.95146),
             (95.000, -2.83179), (96.000, -2.76465), (97.000, -2.68624),
             (98.000, -2.57884), (99.000, -2.40044), (99.900, -1.88932), )
        self.__za_critical_values['c'] = np.asarray(self.__c)
        # trend-only model
        self.__t = ((0.001, -83.9094), (0.100, -13.8837), (0.200, -9.13205),
             (0.300, -6.32564), (0.400, -5.60803), (0.500, -5.38794),
             (0.600, -5.26585), (0.700, -5.18734), (0.800, -5.12756),
             (0.900, -5.07984), (1.000, -5.03421), (2.500, -4.65634),
             (5.000, -4.40580), (7.500, -4.25214), (10.000, -4.13678),
             (12.500, -4.03765), (15.000, -3.95185), (17.500, -3.87945),
             (20.000, -3.81295), (22.500, -3.75273), (25.000, -3.69836),
             (27.500, -3.64785), (30.000, -3.59819), (32.500, -3.55146),
             (35.000, -3.50522), (37.500, -3.45987), (40.000, -3.41672),
             (42.500, -3.37465), (45.000, -3.33394), (47.500, -3.29393),
             (50.000, -3.25316), (52.500, -3.21244), (55.000, -3.17124),
             (57.500, -3.13211), (60.000, -3.09204), (65.000, -3.01135),
             (70.000, -2.92897), (75.000, -2.83614), (80.000, -2.73893),
             (85.000, -2.62840), (90.000, -2.49611), (92.500, -2.41337),
             (95.000, -2.30820), (96.000, -2.25797), (97.000, -2.19648),
             (98.000, -2.11320), (99.000, -1.99138), (99.900, -1.67466), )
        self.__za_critical_values['t'] = np.asarray(self.__t)
        # constant + trend model
        self.__ct = ((0.001, -38.17800), (0.100, -6.43107), (0.200, -6.07279),
              (0.300, -5.95496), (0.400, -5.86254), (0.500, -5.77081),
              (0.600, -5.72541), (0.700, -5.68406), (0.800, -5.65163),
              (0.900, -5.60419), (1.000, -5.57556), (2.500, -5.29704),
              (5.000, -5.07332), (7.500, -4.93003), (10.000, -4.82668),
              (12.500, -4.73711), (15.000, -4.66020), (17.500, -4.58970),
              (20.000, -4.52855), (22.500, -4.47100), (25.000, -4.42011),
              (27.500, -4.37387), (30.000, -4.32705), (32.500, -4.28126),
              (35.000, -4.23793), (37.500, -4.19822), (40.000, -4.15800),
              (42.500, -4.11946), (45.000, -4.08064), (47.500, -4.04286),
              (50.000, -4.00489), (52.500, -3.96837), (55.000, -3.93200),
              (57.500, -3.89496), (60.000, -3.85577), (65.000, -3.77795),
              (70.000, -3.69794), (75.000, -3.61852), (80.000, -3.52485),
              (85.000, -3.41665), (90.000, -3.28527), (92.500, -3.19724),
              (95.000, -3.08769), (96.000, -3.03088), (97.000, -2.96091),
              (98.000, -2.85581), (99.000, -2.71015), (99.900, -2.28767), )
        self.__za_critical_values['ct'] = np.asarray(self.__ct)

    def __za_crit(self, stat, model='c'):
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        stat : float
            The ZA test statistic
        model : {'c','t','ct'}
            The model used when computing the ZA statistic. 'c' is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        table = self.__za_critical_values[model]
        y = table[:, 0]
        x = table[:, 1]
        # ZA cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, y, x)
        cvdict = {"1%" : crit_value[0], "5%" : crit_value[1],
                  "10%" : crit_value[2]}
        return pvalue, cvdict

    def __quick_ols(self, endog, exog):
        """
        Minimal implementation of LS estimator for internal use
        """
        xpxi = np.linalg.inv(exog.T.dot(exog))
        xpy = exog.T.dot(endog)
        nobs, k_exog = exog.shape
        b = xpxi.dot(xpy)
        e = endog - exog.dot(b)
        sigma2 = e.T.dot(e) / (nobs - k_exog)
        return b / np.sqrt(np.diag(sigma2 * xpxi))

    def run(self, x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
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
            critical values for the test statistic at the 1%, 5%, and 10%
            levels
        bpidx : int
            index of x corresponding to endogenously calculated break period
            with values in the range [0..nobs-1]
        baselag : int
            number of lags used for period regressions

        Notes
        -----
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original
        Zivot-Andrews method. Rather than performing an autolag regression at
        each candidate break period (as per the original paper), a single
        autolag regression is run up-front on the base model (constant + trend
        with no dummies) to determine the best lag length. This lag length is
        then used for all subsequent break-period regressions. This results in
        significant run time reduction but also slightly more pessimistic test
        statistics than the original Zivot-Andrews method, although no attempt
        has been made to characterize the size/power tradeoff.

        References
        ----------
        Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews
        unit root test in presence of structural break," Statistical Software
        Components S437301, Boston College Department of Economics, revised
        2015.

        Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
        investigation. Journal of Business & Economic Statistics, 7: 147-159.

        Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great
        crash, the oil-price shock, and the unit-root hypothesis. Journal of
        Business & Economic Studies, 10: 251-270.
        """
        if regression not in ['c', 't', 'ct']:
            raise ValueError(
                'ZA: regression option \'%s\' not understood' % regression)
        if not isinstance(trim, float) or trim < 0 or trim > (1. / 3.):
            raise ValueError(
                'ZA: trim value must be a float in range [0, 0.333]')
        x = np.asarray(x)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError(
                'ZA: x must be a 1d array or a 2d array with a single column')
        x = np.reshape(x, (-1, 1))
        nobs = x.shape[0]
        if autolag:
            baselags = tsa.adfuller(x[:, 0], maxlag=maxlag, regression='ct',
                                    autolag=autolag)[2]
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
        # first-diff y and standardize for numerical stability
        dy = np.diff(x, axis=0)[:, 0]
        dy /= np.sqrt(dy.T.dot(dy))
        x = x / np.sqrt(x.T.dot(x))
        # reserve exog space
        exog = np.zeros((dy[baselags:].shape[0], basecols + baselags))
        # normalize constant for stability in long time series
        c_const = 1 / np.sqrt(nobs)  # Normalize
        exog[:, 0] = c_const
        # lagged y and dy
        exog[:, basecols - 1] = x[baselags:(nobs - 1), 0]
        exog[:, basecols:] = tsa.lagmat(
            dy, baselags, trim='none')[baselags:exog.shape[0] + baselags]
        # better time trend: t_const @ t_const = 1 for large nobs
        t_const = np.arange(1.0, nobs + 2)
        t_const *= np.sqrt(3) / nobs ** (3 / 2)
        # iterate through the time periods
        stats = np.full(end_period + 1, np.inf)
        for bp in range(start_period + 1, end_period + 1):
            # update intercept dummy / trend / trend dummy
            cutoff = (bp - (baselags + 1))
            if regression != 't':
                exog[:cutoff, 1] = 0
                exog[cutoff:, 1] = c_const
                exog[:, 2] = t_const[(baselags + 2):(nobs + 1)]
                if regression == 'ct':
                    exog[:cutoff, 3] = 0
                    exog[cutoff:, 3] = t_const[1:(nobs - bp + 1)]
            else:
                exog[:, 1] = t_const[(baselags + 2):(nobs + 1)]
                exog[:(cutoff-1), 2] = 0
                exog[(cutoff-1):, 2] = t_const[0:(nobs - bp + 1)]
            # check exog rank on first iteration
            if bp == start_period + 1:
                o = lm.OLS(dy[baselags:], exog, hasconst=1).fit()
                if o.df_model < exog.shape[1] - 1:
                    raise ValueError(
                        'ZA: auxiliary exog matrix is not full rank.\n \
                        cols (exc intercept) = {}  rank = {}'.format(
                            exog.shape[1] - 1, o.df_model))
                stats[bp] = o.tvalues[basecols - 1]
            else:
                stats[bp] = self.__quick_ols(dy[baselags:], exog)[basecols - 1]
        # return best seen
        zastat = np.min(stats)
        bpidx = np.argmin(stats) - 1
        crit = self.__za_crit(zastat, regression)
        pval = crit[0]
        cvdict = crit[1]
        return zastat, pval, cvdict, baselags, bpidx

    def __call__(self, x, trim=0.15, maxlag=None, regression='c',
                 autolag='AIC'):
        return self.run(x, trim=trim, maxlag=maxlag, regression=regression,
                        autolag=autolag)

za = ur_za()
za.__doc__ = za.run.__doc__

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
            assert_almost_equal(res[0], -5.57615, decimal=3)
            assert_almost_equal(res[1], 0.00312, decimal=3)
            assert_equal(res[4], 20)
        elif file == 'gnpdef.csv':
            res = za(mdl, maxlag=8, regression='c', autolag='t-stat')
            assert_almost_equal(res[0], -4.12155, decimal=3)
            assert_almost_equal(res[1], 0.28024, decimal=3)
            assert_equal(res[3], 5)
            assert_equal(res[4], 40)
        elif file == 'stkprc.csv':
            res = za(mdl, maxlag=8, regression='ct', autolag='t-stat')
            assert_almost_equal(res[0], -5.60689, decimal=3)
            assert_almost_equal(res[1], 0.00894, decimal=3)
            assert_equal(res[3], 1)
            assert_equal(res[4], 65)
        elif file == 'rgnpq.csv':
            res = za(mdl, maxlag=12, regression='t', autolag='t-stat')
            assert_almost_equal(res[0], -3.02761, decimal=3)
            assert_almost_equal(res[1], 0.63993, decimal=3)
            assert_equal(res[3], 12)
            assert_equal(res[4], 102)
        else:
            res = za(mdl, regression='c', autolag='t-stat')
            assert_almost_equal(res[0], -3.48223, decimal=3)
            assert_almost_equal(res[1], 0.69111, decimal=3)
            assert_equal(res[3], 25)
            assert_equal(res[4], 7071)
        print("  zastat =", "{0:0.5f}".format(res[0]), " pval =", "{0:0.5f}".format(res[1]))
        print("    cvdict =", res[2])
        print("    lags =", res[3], " break_idx =", res[4], " time =",
              "{0:0.5f}".format(time.time() - st))

if __name__ == "__main__":
    sys.exit(int(main() or 0))
