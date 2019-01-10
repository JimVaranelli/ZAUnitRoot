# ZAUnitRoot / ZAUnitRootMP
Python implementation of Zivot-Andrews structural-break unit-root test. Multi-processing version included for large-sample series. Version contained in ZAUnitRoot/ZAUnitRootClass.py has been submitted to statsmodels package.

## Parameters
x : array_like \
&nbsp;&nbsp;&nbsp;&nbsp;data series \
trim : float \
&nbsp;&nbsp;&nbsp;&nbsp;percentage of series at begin/end to exclude from break-period \
&nbsp;&nbsp;&nbsp;&nbsp;calculation in range [0, 0.333] (default=0.15) \
maxlag : int \
&nbsp;&nbsp;&nbsp;&nbsp;maximum lag which is included in test, default=12*(nobs/100)^{1/4} \
&nbsp;&nbsp;&nbsp;&nbsp;(Schwert, 1989) \
regression : {'c','t','ct'} \
&nbsp;&nbsp;&nbsp;&nbsp;Constant and trend order to include in regression \
&nbsp;&nbsp;&nbsp;&nbsp;'c' : constant only (default) \
&nbsp;&nbsp;&nbsp;&nbsp;'t' : trend only \
&nbsp;&nbsp;&nbsp;&nbsp;'ct' : constant and trend \
autolag : {'AIC', 'BIC', 't-stat', None} \
&nbsp;&nbsp;&nbsp;&nbsp;- if None, then maxlag lags are used \
&nbsp;&nbsp;&nbsp;&nbsp;- if 'AIC' (default) or 'BIC', then the number of lags is chosen \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to minimize the corresponding information criterion \
&nbsp;&nbsp;&nbsp;&nbsp;- 't-stat' based choice of maxlag.  Starts with maxlag and drops a \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;lag until the t-statistic on the last lag length is significant \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;using a 5%-sized test

## Returns
zastat : float \
&nbsp;&nbsp;&nbsp;&nbsp;test statistic \
pvalue : float \
&nbsp;&nbsp;&nbsp;&nbsp;based on MC-derived critical values \
cvdict : dict \
&nbsp;&nbsp;&nbsp;&nbsp;critical values for the test statistic at the 1%, 5%, and 10% \
&nbsp;&nbsp;&nbsp;&nbsp;levels \
bpidx : int \
&nbsp;&nbsp;&nbsp;&nbsp;index of x corresponding to endogenously calculated break period \
&nbsp;&nbsp;&nbsp;&nbsp;with values in the range [0..nobs-1] \
baselag : int \
&nbsp;&nbsp;&nbsp;&nbsp;number of lags used for period regressions

## Notes
Critical values for the three different models are generated through Monte Carlo simulation using 100,000 replications and 2000 data points

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

## References
Baum, C.F. (2004/2015). ZANDREWS: Stata module to calculate Zivot-Andrews unit root test in presence of structural break," Statistical Software Components S437301, Boston College Department of Economics, revised 2015.

Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business & Economic Statistics, 7: 147-159.

Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Studies, 10: 251-270.

## Requirements
Python 3.6 \
Numpy 1.13.1 \
Statsmodels 0.9.0 \
Pandas 0.20.3

## Running
There are no parameters. The program is set up to access a test file in the ..\results directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced output for the test file.
