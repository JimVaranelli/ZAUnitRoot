# ZAUnitRoot / ZAUnitRootMP
Python implementation of Zivot-Andrews structural-break unit-root test. Multi-processing version included for large-sample series.

## References
Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews unit root test in presence of structural break," Statistical Software Components S437301, Boston College Department of Economics, revised 2015.

Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business & Economic Statistics, 7: 147-159.

Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Studies, 10: 251-270.

## Description
A Python program that implements the Zivot-Andrews structural-break unit-root test.

H0 : series has a unit-root in the presence of a single structural break

## Requirements
Python 3.6
Numpy 1.13.1
Statsmodels 0.9.0
Pandas 0.20.3

## Running
There are no parameters. The program is set up to access a test file in the ..\results directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced output for the test file.
