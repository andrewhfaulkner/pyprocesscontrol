My vision for this project is to provide a python package that can do most of the process controls
calculations for you. This project will include things like plotting packages, calculations,
statistics, and other useful tools for process controls and improvement uses.

For a minumum viable product, the package must include

1. Control chart building and plotting <- Done
2. Time series data collection and filtering 
3. Overlay quality data with time-series process data
4. Statistics calulations such as mean and standard deviation across multiple specs <- Done w/ metadata class
5. Hypothesis testing for time period changes, machine-to-machine changes, normality, regression, etc.
6. Pair plotting variables and selecting relationships with positive or negative regressions
7. Data storage and collection <- Done
8. Time series plotting and calculations. I.e. changes over time.
    - time-phased control charts are complete. <-done
    - time-phased regression (i.e decoupling two variables etc.)


Future versions may include:
1. Model building with scikitlearn or tensorflow
2. DOE development