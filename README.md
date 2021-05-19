# Mathematical-Modeling-of-COVID-19
A SEIR model has been developed to understand the increase and spread of the coronavirus.
The SEIR model is based on python API called lmfit. It uses the technique of non-linear least-square minimization and curve fitting. Parameters such as transmission rate(beta), incubation rate(sigma) and mortality rate(gamma) is estimated using lmfit. With these parameters a model can predict the pattern of the number of cases.




The SEIR Simulation file is used to plot the observed versus fitted number of cases of two categories: 'Infected' and 'Recovered/Deceased'. The timespan and number of days in the code can be tuned to plot time series graphs for every three months, starting March 2020 to understand how the transmission rate changes every three months. 
