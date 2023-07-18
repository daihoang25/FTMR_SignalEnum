# Coherent Signal Enumeration based on Deep Learning and the FTMR Algorithm
We introduce the logarithmic eigenvalue-based classification network (LogECNet) to detect the signal number. In the proposed scheme, the full-row Toeplitz matrices reconstruction (FTMR) algorithm is employed to avoid the rank loss of the signal covariance matrix (SCM) in highly correlated signal environments. The simulation results show that the FTMR method not only achieves the complexity reduction with respect to the prior forward/backward spatial smoothing (FBSS) algorithm but also improves the signal number detection performance when combined with LogECNet. 

We note that the FTMR Algorithm is implemented in the 'data_toeplitz' function in 'data_gerator.py'.

If you use this code in your research, please cite the following papers:

[1] D. T. Hoang and K. Lee, ‘‘Deep learning-aided coherent direction-of-arrival estimation with the FTMR algorithm,’’ IEEE Trans. Signal Process., vol. 70, pp. 1118–1130, 2022.
[2] D. T. Hoang and K. Lee, ‘‘Coherent signal enumeration based on deep learning and the FTMR algorithm,’’ in Proc. IEEE Int. Conf. Commun., Seoul, South Korea, May 2022, pp. 5098–510.


