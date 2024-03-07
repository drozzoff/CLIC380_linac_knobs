# Emittance tuning knobs for the Main Linac of CLIC 380 GeV.

This a version from **2024/03/07**.

The knobs uploaded here is the further development from the knobs given in [2024_01_24](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_01_24). The difference inroduced here are the following:

- The beam orbit that is not penalyzed is increased from **20 micron** to **40 micron**. But the hyperparameter for the *orbit_loss* is increased.
- Additional regularization called *exit_orbit_loss* is introduced. It penalyzes the beam orbit at the ML exit. Doing so, we make sure the knobs do not introduce any significant beam orbit at the ML exit. Making sure it is so is very important. Unlike the BBA, the knobs are to be used with the luminosity as a figure of merit. BBA on the other hand is to be performed from ML and then on BDS, thus any beam orbit at the ML exit can be corrected afterwards. With the knobs, we do not have such option. So, if the knob introduces significant beam orbit it is going to lead to the luminosity drop. From the estimation, **50 nm** exit orbit leads to **0.02 nm** offset at the IP. That means with the exit orbit of ~ **0.5 micron** (~ 0.2 mn), we could start seeing lumi drop (the simulations will follow). So, we want to keep these values as small as possible. In [2024_01_24](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_01_24), the exit orbit is somewhat random as it is not controlled and varies in the range **1-10 micron** for majority of the knobs and couple tens for the problematic knobs such as **Y7**, **Y8**, and **Y10**.

***Also, another difference is that, I introduce the last quads to every calcualtion by default***. These quads are important for the exit orbit correction are would be added in the calculation anyway.

To constract the knobs, I follow the same procedure described in [2024_01_24](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_01_24), the script used for the calculations is [here](learning_model_february_parallel.py).

The summary table with the optimum setup for the knobs is:

| Knob | N_features |   Score  |
|:----:|:----------:|:--------:|
|  Y1  |     12     | 0.962125 |
|  Y2  |     15     | 0.964892 |
|  Y3  |     10     | 0.970607 |
|  Y4  |     ..     | ........ |
|  Y5  |     ..     | ........ |
|  Y6  |     ..     | ........ |
|  Y7  |     ..     | ........ |
|  Y8  |     ..     | ........ |
|  Y9  |     ..     | ........ |
|  Y10 |     ..     | ........ |

More detailed info on how the knobs are constructed is here:

### Knob **Y1**
```
features = [2061, 2062, 2027, 1763, 1909, 1970, 1728, 1617, 2038, 1589, 1659, 1563]
offsets = [1.0034031, -2.4138765, 29.554003, 6.9769125, -11.943935, -8.755246, -1.7526971, 12.514738, -4.8155403, 8.288669, 3.7992966, 2.559954] 
```
[Here](Y1_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob **Y2**
```
features = [2061, 2062, 2021, 1639, 2040, 2059, 2057, 2058, 2028, 2055, 1635, 1611, 1594, 1568, 1674]
offsets = [2.8762712, -7.17054, 21.905436, -10.040492, -9.202621, 26.251575, 57.0944, 23.385427, 6.566971, -48.30085, -17.444534, 18.316664, 4.3894773, 1.0602893, -1.0055443] 
```
[Here](Y2_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob **Y3**
```
features = [2061, 2062, 1495, 1521, 2059, 2039, 2060, 1491, 1543, 1532, 1993, 2043, 2003]
offsets = [-1.84305, -8.012489, 3.6600113, 1.0008427, 27.782095, 26.244236, 7.5782285, 1.1914557, -1.9112644, 1.0029606, -1.0098681, 23.005943, 1.002427]
```
[Here](Y3_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob  **Y4**

*to be updated*

**Note:** the beamline that is used to create the knobs should be the same as the one that was used to construct the knobs on. **Otherwise it does not make any sense.**
