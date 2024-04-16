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
|  Y4  |      9     | 0.954275 |
|  Y5  |     13     | 0.972522 |
|  Y6  |     23     | 0.964387 |
|  Y7  |     ..     | ........ |
|  Y8  |     ..     | ........ |
|  Y9  |     19     | 0.962303 |
|  Y10 |     25     | 0.956407 |

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
```
features = [2061, 2062, 1493, 1515, 2060, 1526, 1504, 1489, 2059]
offsets = [-20.725388, -5.951438, 1.0033137, -6.2002664, -6.281371, 1.0087113, 1.2091062, -1.00304, -1.0033016]
```
[Here](Y4_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob  **Y5**
```
features = [2061, 2062, 1687, 2059, 2043, 1729, 1645, 1568, 1605, 1557, 2054, 1553, 1531]
offsets = [52.589233, 24.192154, -6.794914, -53.63966, 4.5129313, -3.119878, -1.0049086, -4.1625276, -5.0811276, 5.888484, 6.1651525, -1.9204162, 0.9997077]
```
[Here](Y5_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob  **Y6**
```
features = [2061, 2062, 1689, 2059, 1693, 1705, 2044, 1748, 1623, 1569, 2029, 1595, 1539, 1606, 1666, 1907, 1535, 1895, 2058, 1669, 1732, 1517, 1509]
offsets = [1.007775, -1.0060145, -23.89543, -11.346015, -24.16606, -4.691198, -10.451449, -1.0043415, -1.7761153, 22.075333, -16.784285, 29.727722, -18.901419, -7.284757, 1.8254659, 15.494361, -9.863385, 11.994422, -1.315509, -1.9800675, 1.9732033, 2.7335136, -0.9997426]
```
[Here](Y6_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob  **Y7**
This knob cannot be matched with the conditions used in this study. Multiple hyperparameters were used with little success. The optimization always fails to reduce the exit orbit to the acceptable level.

### Knob  **Y8**
This knob cannot be matched with the conditions used in this study. Multiple hyperparameters were used with little success. The optimization always fails to reduce the exit orbit to the acceptable level.

### Knob  **Y9**
```
features = [2061, 2062, 1589, 1713, 1559, 2059, 1755, 1490, 1501, 2060, 2035, 1523, 1767, 1512, 1541, 1625, 1686, 1489, 1997]
offsets = [-52.929802, -16.318533, 14.455154, -1.0021956, -20.712362, 3.9437888, -9.480821, -3.8107805, 19.33842, -14.172591, -3.035246, -38.7336, -10.866332, 4.275113, -30.68692, 13.717008, 1.2279714, 2.7988489, 1.0618254]
```
[Here](Y9_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

### Knob  **Y10**
```
feature = [2061, 2062, 1589, 1713, 1559, 2059, 1755, 1490, 1501, 2060, 2035, 1523, 1767, 1512, 1541, 1625, 1686, 1489, 1997, 1644, 1721, 1605, 1745, 1630, 1680]
offsets = [-32.787884, -9.262848, 15.856654, -3.5411239, -19.070496, 3.2364144, -12.878025, -3.281437, 19.582462, -8.2011595, -8.383424, -29.706821, -20.148241, 1.0778022, -26.771458, 14.183673, 4.3581166, 2.77738, 5.216679, 3.7812219, 6.17213, 3.0716708, -2.305692, 2.2195277, 1.0227758]
```
[Here](Y10_convergence_study.ipynb) is the jupyter notebook with more details of the scan.

**Note:** the beamline that is used to create the knobs should be the same as the one that was used to construct the knobs on. **Otherwise it does not make any sense.**
