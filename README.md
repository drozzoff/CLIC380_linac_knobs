# # Optimal emittance tuning knobs for the Main Linac of CLIC 380 GeV.

This a version from **2024/10/09**.

The set of the knobs presented here is the final version of the **Optimal Knobs** that feature all the requirements.

The key part of the knobs construction routine is the linear model defined `Tensorflow` that feature multiple addition regularizations and limits:
- Offset of each element concerned must be larger than `BOTTOM_LIMIT` and smaller than `UPPER_LIMIT`. The strength of the bottom penalty is defined by the hyperparameter `error_scale`. The upper limit is defined as a constraint.
- Beam orbit must remain in the range defined by `ORBIT_LIMIT`. The strength of this regularization is defined by `orbit_error_scale`.
- Beam orbit at the Main Linac exit must remain zero. The strength of this regularization is defined by `exit_orbit_error_scale`.

The full list of the parameters is given:
|       **Parameter**      | **Value** | **Meaning** |
|:------------------------:|:---------:|:-----------:|
|       `error_scale`      |    5e-7   |             |
|    `orbit_error_scale`   |    1e-6   |             |
| `exit_orbit_error_scale` |    1e-8   |             |
|      `BOTTOM_LIMIT`      |    1(10)  |    $\mu$m   |
|       `UPPER_LIMIT`      |    100    |    $\mu$m   |
|       `ORBIT_LIMIT`      |     60    |    $\mu$m   |
| `N_LAST_BPMS_TO_FLATTEN` |     2     |             |

To evaluate the optimal elements to use for the knobs, multiple feature selection routines were tested including [**LASSO**](https://github.com/drozzoff/CLIC380_linac_knobs/tree/Lasso), [**OMP**](https://github.com/drozzoff/CLIC380_linac_knobs/tree/OMP), [**Variance threshold Filter**](https://github.com/drozzoff/CLIC380_linac_knobs/tree/Variance_threshold), and [**Sequential Feature Selector**](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07). There are several **SFS** tests in the repository, but [2024_03_07](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07) is supposed to be used, the other does not have exit orbit penalty in the model. 

Typically **SFS** provides the best performance, but it failed to provide a proper solution for the knobs **Y7** and **Y8**. So for **Y7** and **Y8** the features evaluated with **OMP** are used. The summary table of the knobs is given below:

Knob **Y1**
```
features = [2061, 2062, 2027, 1763, 1909, 1970, 1728, 1617, 2038, 1589, 1659, 1563]
offsets = [1.0034031, -2.4138765, 29.554003, 6.9769125, -11.943935, -8.755246, -1.7526971, 12.514738, -4.8155403, 8.288669, 3.7992966, 2.559954] 
```

Knob **Y2**
```
features = [2061, 2062, 2021, 1639, 2040, 2059, 2057, 2058, 2028, 2055, 1635, 1611, 1594, 1568, 1674]
offsets = [2.8762712, -7.17054, 21.905436, -10.040492, -9.202621, 26.251575, 57.0944, 23.385427, 6.566971, -48.30085, -17.444534, 18.316664, 4.3894773, 1.0602893, -1.0055443] 
```

Knob **Y3**
```
features = [2061, 2062, 1495, 1521, 2059, 2039, 2060, 1491, 1543, 1532, 1993, 2043, 2003]
offsets = [-1.84305, -8.012489, 3.6600113, 1.0008427, 27.782095, 26.244236, 7.5782285, 1.1914557, -1.9112644, 1.0029606, -1.0098681, 23.005943, 1.002427]
```

Knob **Y4**
```
features = [2061, 2062, 1493, 1515, 2060, 1526, 1504, 1489, 2059]
offsets = [-20.725388, -5.951438, 1.0033137, -6.2002664, -6.281371, 1.0087113, 1.2091062, -1.00304, -1.0033016]
```

Knob **Y5**
```
features = [2061, 2062, 1687, 2059, 2043, 1729, 1645, 1568, 1605, 1557, 2054, 1553, 1531]
offsets = [52.589233, 24.192154, -6.794914, -53.63966, 4.5129313, -3.119878, -1.0049086, -4.1625276, -5.0811276, 5.888484, 6.1651525, -1.9204162, 0.9997077]
```

Knob **Y6**
```
features = [2061, 2062, 1689, 2059, 1693, 1705, 2044, 1748, 1623, 1569, 2029, 1595, 1539, 1606, 1666, 1907, 1535, 1895, 2058, 1669, 1732, 1517, 1509]
offsets = [1.007775, -1.0060145, -23.89543, -11.346015, -24.16606, -4.691198, -10.451449, -1.0043415, -1.7761153, 22.075333, -16.784285, 29.727722, -18.901419, -7.284757, 1.8254659, 15.494361, -9.863385, 11.994422, -1.315509, -1.9800675, 1.9732033, 2.7335136, -0.9997426]
```


Knob **Y7**
```
features = [1, 3, 30, 1475, 1483, 1490, 1492, 1498, 1510, 1512, 1546, 1548, 1578, 1580, 1598, 1600, 1632, 1634, 1674, 1676, 1750, 1752, 1822, 1824, 1860, 1888, 1902, 1904, 1940, 1946, 1992, 2028, 2034, 2060, 2062]
offsets = [1.0032152, -1.0005312, -1.0110478, -1.0085683, -1.000856, 2.288177, 1.0651124, -5.883464, -3.5267127, 3.0554736, 3.151293, 2.4057226, -1.69496, 5.0512385, 2.6482906, 1.0050241, 3.5532622, -1.0044675, 1.9221967, -1.2278516, 1.0021572, -1.291427, 8.162435, -1.0033009, -9.288089, 1.0031923, -1.2224729, 1.0013938, -5.3106327, -1.0040396, 7.0271564, -5.249662, 7.524716, 3.6393862, -1.0083928]
```

Knob **Y8**
```
features = [0, 3, 13, 41, 1475, 1483, 1490, 1492, 1494, 1514, 1524, 1544, 1564, 1580, 1586, 1602, 1604, 1654, 1656, 1704, 1706, 1750, 1752, 1798, 1800, 1838, 1848, 1880, 1890, 1902, 1904, 1986, 1988, 2034, 2060, 2062]
offsets = [3.4103918, 3.2223666, 3.9751186, 1.0034472, 1.4816511, 4.9501753, 5.2467346, 1.0040987, 6.9957967, 1.0014651, -4.800699, -6.8043847, -14.458933, 10.858159, 11.560809, -6.6181216, 3.4727547, 4.95835, 1.2718078, 6.602847, -1.0045012, -0.99859405, -1.0027006, 2.7748594, 9.469465, -7.181139, -1.0646541, -1.6479657, -4.6633325, -1.268393, 5.0901155, 1.0261942, -10.321095, 13.488639, 5.241846, 3.7375462]
```

Knob **Y9**
```
features = [2061, 2062, 1589, 1713, 1559, 2059, 1755, 1490, 1501, 2060, 2035, 1523, 1767, 1512, 1541, 1625, 1686, 1489, 1997]
offsets = [-52.929802, -16.318533, 14.455154, -1.0021956, -20.712362, 3.9437888, -9.480821, -3.8107805, 19.33842, -14.172591, -3.035246, -38.7336, -10.866332, 4.275113, -30.68692, 13.717008, 1.2279714, 2.7988489, 1.0618254]
```

Knob **Y10**
```
feature = [2061, 2062, 1589, 1713, 1559, 2059, 1755, 1490, 1501, 2060, 2035, 1523, 1767, 1512, 1541, 1625, 1686, 1489, 1997, 1644, 1721, 1605, 1745, 1630, 1680]
offsets = [-32.787884, -9.262848, 15.856654, -3.5411239, -19.070496, 3.2364144, -12.878025, -3.281437, 19.582462, -8.2011595, -8.383424, -29.706821, -20.148241, 1.0778022, -26.771458, 14.183673, 4.3581166, 2.77738, 5.216679, 3.7812219, 6.17213, 3.0716708, -2.305692, 2.2195277, 1.0227758]
```

The elements' ids have the following meaning:
- `[0, 1488]` are the girders' ids
- `[1489, 2063]` are the quadrupoles ids starting the ML entrance.
