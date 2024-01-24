# Emittance tuning knobs for the Main Linac of CLIC 380 GeV.

This a version from **2024/01/24**.

The knobs used here were constructed using the **Sequential Forward Selector** described here [here](https://indico.cern.ch/event/1335148/contributions/5662745/attachments/2769351/4824771/CLIC_week_11_12_2023.pptx). It is a presentation on the "Emittance tuning knobs for CLIC ML" given during CLIC Mini Week 2023.

In short I do the following:

- On top of the model prediction missmatch I regularize the elements' offsets with 2 regularizations.
- I penalyze the small offsets, with **L2-like** regularization around 0.0. The offsets larger than 1 $\mu$ m m are not prenalyzed. The penalty applied is:
```Python
filtered_weights = tf.where(tf.abs(weights) < BOTTOM_LIMIT, tf.abs(weights) - BOTTOM_LIMIT, tf.zeros_like(weights))

zero_penalty = tf.abs(tf.reduce_mean(filtered_weights))
```
here, `BOTTOM_LIMIT` (in my context it is equal to 1 $\mu$ m) is the bottom limit for the offsets after which they are penalyzed.
- I also penalyze the offsets in a way to minimize the beam orbit along the Main Linac. Assuming the response is linear, I predict the beam orbit through the response matrix `R_orbit_tensor`. The beam orbit that exceeds the limit is penalyzed. The penalty is evaluated like this:
```Python
orbit_vector = tf.linalg.matvec(R_orbit_tensor, tf.squeeze(weights, axis=-1))

filtered_vector = tf.where(abs(orbit_vector) > ORBIT_LIMIT, abs(orbit_vector) - ORBIT_LIMIT, tf.zeros_like(orbit_vector))

orbit_penalty = tf.reduce_mean(filtered_vector)
```
here `ORBIT_LIMIT` is the beam orbit above which the beam orbit start to be penalyzed. In my case it is 20 $\mu$ m.

In the study above, I set the maximum number of quadrupoles for the **SFS** to 20. For most of the knobs it is more than enough to have a decent solution. But in the case of the knobs **Y7**, **Y8**, and **Y10** there are still some spikes (around 60-100 $\mu$ m), usually in just one place.

## Based on the results of the SFS, I picked the solutions for the knobs. 

I tried to base my desicion on the factors like score, number of the elements involved, presence of the regularization loss, etc. This is a setup for thise set of knobs:

| Knob | N_features |   Score  |
|:----:|:----------:|:--------:|
|  Y1  |     13     | 0.973669 |
|  Y2  |     19     | 0.992812 |
|  Y3  |     11     | 0.966055 |
|  Y4  |      8     | 0.953420 |
|  Y5  |     20     | 0.972630 |
|  Y6  |     15     | 0.984363 |
|  Y7  |     11     | 0.926810 |
|  Y8  |     19     | 0.956145 |
|  Y9  |     20     | 0.944068 |
|  Y10 |     19     | 0.931552 |

The smallest score obtained is indeed in the case of **Y7**, **Y8**, and **Y10** knobs. I believe I could extend the number of elements to 30 in the SFS run and see if I can find a solution with good score + no orbit spikes.

The solution used for the knobs here looks like this. Let's call it a `knobs_setup` variable:
```Python
{'Y1': {'quads': [12410, 4162, 13425, 7394, 3898, 7680, 12060, 7108, 2378, 4050, 4834, 3214, 4386], 'offsets': [-1.4499317, 1.0086112, -1.7835302, 17.416288, 14.735407, -4.104488, -1.6451167, -3.3707256, 0.999744, -24.63238, 10.131407, -1.4434994, -15.551787]}, 'Y2': {'quads': [13495, 12200, 1846, 1770, 1238, 1050, 11395, 13460, 13285, 12865, 13390, 13355, 640, 2587, 3898, 5562, 6536, 600, 1998], 'offsets': [-12.644044, -12.95743, 3.972991, 1.1433855, 5.6239467, 1.6334957, -1.0004653, -42.10206, 1.0103716, 7.0255275, 44.215057, 8.905501, -1.7946887, -2.7856796, -5.251406, 9.241636, -2.1251884, -1.0008932, -8.53229]}, 'Y3': {'quads': [60, 320, 13495, 20, 100, 0, 250, 300, 13390, 12060, 12725], 'offsets': [13.627413, 4.406359, -1.1989676, 1.1548543, 15.497834, 3.5694447, 3.163236, -1.0031986, -1.0209639, 5.8221865, -4.190667]}, 'Y4': {'quads': [0, 220, 330, 700, 960, 40, 1276, 440], 'offsets': [-1.0776453, 5.5204387, -1.0026138, -6.105201, -3.4961488, 2.2045083, -1.2163846, -4.6496086]}, 'Y5': {'quads': [2682, 13495, 2758, 3214, 6290, 3955, 13390, 6510, 8070, 13460, 13355, 3879, 5030, 3651, 13425, 13005, 380, 1160, 640, 8044], 'offsets': [-5.31099, 10.06933, -1.0023825, 5.31446, -1.0042615, -4.1056385, -76.65949, 3.180749, -3.7422626, 30.972218, -18.13835, -1.0085361, 1.5229276, 4.934485, -11.300055, 1.3797066, 0.99944574, -2.768395, 2.5680046, -1.2737335]}, 'Y6': {'quads': [2720, 13495, 13425, 12900, 2796, 3632, 5282, 1694, 13145, 880, 1180, 2131, 1257, 1466, 13390], 'offsets': [-1.0169718, -7.845782, -9.1169, -15.570315, -2.0200326, -1.7169768, -1.0010285, 5.445716, -9.554711, 1.483227, -16.670923, 2.6429608, 13.437764, -6.4317274, -22.504269]}, 'Y7': {'quads': [13495, 1238, 2112, 3708, 940, 1675, 13250, 13460, 13215, 9680, 5282], 'offsets': [-19.430765, -1.0039295, 7.7423434, -1.4062268, -0.9985544, 3.0697238, 67.31701, 88.028946, 22.387413, 2.5665972, 1.0084234]}, 'Y8': {'quads': [760, 13460, 12900, 13425, 7082, 500, 5730, 1390, 10625, 13390, 13215, 11290, 1020, 2910, 8420, 1751, 13355, 13495, 20], 'offsets': [-1.5487037, -100.0, 1.0131336, -1.007933, 2.4393258, -0.9984688, 2.6975286, 1.0089015, -4.2822146, 65.114975, -1.0120906, 12.185797, 2.7766652, 3.4222758, 1.6023216, 2.225133, 13.608629, 4.709115, 0.35653883]}, 'Y9': {'quads': [1276, 3024, 960, 12410, 1200, 13075, 7706, 10135, 1352, 0, 220, 13320, 440, 8018, 2663, 8805, 110, 13390, 3100, 4386], 'offsets': [1.0003417, -1.7573906, 1.0007737, -13.056088, -1.0022963, 7.6475673, -3.996771, 1.3432167, -3.5233371, 1.0020424, -5.1186123, 15.581015, 2.126359, 6.231844, -5.36884, 4.578079, 1.2001678, 7.3867984, -13.798097, 4.0949006]}, 'Y10': {'quads': [11710, 13495, 2416, 440, 1180, 550, 660, 13320, 1333, 260, 2131, 330, 4666, 270, 2492, 8910, 1140, 3727, 13390], 'offsets': [-1.1069509, -8.66518, 1.4460169, 17.944693, 21.14466, -10.533144, 17.144577, 6.9123564, 24.125498, -12.981194, 1.0085223, -7.424932, 1.0070262, -1.0012834, 12.713249, 1.241692, -8.958331, -1.4129344, 7.449947]}}
```
*I think, this could stored as a json instead*

To build this set of knobs, I do the following:

```Python
import placetmachine as pl
import typing

def construct_knobs(beamline: pl.Beamline) -> List[pl.Knob]:

    knobs = []

    for knob in knobs_setup:
        elements_list = []
        for quad_id in knobs_setup[knob]['quads]:
            elements_list.append(beamline[quad_id])

        knobs.append(pl.Knob(elements_list, 'y', knobs_setup[knob]['offsets], name = knob))
    return knobs
```

**Note:** the beamline that is used to create the knobs should be the same as the one that was used to construct the knobs on. **Otherwise it does not make any sense.**