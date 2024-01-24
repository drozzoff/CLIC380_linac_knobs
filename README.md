# Emittance tuning knobs for the Main Linac of CLIC 380 GeV.

This a version from **2024/01/24**.

The knobs used here were constructed using the **Sequential Forward Selector** described here [here](https://indico.cern.ch/event/1335148/contributions/5662745/attachments/2769351/4824771/CLIC_week_11_12_2023.pptx). It is a presentation on the "Emittance tuning knobs for CLIC ML" given during CLIC Mini Week 2023.

In short I do the following:

- On top of the model prediction missmatch I regularize the elements' offsets with 2 regularizations.
- I penalyze the small offsets, with **L2-like** regularization around 0.0. The offsets larger than 1$\mu$m are not prenalyzed. The penalty applied is:
```
filtered_weights = tf.where(tf.abs(weights) < BOTTOM_LIMIT, tf.abs(weights) - BOTTOM_LIMIT, tf.zeros_like(weights))

zero_penalty = tf.abs(tf.reduce_mean(filtered_weights))
```
here, `BOTTOM_LIMIT` (in my context it is equal to 1 $\mu$m) is the bottom limit for the offsets after which they are penalyzed.
- I also penalyze the offsets in a way to minimize the beam orbit along the Main Linac. Assuming the response is linear, I predict the beam orbit through the response matrix `R_orbit_tensor`. The beam orbit that exceeds the limit is penalyzed. The penalty is evaluated like this:
```
orbit_vector = tf.linalg.matvec(R_orbit_tensor, tf.squeeze(weights, axis=-1))

filtered_vector = tf.where(abs(orbit_vector) > ORBIT_LIMIT, abs(orbit_vector) - ORBIT_LIMIT, tf.zeros_like(orbit_vector))

orbit_penalty = tf.reduce_mean(filtered_vector)
```
here `ORBIT_LIMIT` is the beam orbit above which the beam orbit start to be penalyzed. In my case it is 20 $\mu$m.