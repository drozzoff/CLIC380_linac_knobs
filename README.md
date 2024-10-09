# Knobs constructed using the features selected with Orthogonal Matching Pursuit (OMP)

The knobs here were constructed using the features that were selected with **OMP** algorithm. The number elements chosen for the **OMP** is in the range from 1 to 60.

The model fitting is done in `Tensorflow` with the following hyperparameters:

|       **Parameter**      | **Value** | **Meaning** |
|:------------------------:|:---------:|:-----------:|
|       `error_scale`      |    5e-7   |             |
|    `orbit_error_scale`   |    1e-6   |             |
| `exit_orbit_error_scale` |    1e-8   |             |
|      `BOTTOM_LIMIT`      |     1     |    $\mu$m   |
|       `UPPER_LIMIT`      |    100    |    $\mu$m   |
|       `ORBIT_LIMIT`      |     60    |    $\mu$m   |
| `N_LAST_BPMS_TO_FLATTEN` |     2     |             |

Since **OMP** is a regression model with $l_0$-regularization, it provides the elements' offsets in the solution. So, 2 options were investigated:
- Where the offsets evaluated with **OMP** are transfered to `Tensorflow` model as initial guess. This is reffered to as *omp*.
- Where the offsets evaluated with **OMP** are ignored and are initiated by default with `0.0`. This is reffered to as *omp_zero*.

[Notebook](Knobs_omp_construction.ipynb) features the study summary.