# CLIC380_linac_knobs

This a version from **2024/10/09**.

This branch features the knobs construction summary using the ***Variance Threshold Filter**. Since this method of the feature selection only relies on the data it fails to predict all relations for the proper reduction of the regularization loss. And consequently the model fitting fails for the features selected with **Threshold Variance Filter**. 

The model fitting is done in Tensorflow with the following hyperparameters:

|       **Parameter**      | **Value** | **Meaning** |
|:------------------------:|:---------:|:-----------:|
|       `error_scale`      |    5e-7   |             |
|    `orbit_error_scale`   |    1e-6   |             |
| `exit_orbit_error_scale` |    1e-8   |             |
|      `BOTTOM_LIMIT`      |     1     |    $\mu$m   |
|       `UPPER_LIMIT`      |    100    |    $\mu$m   |
|       `ORBIT_LIMIT`      |     60    |    $\mu$m   |
| `N_LAST_BPMS_TO_FLATTEN` |     2     |             |

The threshold for the ***Variance Threshold Filter** was iterated in the certain range to get the good number of features.
[Notebook](Knobs_variance_threshold_construction.ipynb) features the scan summaries using the data generated from the fits. The data itself is stored in the corresponding folders.