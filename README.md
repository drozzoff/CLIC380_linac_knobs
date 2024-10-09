# CLIC380_linac_knobs

In this repository I store the emittance tuning knobs I constructed for the Main Linac of CLIC 380 GeV. Each branch hosts some version of the knobs. The name of branch indicates when this particular set was constructed. The full details on each knob set is given in the dedicated README. I try to accompany the knobs creation functions with some Notebooks, displaying the knobs' tests and their usage in the tuning.

### So far there are the following Knobs sets here:
- [2024_01_24](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_01_24) - A set constructed with **Sequential Forward Selection (SFS)** with a custom regularization. Regularization consists of penalties on:
	- the elements' offsets
	- the beam orbit in the ML.
- [2024_03_07](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07) - A set constructed with **Sequential Forward Selection (SFS)** with a custom regularization. Regularization consists of penalties on
	- the elements' offsets
	- the beam orbit in the ML.
	- the beam orbit at the ML exit.
- [LASSO](https://github.com/drozzoff/CLIC380_linac_knobs/tree/Lasso) - A set constructed using exactly the same `Tensorflow` model (*forward-backward*!) as in [2024_03_07](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07) but with **LASSO** as a feature selector.
- [OMP](https://github.com/drozzoff/CLIC380_linac_knobs/tree/OMP) - A set constructed using exactly the same `Tensorflow` model (*forward-backward*!) as in [2024_03_07](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07) but with **OMP** as a feature selector.
- [Variance Threshhold Filter](https://github.com/drozzoff/CLIC380_linac_knobs/tree/Variance_threshold) - A set constructed using exactly the same `Tensorflow` model (*forward-backward*!) as in [2024_03_07](https://github.com/drozzoff/CLIC380_linac_knobs/tree/2024_03_07) but with **Variance Threshold Filter** as a feature selector.