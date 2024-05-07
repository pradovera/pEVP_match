# README

This repository contains minimal code for running the numerical examples in the paper:

D. Pradovera and A. Borghi, _Match-based solution of general parametric eigenvalue problems_ (2023)

Preprint publicly available [here](https://arxiv.org/abs/2308.05335)!

## Prerequisites
* **numpy** and **scipy**
* **matplotlib**
* (only for the "waveguide" example) **fenics** and **mshr**

## Execution
The four simulations in the paper are in the scripts `test_poly3.py`, `test_waveguide.py`, `test_heat_delay.py`, and `test_heat_delay_loop.py`.

### "Poly3" example
The first example can be run as
```
python3 test_poly3.py
```

### "Waveguide" example
The second example can be run as
```
python3 test_waveguide.py
```

### "Heat_delay" example
The third example contains two different flavors of the simulation, which vary the tolerance level.
To select the flavor, one can pass a label from command line, as
```
python3 test_heat_delay.py $example_tag
```
The placeholder `$example_tag` can take the values
* `NORMAL` (normal accuracy)
* `FINE` (higher accuracy)

Otherwise, one can simply run
```
python3 test_heat_delay.py
```
and then input `$example_tag` later.

### "Heat_delay_loop" example
The fourth example contains two different flavors of the simulation, which vary the interpolation strategy.
To select the flavor, one can pass a label from command line, as
```
python3 test_heat_delay_loop.py $example_tag
```
The placeholder `$example_tag` can take the values
* `LINEAR` (piecewise-linear interpolation)
* `SPLINE7` (degree-7 spline interpolation)

Otherwise, one can simply run
```
python3 test_heat_delay_loop.py
```
and then input `$example_tag` later.

### "Gun" example
The fifth example can be run as
```
python3 test_gun.py
```
NOTE: the file `gun_data.mat` is a copy of [this file](https://github.com/ftisseur/nlevp/blob/master/private/gun.mat) from the NLEVP repository (copyright Timo Betcke, Nicholas J. Higham, Volker Mehrmann, Gian Maria Negri Porzio, Christian Schroeder and Francoise Tisseur).
