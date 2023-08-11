# README

This repository contains minimal code for running the numerical examples in the paper:

D. Pradovera and A. Borghi, _Match-based solution of general parametric eigenvalue problems_ (2023)

Preprint publicly available [here](https://arxiv.org/abs/2308.05335)!

## Prerequisites
* **numpy** and **scipy**
* **matplotlib**

## Execution
The two simulations in the paper are in the scripts `test_poly3.py` and `test_heat_delay.py`, respectively.

### "Poly3" example
The first example can be run as
```
python3 test_poly3.py
```

### "Heat_delay" example
The second example contains two different flavors of the simulation, which vary the tolerance level.
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
