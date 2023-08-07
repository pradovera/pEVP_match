# README
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
The second example contains three different flavors of the simulation, which vary the interpolation strategy and the tolerance level.
To select the flavor, one can pass a label from command line, as
```
python3 test_heat_delay.py $example_tag
```
The placeholder `$example_tag` can take the values
* `SPLINE` (splines, normal accuracy)
* `PIECEWISELINEAR` (piecewise-linear functions, normal accuracy)
* `SPLINE_FINE` (splines, higher accuracy)

Otherwise, one can simply run
```
python3 test_heat_delay.py
```
and then input `$example_tag` later.
