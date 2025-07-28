# Mossbauer Analysis Framework

Trying to make the analysis framework for the Mossbauer experiment centralized and comprehensible to others.


## Setup
In your favored environment (e.g. an anaconda env), do:

```
pip install -r requirements.txt
pip install -e .
```

Then anywhere, you should be able to do in python:
```
import mossbauer

my_measurement = mossbauer.MossbauerMeasurement(
    source_parameters,
    absorber_parameters,
    measurement_parameters,
)
```

See the docstrings for what these need to contain.
