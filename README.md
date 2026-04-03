# Benchmark Project

## Getting Started

After activating the `benchmark` conda environment, run:

```bash
python getting_started.py
```

This is the simplest end-to-end example in the repository. It uses the `adult`
dataset and the `wachter` method so a new user can see the full benchmark
pipeline in one place.

The script walks through:

- loading the Adult dataset
- normalizing numerical features
- one-hot encoding categorical features
- splitting the data into train and test sets
- training a linear target model
- generating counterfactuals with Wachter
- evaluating the results with validity and distance

The example is tuned for readability and a short first run. It is not intended
to reproduce paper results.

If your environment reports that Python or Torch cannot find a usable temporary
directory, run:

```bash
TMPDIR=/tmp python getting_started.py
```
