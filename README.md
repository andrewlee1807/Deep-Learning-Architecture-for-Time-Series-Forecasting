# Deep-Learning-Architecture-for-Time-Series-Forecasting



# Initialized environment
```shell
# using pip
pip install -r environment.yml
# using Conda
conda create --name ts_model --file environment.yml
conda activate ts_model
pip install -r requirements.txt
```
## Verify
```shell
python -c "import tensorflow as tf; tf.test.is_gpu_available()"
```

## Export environment
```shell
conda env create --file environment.yml
```

# Components

1. Data module
   1. Load data: via name of dataset
      1. read configuration file
   2. Util create time series form
   3. Pre-processing:
      1. Split data into Train/Val/Test
      2. fill_missing
      3. Normalize data
      4. ....
2. Visualization module:
   1. Training process
   2. Inferences results
   3. Raw sequence data
3. Methods module
   1. Comparison methods: List of comparison methods
   2. Proposed methods:
      1. Version 1
      2. Version 2
      3. ....
   