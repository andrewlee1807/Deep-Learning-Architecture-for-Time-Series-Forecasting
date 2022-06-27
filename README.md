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
