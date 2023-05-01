# Deep-Learning-Architecture-for-Time-Series-Forecasting

# Initialized environment (Linux OS)

```shell
# using Conda
conda env create -f environment.yaml  # Check the name of environment before import
conda activate ts_model
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

```
### Notice: 
We are using TensorFlowV2.11 in order to use Keras-TCN library. So, there are some expected issues installation.
## Verify

```shell
python -c "import tensorflow as tf; tf.test.is_gpu_available()"
```

## Pack the environment
[Conda environment](https://github.com/grst/containerize-conda/tree/master/conda-pack) can't be just "moved" to another location, as some paths are hardcoded into the environment. conda-pack takes care of replacing these paths back to placeholders and creates a .tar.gz archive that contains the environment

`conda install -c conda-forge conda-pack`

`conda pack -n ts_model`
```shell
Collecting packages...
Packing environment at 'C:\\Users\\Andrew\\anaconda3\\envs\\ts_model' to 'ts_model.tar.gz'
[########################################] | 100% Completed |  3min 42.6s
```
- To use the environment, unpack the archive and run the activate script:

```shell
tar -xzf ts_model.tar.gz
source ts_model/bin/activate # Linux version
ts_model\Scripts\activate.bat # Windows version
```

# Docker build

```shell
docker build -t test .
docker login docker.io
docker images
docker tag test docker.io/hoanganh97/test

```
```shell
docker push lehoanganh97/test

Using default tag: latest
The push refers to repository [docker.io/lehoanganh97/test]
97f6d08e40b1: Pushed
c3e5be8ff12b: Pushed
c0b232971146: Pushed
aa0032bce758: Pushed
48df0d7cfecb: Pushed
529f4a059361: Pushed
e468d78ea69e: Pushed
219c6c2423f1: Pushed
3af14c9a24c9: Pushed
latest: digest: sha256:2491e14b89ecebdd36b7042ff165d2a9a8ac1e2afbcb0afed7c183f039e0d4f3 size: 2208
```


# Tips

- Git repository stop being tracked for changes

```shell
git update-index --assume-unchanged <file_path>
```

- Export package list only installed by `pip` in linux:
```shell
pip freeze | grep -v "@ file://" > requirements.txt
```
- Export package list only installed by `pip` in windows:
```shell
pip freeze > requirements.txt
type requirements.txt | findstr /V /C:"@ file://" > requirements_filtered.txt
```





# Components

1. Data module
    1. Load data: via name of dataset
        - read configuration file
    2. Util create time series form
    3. Pre-processing:
       The procedure of data preparation:
        1. Split data into TRAIN and TEST

           `[num_record, timeseries-past, feature, timeseries-label]`

                example: [16752, X] -> [15076, X], [1676, X]

        2. Building the Time series data type:

           `[num_record, timeseries-past, feature, timeseries-label]`

               example (1093, 168, 1, 7)
        3. Normalize data
        4. Split train data into `TRAIN` and `VALID`
        5. Normalize data via `TRAIN` data
2. Visualization module:
    1. Training process
    2. Inferences results
    3. Raw sequence data
3. Method module
    1. Comparison methods: List of comparison methods
    2. Proposed methods:
        1. Version 1
        2. Version 2
        3. ....
    3. Baseline methods:
        1. LSTM
        2. TCN
        3. TCN-Deep
        4. ...
   