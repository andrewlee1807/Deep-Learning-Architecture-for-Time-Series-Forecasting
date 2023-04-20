#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 24; do
  echo "Starting to train model with output length = $i"
  python main.py \
          --dataset_name="CNU_ENGINEERING_7" \
          --model_name="lstm" \
          --config_path="multivariate_ts/config/baseline/building_7_lstm.yaml" \
          --output_length=$i \
          --device=0 \
          --output_dir="results/multivariable/lstm/ENG_7"
  echo "Finished training model with output length = $i"
  echo "=================================================="
done
