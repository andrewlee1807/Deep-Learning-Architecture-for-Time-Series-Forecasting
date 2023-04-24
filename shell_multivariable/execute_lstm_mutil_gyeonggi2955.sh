#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 1 24; do
  echo "Starting to train model with output length = $i"
  python main.py \
          --dataset_name="GYEONGGI2955" \
          --model_name="lstm" \
          --config_path="multivariate_ts/config/baseline/gyeonggi_2955_lstm.yaml" \
          --output_length=$i \
          --device=0 \
          --output_dir="results/multivariable/lstm/GYEONGGI2955"
  echo "Finished training model with output length = $i"
  echo "=================================================="
done
