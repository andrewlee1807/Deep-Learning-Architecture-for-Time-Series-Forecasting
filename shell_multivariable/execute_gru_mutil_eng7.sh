#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
  echo "Starting to train model with output length = $i"
  python main.py \
          --dataset_name="CNU_ENGINEERING_7" \
          --model_name="gru" \
          --config_path="multivariate_ts/config/baseline/building_7_gru.yaml" \
          --output_length=$i \
          --device=0 \
          --output_dir="results/multivariable/gru/ENG_7"
  echo "Finished training model with output length = $i"
  echo "=================================================="
done
