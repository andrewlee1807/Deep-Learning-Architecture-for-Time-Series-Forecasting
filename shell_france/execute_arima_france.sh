#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 36 48 60 72; do
  echo "Starting to train model with output length = $i"
  python arima.py \
    --dataset_name="FRANCE_HOUSEHOLD_HOUR" \
    --write_log_file=True \
    --model_name="arima" \
    --config_path="benchmark/config/france/france_arima.yaml" \
    --output_length=$i \
    --device=0 \
    --output_dir="benchmark/exp/france/arima"
  echo "Finished training model with output length = $i"
  echo "=================================================="
done
