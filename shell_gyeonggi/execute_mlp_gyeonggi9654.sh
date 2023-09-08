#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 36 48 60 72; do
  echo "Starting to train model with output length = $i"
  python main.py \
    --dataset_name="GYEONGGI9654" \
    --write_log_file=True \
    --model_name="MLP" \
    --config_path="benchmark/config/gy/gyeonggi_9654_mlp.yaml" \
    --output_length=$i \
    --device=0 \
    --output_dir="benchmark/exp/gy/mlp"

  echo "Finished training model with output length = $i"
  echo "=================================================="
done
