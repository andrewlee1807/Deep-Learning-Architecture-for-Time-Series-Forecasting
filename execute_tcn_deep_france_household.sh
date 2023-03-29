#
# Copyright (c) 2022 Andrew
# Email: andrewlee1807@gmail.com
#

#python main.py --dataset_name="CNU" ----model_name="Model1" --history_len=24 --output_len=7 --num_features=1 --device=0

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
   echo "Starting to train model with output length = $i"
   python main.py --dataset_name="FRANCE_HOUSEHOLD_HOUR" --write_log_file=True --model_name="TCN" --config_path="config/tcn/france_household_deep_layer.yaml" --output_length=$i --device=0 --output_dir="results/tcn/france_household_deep"
   echo "Finished training model with output length = $i"
   echo "=================================================="
done