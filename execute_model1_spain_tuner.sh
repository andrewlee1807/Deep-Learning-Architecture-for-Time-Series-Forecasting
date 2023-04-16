#
# Copyright (c) 2022-2023 Andrew
# Email: andrewlee1807@gmail.com
#

#python main.py --dataset_name="CNU" ----model_name="Model1" --history_len=24 --output_len=7 --num_features=1 --device=0

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
   echo "Starting to train model with output length = $i"
   python main_tuner.py --dataset_name="spain" --write_log_file=True --model_name="Model1" --config_path="config/spain_model_1.yaml" --output_length=$i --device=0 --output_dir="results/spain_vanilla_model_1_tuner"
   echo "Finished training model with output length = $i"
   echo "=================================================="
done