#
# Copyright (c) 2022 Andrew
# Email: andrewlee1807@gmail.com
#

#python main.py --dataset_name="CNU" ----model_name="Model1" --history_len=24 --output_len=7 --num_features=1 --device=0

for i in 1 12 24 36 48 60 72 84
do
   echo "Welcome $i output_len"
   python main.py --dataset_name="household" --write_log_file=True --model_name="Model1" --output_len=$i --device=0
done