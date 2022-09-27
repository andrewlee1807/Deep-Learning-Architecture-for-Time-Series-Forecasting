#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import argparse
from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator


# Get all arguments from command
def arg_parse(parser):
    parser.add_argument('--dataset_name', type=str, default='cnu', help='Dataset Name: household; cnu; spain')
    parser.add_argument('--model_name', type=str, default='model1', help='Model Name: model1; model2; model3')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset path')
    parser.add_argument('--history_len', type=int, default=24, help='History Length')
    parser.add_argument('--output_len', type=int, default=7, help='Prediction Length')
    parser.add_argument('--num_features', type=int, default=1, help='Number of features')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--write_log_file', type=bool, default=True, help='Export to log file')
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    # setup CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Load dataset
    cnu = Dataset(dataset_name="cnu")
    cnu = cnu.dataloader.raw_data
    tsf = TimeSeriesGenerator(data=cnu,
                              input_width=args.history_len,
                              output_width=args.output_len)
    tsf.normalize_data()

    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      input_width=args.history_len,
                      target_size=args.output_len,
                      num_features=args.num_features)

    # callbacks
    callbacks = build_callbacks()

    # Train model
    history = model.fit(x=tsf.data_train[0],
                        y=tsf.data_train[1],
                        validation_data=tsf.data_valid,
                        epochs=100,
                        callbacks=[callbacks],
                        verbose=2,
                        use_multiprocessing=True)


if __name__ == '__main__':
    main()
