#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import sys
import argparse
from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
import yaml

# folder to load config file
CONFIG_PATH = "./"


# Get all arguments from command
def arg_parse(parser):
    parser.add_argument('--dataset_name', type=str, default='cnu', help='Dataset Name: household; cnu; spain; gyeonggi')
    parser.add_argument('--model_name', type=str, default='model1', help='Model Name: model1; model2; model3')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset path')
    parser.add_argument('--config_path', type=str, help='Configuration file path')
    parser.add_argument('--output_length', type=int, default=1, help='Prediction Length')
    # parser.add_argument('--num_features', type=int, default=1, help='Number of features')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--write_log_file', type=bool, default=False, help='Export to log file')
    return parser.parse_args()


def initialize_logging(file_name):
    orig_stdout = sys.stdout
    f = open(f'{file_name}.txt', 'w')
    sys.stdout = f
    return f, orig_stdout


def close_logging(f, orig_stdout):
    sys.stdout = orig_stdout
    f.close()


def warming_up(args):
    """
    Configuring environment
    """
    print('Setting up environment...')
    # setup CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print('Loading configuration file...')
    # read configuration
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    config["output_length"] = args.output_length
    print("Loaded configuration successfully ", args.config_path)

    print('Setting up output directory...')
    # initialize output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Output directory: ", args.output_dir)

    print("Starting running background")
    print(
        f"To check output running, open file {os.path.join(args.output_dir, args.dataset_name)}_{config['output_length']}")
    # initialize log file
    if args.write_log_file:
        file, orig_stdout = initialize_logging(f'{os.path.join(args.output_dir, args.dataset_name)}_'
                                               f'{config["output_length"]}')
    config["file"] = file
    config["orig_stdout"] = orig_stdout
    return config


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=args.dataset_name)
    data = dataset.dataloader.export_a_single_sequence()

    # HOUSEHOLD dataset
    # dataloader = Dataset(dataset_name=args.dataset_name)
    # data = dataloader.dataloader.data_by_hour['Global_active_power']

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config)

    tsf.re_arrange_sequence(config)

    tsf.normalize_data()

    print("Building model...")
    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      config=config)

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

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                            verbose=2,
                            use_multiprocessing=True)
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
