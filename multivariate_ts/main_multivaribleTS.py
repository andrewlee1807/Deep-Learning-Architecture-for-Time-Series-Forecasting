#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import sys
import argparse
from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
import yaml

from utils.directory import create_file

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
    parser.add_argument('--write_log_file', type=bool, default=False,
                        help='Export to log file')  # if --write_log_file added, then we will export to log file
    return parser.parse_args()


def initialize_logging(file_name, pred_length):
    orig_stdout = sys.stdout
    if pred_length == 1:  # firstly execute, so we need to create a new file
        file_name = create_file(f'{file_name}')
    f = open(file_name, 'a')
    sys.stdout = f
    # Because of using a file to log, so we need to print the time to know when the program is running
    import datetime
    current_time = datetime.datetime.now().time()
    print("Running time - ", current_time)
    return f, orig_stdout


def close_logging(f, orig_stdout):
    print("====================================================================================================\n\n\n")
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
    with open(args.config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config["output_length"] = args.output_length
    config["dataset_name"] = args.dataset_name
    config["tensorboard_log_dir"] = f'{args.output_dir}/tensorboard_log/{config["output_length"]}'
    print("Loaded configuration successfully ", args.config_path)

    print('Setting up output directory...')
    # initialize output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Output directory: ", args.output_dir)

    # initialize log file
    if args.write_log_file:
        print("Starting running background")
        print(
            f"To check output running, open file \"{os.path.join(args.output_dir, args.dataset_name)}_{config['output_length']} \"")
        file, orig_stdout = initialize_logging(f'{os.path.join(args.output_dir, args.dataset_name)}_training.log',
                                               config["output_length"])
        config["file"] = file
        config["orig_stdout"] = orig_stdout
    return config


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])
    # data = dataset.dataloader.export_a_single_sequence()
    data = dataset.dataloader.export_the_sequence(config["features"])

    # data = data[648:]

    # HOUSEHOLD dataset
    # dataloader = Dataset(dataset_name=args.dataset_name)
    # data = dataloader.dataloader.data_by_hour['Global_active_power']

    print("Building time series generator...")
    tsf = TimeSeriesGenerator(data=data,
                              config=config,
                              normalize_type=1,
                              shuffle=False)

    if "model" in config["dataset_name"]:  # delayNet model
        tsf.re_arrange_sequence(config)

    # tsf.normalize_data()

    print("Building model...")
    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      config=config)

    # callbacks
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"])

    # Train model
    history = model.fit(x=tsf.data_train[0],  # [number_recoder, input_len, number_feature]
                        y=tsf.data_train[1],  # [number_recoder, output_len, number_feature]
                        validation_data=tsf.data_valid,
                        epochs=100,
                        callbacks=[callbacks],
                        verbose=2,
                        batch_size=128,
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

    result_file = f'{os.path.join(args.output_dir, args.dataset_name)}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]} {result[1]} {result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
