#  Copyright (c) 2022-2022 Andrew
#  Email: andrewlee1807@gmail.com
#
import os
import argparse
import numpy as np
from models import get_model, build_callbacks
from utils.data import Dataset, TimeSeriesGenerator
from utils.logging import arg_parse, warming_up, close_logging
from utils.directory import open_file_pkl


def main():
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)

    config = warming_up(args)

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])
    # Train, Valid, Test variables
    data_train = None
    data_valid = None
    data_test = None

    # check txt local files exist then load them and skip this steps
    if "MODEL" in args.model_name.upper() and os.path.exists(
            f'{config["output_dir"]}/{config["dataset_name"]}_data_train.pkl') and os.path.exists(
        f'{config["output_dir"]}/{config["dataset_name"]}_data_valid.pkl'):  # delayNet model
        print("Loading Train data...")
        # load data from local files with numpy
        data_train = open_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_train.pkl')
        print("Loading Valid data...")
        # load data from local files with numpy
        data_valid = open_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_valid.pkl')
        if os.path.exists(f'{config["output_dir"]}/{config["dataset_name"]}_data_test.pkl'):
            print("Loading Test data...")
            # load data from local files with numpy
            data_test = open_file_pkl(f'{config["output_dir"]}/{config["dataset_name"]}_data_test.pkl')
    else:
        # data = dataset.dataloader.export_a_single_sequence()
        data = dataset.dataloader.export_the_sequence(config["features"])

        # data = data[648:]

        print("Building time series generator...")
        tsf = TimeSeriesGenerator(data=data,
                                  config=config,
                                  normalize_type=1,
                                  shuffle=False)

        if "MODEL" in args.model_name.upper():  # delayNet model
            tsf.re_arrange_sequence(config)
        data_train = tsf.data_train
        data_valid = tsf.data_valid
        data_test = tsf.data_test

    print("Building model...")
    # Get model (built and summary)
    model = get_model(model_name=args.model_name,
                      config=config)

    # callbacks
    callbacks = build_callbacks(tensorboard_log_dir=config["tensorboard_log_dir"])

    # Train model
    history = model.fit(x=data_train[0],  # [number_recoder, input_len, number_feature]
                        y=data_train[1],  # [number_recoder, output_len, number_feature]
                        validation_data=data_valid,
                        epochs=config["epochs"],
                        callbacks=[callbacks],
                        verbose=2,
                        batch_size=64,
                        use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    result = model.evaluate(data_test[0], data_test[1], batch_size=1,
                            verbose=2,
                            use_multiprocessing=True)
    print("Evaluation result: ")
    print(config['output_length'], result[1], result[2])

    result_file = f'{os.path.join(config["output_dir"], config["dataset_name"])}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[1]},{result[2]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])


if __name__ == '__main__':
    main()
