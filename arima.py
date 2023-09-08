from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import argparse
from utils.data import Dataset
from utils.logging import arg_parse, warming_up, close_logging
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
import time

t1 = time.time()


# # rolling forecasts
# for i in range(0, len(y)):
#     # predict
#     model = ARIMA(history, order=(1, 1, 0))
#     model_fit = model.fit()
#     yhat = model_fit.forecast()[0]
#     # invert transformed prediction
#     predictions.append(yhat)
#     # observation
#     obs = y[i]
#     history.append(obs)
#
# # report performance
# mse = mean_squared_error(y, predictions)
# print('MSE: ' + str(mse))
# mae = mean_absolute_error(y, predictions)
# print('MAE: ' + str(mae))

# Function to fit ARIMA model and make forecasts
def fit_arima(train_data, forest_length):
    # ARIMA order (adjust as needed)
    order = (1, 1, 0)
    model = ARIMA(train_data[:-forest_length], order=order)
    model_fit = model.fit()
    yhat = model_fit.forecast(forest_length)
    return yhat.tolist(), train_data[-forest_length:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = arg_parse(parser)
    config = warming_up(args)

    scaler_engine = MinMaxScaler()

    # Load dataset
    dataset = Dataset(dataset_name=config["dataset_name"])
    # data = dataset.dataloader.export_a_single_sequence()
    data = dataset.dataloader.export_the_sequence(config["features"])

    train_arima, test_arima = train_test_split(data.tolist(), train_size=config['train_ratio'], shuffle=False)

    train_arima_scale = scaler_engine.fit_transform(train_arima)
    test_arima_scale = scaler_engine.transform(test_arima)

    history = train_arima_scale[:, -1].tolist()
    y = test_arima_scale[:, -1].tolist()
    # make first prediction
    predictions = list()

    # Number of processes to use (adjust as needed)
    num_processes = multiprocessing.cpu_count()

    # Create a pool of worker processes
    pool = multiprocessing.Pool(num_processes)

    # Parallelize the forecasting process
    output_pair = pool.starmap(fit_arima, [(history + y[:i + config['output_length']], config['output_length']) for i in
                                           range(0, len(y) - config['output_length'])])

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print("Time : ", time.time() - t1)

    # # Extract predicted and actual values
    # predicted_values = [item[0][0] for item in output_pair]
    # actual_values = [item[1][0] for item in output_pair]
    # Calculate and print metrics (MSE and MAE)
    mse_values = []
    mae_values = []
    for predicted, actual in output_pair:
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        mse_values.append(mse)
        mae_values.append(mae)
    result = (sum(mse_values) / len(mse_values), sum(mae_values) / len(mae_values))
    print(f'MSE: {result[0]}')
    print(f'MAE: {result[1]}')

    import os

    result_file = f'{os.path.join(args.output_dir, args.dataset_name)}_evaluation_result.txt'
    file = open(result_file, 'a')
    file.write(f'{config["output_length"]},{result[0]},{result[1]}\n')
    file.close()

    if args.write_log_file:
        close_logging(config["file"], config["orig_stdout"])
    # mse = mean_squared_error(actual_values, predicted_values)
    # mae = mean_absolute_error(actual_values, predicted_values)
    # print(f'MSE: {mse}')
    # print(f'MAE: {mae}')
