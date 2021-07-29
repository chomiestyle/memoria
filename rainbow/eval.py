
"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

"""
Model Evaluation
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""


import os
import coloredlogs

from docopt import docopt

from rainbow.trading_bot.agent import Agent
from rainbow.trading_bot.methods import evaluate_model
from rainbow.trading_bot.utils import (get_stock_data_1d,get_stock_data_1m,format_currency,format_position,show_eval_result,switch_k_backend_device)


def main(eval_stock, window_size, model_name, debug, manual_run,per_value,is_prediction):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """

    if type(eval_stock) == str:
        if per_value == 'minute':
            data = get_stock_data_1m(eval_stock)
        elif per_value == 'day':
            data = get_stock_data_1d(eval_stock)
    elif type(eval_stock) == list:
        data = eval_stock
    else:
        data = eval_stock

    #print(data)
    initial_offset = data[1] - data[0]

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained=True, model_name=model_name, manual = manual_run)
        profit, history = evaluate_model(agent, data, window_size, debug,is_prediction)
        #print('profit: {}'.format(profit))
        #print('history: {}'.format(history))
        show_eval_result(model_name, profit, initial_offset)
        return profit,history,agent
        
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained=True, model_name=model)
                profit,history = evaluate_model(agent, data, window_size, debug,is_prediction)
                show_eval_result(model, profit, initial_offset)
                del agent


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["<eval-stock>"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
    debug = args["--debug"]
    manual_run = True
    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(eval_stock, window_size, model_name, debug, manual_run)
    except KeyboardInterrupt:
        print("Aborted")
