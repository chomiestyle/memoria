"""
Model Methods
Train and Evaluate Model Implementation.
Copyright (c) 2018 Prabhsimran Singh
Licensed under the MIT License (see LICENSE for details)
Written by Prabhsimran Singh
"""
import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data,ep_count, batch_size=32):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []
    reward_list = []
    initial_windows = data[0]
    block = initial_windows[:, 3]
    state = get_state(block)
    #print(state)
    reward_max = 0
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        #print('t={}'.format(t))
        reward = 0
        full_data_windows = data[t+1]
        block = full_data_windows[:, 3]
        #print(block)
        next_state = get_state(block)
        # select an action
        action = agent.act(state)
        #full_window=data[t+1]
        actual_price=block[4]
        #print('actual price: {}'.format(actual_price))
        # BUY
        if action == 1:
            agent.inventory.append(actual_price)

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            min_value=min(agent.inventory)
            min_index = agent.inventory.index(min_value)
            bought_price = agent.inventory.pop(min_index)

            delta = actual_price - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        reward_list.append(reward)
        td_error = agent.calculate_td_error(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done, td_error)

        if len(agent.buffer) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
        #print('rewards: {}'.format(reward))
        if reward > reward_max:
            reward_max = reward 
            agent.save()
    # if episode % 10 == 0:
    #     agent.save(episode)
 
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss))), reward_list


def evaluate_model(agent, data, window_size, debug,is_prediction=False):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    start=0
    if is_prediction:
        start=window_size
        data_length = data_length - (window_size)
    block = data[0]
    #block = initial_windows[:, 3]
    state = get_state(block)
    #state = get_state(data, start, window_size + 1)
    for t in range(data_length):
        reward = 0
        block = data[t+1]
        #block = full_data_windows[:, 3]
        next_state = get_state(block)
        #print('llega hasta aca')
        #next_state = get_state(data,start + t + 1, window_size + 1)
        #print('pasa next state')
        # select an action
        action = agent.act(state, is_eval=True)
        actual_price=block[4][0]
        #print('actual price: {}'.format(actual_price))
        # BUY
        if action == 1:
            agent.inventory.append(actual_price)
            history.append((actual_price, "BUY",0))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(actual_price)))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            min_value=min(agent.inventory)
            min_index = agent.inventory.index(min_value)
            bought_price = agent.inventory.pop(min_index)
            delta = actual_price - bought_price
            reward = delta #max(delta, 0)
            total_profit += delta

            history.append((actual_price, "SELL",delta))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(actual_price), format_position(actual_price - bought_price)))
        # HOLD
        else:
            history.append((actual_price, "HOLD",0))

        done = (t == data_length - 1)
        agent.n_step_buffer.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
