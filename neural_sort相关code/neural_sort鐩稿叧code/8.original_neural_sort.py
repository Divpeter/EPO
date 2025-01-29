import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import sys
from sort_by_score import print_value

def neural_sort(values, rank_score, temperature_factor):
    '''
    Please refer to the following paper for details.
    "Stochastic optimization of sorting networks via continuous relaxations"
    :param values: [B, N, E]
    :param rank_score: [B, N]
    :param temperature_factor: scalar
    :return:
    '''
    # pv_size = rank_score.get_shape().as_list()[1]   # [B,20]
    pv_size = tf.shape(rank_score)[1]   # [B,20]

    part1 = tf.cast(pv_size + 1 - 2 * tf.range(1, pv_size + 1), tf.float32) # [20]
    part1 = tf.expand_dims(part1, axis=-1) * tf.expand_dims(rank_score, axis=1) # [20,1] * [B,1,20] = [B,20,20]

    part2 = tf.abs(tf.expand_dims(rank_score, axis=-1) - tf.expand_dims(rank_score, axis=1))    # [B,20,1] - [B,1,20] = [B,20,20]
    part2 = tf.reduce_sum(part2, axis=1)    # [B,20]

    p_sort = part1 - tf.expand_dims(part2, axis=1)  # [B,20,20] - [B,1,20] = [B,20,20]
    p_sort = p_sort / temperature_factor    # [B,20,20]
    p_sort = tf.nn.softmax(p_sort)  # [B,20,20]

    sorted_values = tf.matmul(p_sort, values)   # [B,20,20] * [B,20,D] = [B,20,D]
    # sorted_values = tf.Print(sorted_values, [p_sort[0], values[0], sorted_values[0]], message="check detail", first_n=10000, summarize=10000000)

    return sorted_values


