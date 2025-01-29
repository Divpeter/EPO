# coding=utf-8
from common.base_model_reranking import BaseModelReranking
from developing_models.private_sphere.tuanyu.subscribe_rerank_actor_critic_v1_optimize_reward import V3 as V3Base
import tensorflow as tf
import numpy as np
import prada_model_ops.metrics as metrics
from common.model_utils import ops as base_ops
from common.model_utils import utils
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from common.model_utils import optimizer_ops as myopt
from common.model_utils.general_util import *
from common.model_utils.prada_util_attention import multihead_attention
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from developing_models.private_sphere.tuanyu.utils.original_neural_sort import neural_sort

class V3(V3Base):
    def get_training_sampling_function(self):
        self.training_prediction_order = []
        self.training_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        if self.data_type == "top_item":
            pos_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.max)  # [B,N]
            pos_second_topping = tf.ones([self.batch_size, self.N]) * 10000000  # [B,N]

        self.neural_sort_all_step_tensor = {}
        for block_name in self.main_column_blocks:
            self.neural_sort_all_step_tensor[block_name] = []

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights * self.sample_polarization_degree
            origin_attention_weights = tf.nn.softmax(attention_weights)
            if self.training_sample_manner == "random":
                attention_weights = tf.ones_like(attention_weights)
            # attention_weights = tf.Print(attention_weights, [_, attention_weights[0]], message="training attention_weights", first_n=1000000, summarize=1000000)
            if self.use_masking:
                if self.data_type == "top_item":
                    attention_weights = tf.where(self.top_contents_flag > 0, pos_inf, attention_weights)  # [B,N]
                    attention_weights = tf.where(self.top_second_topping_flag > 0, pos_second_topping, attention_weights)  # [B,N]
                attention_weights = tf.where(self.training_sampled_symbol > 0, neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.sample_noise:
                attention_weights = tf.log(attention_weights + 1e-10) + self.gumbel_sampling(sampling_shape=tf.shape(attention_weights))    # [B,M]

            with tf.name_scope('Training_Step_{}_Max_Score'.format(_)):
                max_origin_attention_weight = tf.reduce_max(origin_attention_weights, axis=-1)  # [B]
                max_final_attention_weight = tf.reduce_max(attention_weights, axis=-1)  # [B]
                tf.summary.scalar('training_step_{}_avg_max_origin_score'.format(_), tf.reduce_mean(max_origin_attention_weight))
                tf.summary.scalar('training_step_{}_avg_max_final_score'.format(_), tf.reduce_mean(max_final_attention_weight))
                tf.summary.histogram('training_step_{}_max_origin_score_histogram'.format(_), max_origin_attention_weight)
                tf.summary.histogram('training_step_{}_max_final_score_histogram'.format(_), max_final_attention_weight)

            if self.training_sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                greedy_result = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
            # sampling_symbol = tf.distributions.Categorical(probs=attention_weights).sample()  # [B,N] -> [B]
            # sampling_symbol = tf.Print(sampling_symbol, [sampling_symbol, greedy_result], message="check training sample", first_n=100, summarize=1000)
            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.training_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                sampling_symbol_onehot = tf.where(self.training_sampled_symbol > 0, tf.zeros_like(sampling_symbol_onehot), sampling_symbol_onehot)  # [B,N]
                # ***** #
                self.training_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            # sampling_symbol = tf.Print(sampling_symbol,
            #                          [_, sampling_symbol, sampling_symbol_score, self.training_sampled_symbol,
            #                           attention_weights],
            #                          message="check sampling_symbol/sampling_symbol_score/training_sampled_symbol/attention_weights",
            #                          first_n=100, summarize=1000)

            if self.decoder_input_from == "encoder_input":
                embedding_matrix = self.model_input  # [B,N,input_size]
            elif self.decoder_input_from == "encoder_output":
                embedding_matrix = self.encoder_states  # [B,N,state_size]
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            # if not update_embedding:
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)

            # 上面那个是输入给decoder的input embedding，下面这个是输入给evaluator的input embedding
            for block_name in self.main_column_blocks:
                block_tensor = self.block_layer_dict[block_name]    # [B,M,D]
                if block_name == 'context_columns':
                    block_tensor += self.block_layer_dict['bias_columns']   # [B,M,D]
                block_tensor = neural_sort(block_tensor, attention_weights, self.temperature_factor)    # [B,M,D]
                neural_sort_step_tensor = block_tensor[:, 0, :] # [B,D]
                self.neural_sort_all_step_tensor[block_name].append(neural_sort_step_tensor)

            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def build_evaluator_input(self):
        position_feature = self.get_position_feature(self.pv_size)  # [B,N,1]

        exposure_input = self.get_candidates_non_sequential_feature_representation(top_n=self.pv_size)
        exposure_input = tf.concat([exposure_input, position_feature], axis=-1)
        base_order_input = exposure_input  # [B,pv_size,input_size]

        reranking_order_input = self.get_candidates_non_sequential_feature_representation(order=self.training_prediction_order)
        reranking_order_input = tf.concat([reranking_order_input, position_feature], axis=-1)

        reranking_inference_order_input = self.get_candidates_non_sequential_feature_representation(order=self.inference_prediction_order)
        reranking_inference_order_input = tf.concat([reranking_inference_order_input, position_feature], axis=-1)

        reranking_training_order_sorting_input = []
        for block_name in self.main_column_blocks:
            self.neural_sort_all_step_tensor[block_name] = tf.stack(self.neural_sort_all_step_tensor[block_name], axis=1)   # [B,N,D]
            if block_name in self.augmenting_blocks:
                reranking_training_order_sorting_input.append(self.augment_context_features(self.neural_sort_all_step_tensor[block_name]))
            else:
                reranking_training_order_sorting_input.append(self.neural_sort_all_step_tensor[block_name])
        reranking_training_order_sorting_input = tf.concat(values=reranking_training_order_sorting_input, axis=-1)  # [B,N,input_size]
        reranking_training_order_sorting_input = tf.nn.tanh(reranking_training_order_sorting_input, name="FEATURE_TANH")
        reranking_training_order_sorting_input = tf.concat([reranking_training_order_sorting_input, position_feature], axis=-1)

        # reranking_training_order_sorting_input = tf.Print(reranking_training_order_sorting_input, [self.training_prediction_order[0], self.inference_prediction_order[0], reranking_order_input[0,:,:20], reranking_inference_order_input[0,:,:20], reranking_training_order_sorting_input[0,:,:20]], message="check result", first_n=100, summarize=1000000)

        if len(set(self.fg.seq_feature_conf_map.keys())) > 0:
            seq_feature_representation = tf.tile(self.seq_feature_representation, [1, self.pv_size, 1])  # [B,pv_size,D]
            base_order_input = tf.concat([seq_feature_representation, exposure_input], axis=-1)
            reranking_order_input = tf.concat([seq_feature_representation, reranking_order_input], axis=-1)
            reranking_inference_order_input = tf.concat([seq_feature_representation, reranking_inference_order_input], axis=-1)
            reranking_training_order_sorting_input = tf.concat([seq_feature_representation, reranking_training_order_sorting_input], axis=-1)

        return base_order_input, reranking_order_input, reranking_inference_order_input, reranking_training_order_sorting_input