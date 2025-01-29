# coding=utf-8
from common.base_model_reranking import BaseModelReranking
from developing_models.private_sphere.tuanyu.all_items_rerank_actor_critic_v1 import PointerNetwork as V3Base
import tensorflow as tf
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

class V3(V3Base):
    def set_labels(self):
        self.label = self.features['ctr_label']  # [B,N,1]
        if 'buy_label' in self.features:
            self.buy_label = self.features['buy_label']  # [B,N,1]
        if 'exposure_label' in self.features:
            self.exposure_label = self.features['exposure_label']   # [B,N,1]

    def mark_output(self, predictions):
        super(V3Base, self).mark_output(predictions)

    def collect_execution_ops(self):
      self.train_ops = {
            "run_ops":{
                "train_op": self.train_op
                , "localvar": self.local_auc_variables
                , "metrics": self.metrics
                , "label": self.training_prediction_order
                , "loss": self.loss
                , "global_step": self.global_step
            },
            "reset_auc_ops": self.reset_auc_ops,
            "auc": self.current_auc,
            "total_auc": self.total_auc,
            "loss": self.loss
        }

      self.predict_ops = {
          "run_ops": {
              "predictions": self.predictions,
              "ctr_label": self.features['ctr_label'],
              "id": self.id,
              "prediction_order": self.inference_prediction_order,
              "better_difference": self.inference_order_better_difference,  # [B,1]
              "better_percentage": self.inference_order_better_percentage,  # [B,1]
              "better_ratio": self.better_ratio,    # [B,1]
              "reranking_order_evaluator_score": self.reranking_order_evaluator_score,  # [B,1]
              "base_order_evaluator_score": self.base_order_evaluator_score,  # [B,1]
              "reranking_inference_order_evaluator_score": self.reranking_inference_order_evaluator_score   # [B,1]
          }
        }

    def get_candidates_non_sequential_feature_representation(self, top_n=None, order=None):
        block_layer_dict = {}
        for block_name in self._column_blocks:
            block_layer_dict[block_name] = self.block_layer_dict[block_name]

            if block_name == 'context_columns':
                block_layer_dict[block_name] = self.block_layer_dict[block_name] + self.block_layer_dict['bias_columns']

            if top_n is not None:
                block_layer_dict[block_name] = block_layer_dict[block_name][:, :top_n, :]

            if order is not None:
                block_layer_dict[block_name] = self.sort_by_indices(indices=order, values=block_layer_dict[block_name])

        feature_representation = [
            self.augment_context_features(block_layer_dict[block_name]) if block_name in self.augmenting_blocks
            else block_layer_dict[block_name]
            for block_name in self.main_column_blocks]
        feature_representation = tf.concat(values=feature_representation, axis=-1)  # [B,N,input_size]
        feature_representation = tf.nn.tanh(feature_representation, name="FEATURE_TANH")

        return feature_representation

    def build_model_input(self):
        super(V3, self).build_model_input()

        if '1.4' in tf.__version__:
            feed_source = tf.sparse_tensor_to_dense(self.features["context__feed_source"], '')
        else:
            feed_source = self.features["context__feed_source"]
        feed_source = tf.squeeze(feed_source, axis=-1)  # [B,N]
        self.top_contents_flag = tf.where(tf.equal(feed_source, 'topContents'), tf.ones_like(feed_source, tf.int32), tf.zeros_like(feed_source, tf.int32))  # [B,N]

    def get_sampling_function(self):
        self.inference_prediction_order = []
        self.inference_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        if self.data_type == "top_item":
            pos_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.max)  # [B,N]

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights * self.sample_polarization_degree
            # attention_weights = tf.Print(attention_weights, [_,attention_weights[0]], message="inference attention_weights", first_n=1000000, summarize=1000000)
            if self.use_masking:
                if self.data_type == "top_item":
                    attention_weights = tf.where(self.top_contents_flag > 0, pos_inf, attention_weights)  # [B,N]
                attention_weights = tf.where(self.inference_sampled_symbol > 0, neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                greedy_result = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
            # sampling_symbol = tf.distributions.Categorical(probs=attention_weights).sample()  # [B,N] -> [B]
            # sampling_symbol = tf.Print(sampling_symbol, [sampling_symbol, greedy_result], message="check inference sample", first_n=100, summarize=1000)
            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.inference_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                sampling_symbol_onehot = tf.where(self.inference_sampled_symbol > 0, tf.zeros_like(sampling_symbol_onehot), sampling_symbol_onehot)   # [B,N]
                # ***** #
                self.inference_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            # sampling_symbol = tf.Print(sampling_symbol,
            # 						   [_, sampling_symbol[0], sampling_symbol_score[0], self.inference_sampled_symbol[0],
            # 							attention_weights[0]],
            # 						   message="check sampling_symbol/sampling_symbol_score/inference_sampled_symbol/attention_weights",
            # 						   first_n=10000000, summarize=1000000000)

            if self.decoder_input_from == "encoder_input":
                embedding_matrix = self.model_input  # [B,N,input_size]
            elif self.decoder_input_from == "encoder_output":
                embedding_matrix = self.encoder_states  # [B,N,state_size]
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            # if not update_embedding:
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def get_training_sampling_function(self):
        self.training_prediction_order = []
        self.training_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        if self.data_type == "top_item":
            pos_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.max)  # [B,N]

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights * self.sample_polarization_degree
            # attention_weights = tf.Print(attention_weights, [_, attention_weights[0]], message="training attention_weights", first_n=1000000, summarize=1000000)
            if self.use_masking:
                if self.data_type == "top_item":
                    attention_weights = tf.where(self.top_contents_flag > 0, pos_inf, attention_weights)  # [B,N]
                attention_weights = tf.where(self.training_sampled_symbol > 0, neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.training_sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                greedy_result = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
            # sampling_symbol = tf.distributions.Categorical(probs=attention_weights).sample()  # [B,N] -> [B]
            # sampling_symbol = tf.Print(sampling_symbol, [sampling_symbol, greedy_result], message="check training sample",
            # 						   first_n=100, summarize=1000)
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
            # 						   [_, sampling_symbol, sampling_symbol_score, self.training_sampled_symbol,
            # 							attention_weights],
            # 						   message="check sampling_symbol/sampling_symbol_score/training_sampled_symbol/attention_weights",
            # 						   first_n=100, summarize=1000)

            if self.decoder_input_from == "encoder_input":
                embedding_matrix = self.model_input  # [B,N,input_size]
            elif self.decoder_input_from == "encoder_output":
                embedding_matrix = self.encoder_states  # [B,N,state_size]
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            # if not update_embedding:
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def build_evaluator_reward(self):
        super(V3, self).build_evaluator_reward()

        if self.reward_method == "click_&_cold_start":
            # click reward
            click_reward = self.better_difference
            click_reward = tf.tile(click_reward, [1, self.pv_size])  # [B,N]

            cold_start_feed_flag = self.features['context__rerank_reserve_4']    # [B,N,1]
            cold_start_pv_flag = tf.reduce_max(cold_start_feed_flag, axis=[1,2])  # [B]

            batch_cold_start_index = tf.range(1, self.pv_size + 1, dtype=tf.float32)
            batch_cold_start_index = tf.reshape(batch_cold_start_index, [-1, self.pv_size])
            batch_cold_start_index = tf.tile(batch_cold_start_index, [self.batch_size, 1])  # [B,pv_size]

            reranking_order_cold_start_flag = tf.squeeze(self.sort_by_indices(indices=self.training_prediction_order, values=cold_start_feed_flag), axis=-1)  # [B,pv_size]
            batch_cold_start_num = tf.reduce_sum(reranking_order_cold_start_flag)
            batch_cold_start_ratio = batch_cold_start_num / (tf.count_nonzero(cold_start_pv_flag, dtype=tf.float32)*self.pv_size + self.epsilon) # [1]

            reranking_order_cold_start_index = reranking_order_cold_start_flag * batch_cold_start_index  # [B,pv_size]
            batch_cold_start_avg_pos = tf.reduce_sum(reranking_order_cold_start_index) / (tf.count_nonzero(reranking_order_cold_start_index, dtype=tf.float32) + self.epsilon) # [1]

            cold_start_negative_reward_gate = tf.squeeze(tf.logical_or(batch_cold_start_ratio<self.cold_start_ratio_threshold,batch_cold_start_avg_pos>self.base_cold_start_avg_pos)) # scalar

            pv_cold_start_ratio = tf.reduce_mean(reranking_order_cold_start_flag, axis=-1)  # [B]
            pv_cold_start_remains = tf.reduce_sum(cold_start_feed_flag, axis=[1,2]) - tf.reduce_sum(reranking_order_cold_start_flag, axis=-1)   # [B]
            pv_cold_start_ratio_punishment_flag = tf.logical_and(pv_cold_start_ratio<self.cold_start_ratio_threshold, pv_cold_start_remains>0)    # [B]
            pv_cold_start_avg_pos = tf.reduce_sum(reranking_order_cold_start_index, axis=-1) / (tf.count_nonzero(reranking_order_cold_start_index, axis=-1, dtype=tf.float32) + self.epsilon) # [B]
            pv_cold_start_avg_pos_punishment_flag = tf.logical_and(tf.equal(cold_start_pv_flag,1), pv_cold_start_avg_pos>self.base_cold_start_avg_pos)   # [B]

            pv_cold_start_punishment_flag = tf.logical_or(pv_cold_start_ratio_punishment_flag, pv_cold_start_avg_pos_punishment_flag)   # [B]

            real_pv_cold_start_punishment_flag = tf.logical_and(cold_start_negative_reward_gate, pv_cold_start_punishment_flag)   # [B]
            real_pv_cold_start_punishment_flag = tf.reshape(real_pv_cold_start_punishment_flag, [-1,1])    # [B,1]

            cold_start_punishment_reward = tf.ones_like(click_reward) * self.cold_start_punishment_reward_value * self.bool_2_float(real_pv_cold_start_punishment_flag) # [B,N]

            self.evaluator_reward = click_reward + cold_start_punishment_reward # [B,N]

            # summary
            self.metrics['scalar/batch_cold_start_ratio'] = batch_cold_start_ratio
            self.metrics['scalar/batch_cold_start_avg_pos'] = batch_cold_start_avg_pos
            self.metrics['scalar/cold_start_negative_reward_gate'] = self.bool_2_float(cold_start_negative_reward_gate)
            self.metrics['scalar/pv_cold_start_ratio_punishment_flag'] = tf.reduce_mean(self.bool_2_float(pv_cold_start_ratio_punishment_flag))
            self.metrics['scalar/pv_cold_start_avg_pos_punishment_flag'] = tf.reduce_mean(self.bool_2_float(pv_cold_start_avg_pos_punishment_flag))
            self.metrics['scalar/pv_cold_start_punishment_flag'] = tf.reduce_mean(self.bool_2_float(pv_cold_start_punishment_flag))
            self.metrics['scalar/real_pv_cold_start_punishment_flag'] = tf.reduce_mean(self.bool_2_float(real_pv_cold_start_punishment_flag))
            self.metrics['scalar/cold_start_punishment_reward'] = tf.reduce_mean(cold_start_punishment_reward)
            self.metrics['scalar/total_reward'] = tf.reduce_mean(self.evaluator_reward)

    def bool_2_float(self, bool):
        return tf.cast(bool, dtype=tf.float32)

    def build_target_weights(self):
        super(V3, self).build_target_weights()

        if self.target_weights_manner == "top_content_mask":
            top_contents_flag = tf.expand_dims(self.top_contents_flag, axis=-1) # [B,candidate_size,1]
            reranking_order_top_contents_flag = tf.squeeze(self.sort_by_indices(indices=self.training_prediction_order, values=top_contents_flag), axis=-1) # [B,pv_size]
            self.target_weights = tf.cast(1-reranking_order_top_contents_flag, tf.float32)   # [B,pv_size]
            self.target_weights = tf.Print(self.target_weights, [reranking_order_top_contents_flag, self.target_weights], message="check top content loss mask", first_n=100, summarize=10000)

    def parse_hyper_params(self, hyper_params):
        super(V3, self).parse_hyper_params(hyper_params)

        algo_conf = get_config_value(hyper_params.algo_configs, self.name, hyper_params.algo_configs)

        scene_specific = get_config_value(algo_conf, "scene_specific", {})
        self.data_type = scene_specific.get("data_type", "default")
        self.cold_start_ratio_threshold = scene_specific.get("cold_start_ratio_threshold", 0.13)
        self.base_cold_start_avg_pos = scene_specific.get("base_cold_start_avg_pos", 6)
        self.cold_start_punishment_reward_value = scene_specific.get("cold_start_punishment_reward_value", -0.03)
        print "data_type = ", self.data_type
        print "cold_start_ratio_threshold = ", self.cold_start_ratio_threshold
        print "base_cold_start_avg_pos = ", self.base_cold_start_avg_pos
        print "cold_start_punishment_reward_value = ", self.cold_start_punishment_reward_value