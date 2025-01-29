# coding=utf-8
from common.base_model_reranking import BaseModelReranking
from developing_models.private_sphere.tuanyu.subscribe_rerank_actor_critic_v1_second_topping import V3 as V3Base
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

class V3(V3Base):
    def evaluate(self):
        # input
        base_order_input, reranking_order_input, reranking_inference_order_input, reranking_training_order_sorting_input = self.build_evaluator_input()  # [B,N,input_size]

        # evaluate
        if self.evaluator_type == "item_wise":
            self.reranking_order_evaluator_score = self.build_item_wise_evaluator_model(reranking_order_input)  # [B,N]
            self.base_order_evaluator_score = self.build_item_wise_evaluator_model(base_order_input)  # [B,N]
            # self.reranking_order_evaluator_score = tf.Print(self.reranking_order_evaluator_score,
            # 												[self.reranking_order_evaluator_score,
            # 												 self.training_prediction_order],
            # 												message="reranking_order_evaluator_score", first_n=100,
            # 												summarize=10000)
            # self.base_order_evaluator_score = tf.Print(self.base_order_evaluator_score,
            # 										   [self.base_order_evaluator_score],
            # 										   message="base_order_evaluator_score", first_n=100, summarize=10000)
            if self.pv_value_evaluation == "upper_bound":
                self.reranking_order_pv_score = self.get_pv_value_upper_bound(self.reranking_order_evaluator_score)
                self.base_order_pv_score = self.get_pv_value_upper_bound(self.base_order_evaluator_score)
            elif self.pv_value_evaluation == "lower_bound":
                self.reranking_order_pv_score = self.get_pv_value_lower_bound(self.reranking_order_evaluator_score)
                self.base_order_pv_score = self.get_pv_value_lower_bound(self.base_order_evaluator_score)
        elif self.evaluator_type == "pv_wise":
            self.reranking_order_pv_score = self.reranking_order_evaluator_score = self.build_pv_wise_evaluator_model(reranking_order_input)  # [B,1]
            self.base_order_pv_score = self.base_order_evaluator_score = self.build_pv_wise_evaluator_model(base_order_input)  # [B,1]
            self.reranking_inference_order_pv_score = self.reranking_inference_order_evaluator_score = self.build_pv_wise_evaluator_model(reranking_inference_order_input)  # [B,1]
            self.reranking_training_order_sorting_pv_score = self.reranking_training_order_sorting_evaluator_score = self.build_pv_wise_evaluator_model(reranking_training_order_sorting_input)  # [B,1]

        baseline_reward_discount_ratio = self.baseline_reward_discount_ratio
        if self.use_dynamic_baseline_reward_discount_ratio:
            baseline_reward_discount_ratio = self.linear_increase()
        tf.summary.scalar("baseline_reward_discount_ratio", baseline_reward_discount_ratio)

        # better percentage
        self.better_difference = self.reranking_order_pv_score - self.base_order_pv_score  # [B,1]
        self.discount_better_difference = self.reranking_order_pv_score - baseline_reward_discount_ratio * self.base_order_pv_score	# [B,1]
        self.inference_order_better_difference = self.reranking_inference_order_pv_score - self.base_order_pv_score  # [B,1]
        self.training_order_sorting_better_difference = self.reranking_training_order_sorting_pv_score - self.base_order_pv_score  # [B,1]
        if "combined_model_and_prior_reward" in self.pv_value_evaluation:
            if self.pv_value_evaluation == "combined_model_and_prior_reward":
                prior_total_reward = self.calculate_list_prior_reward(self.training_prediction_order)  # [B,1]
            elif self.pv_value_evaluation == "combined_model_and_prior_reward_2":
                prior_total_reward = self.calculate_list_prior_reward_2(self.training_prediction_order)  # [B,1]
            self.better_difference = self.better_difference + prior_total_reward  # [B,1]
            self.metrics['scalar/combined_reward'] = tf.reduce_mean(self.better_difference)
        self.better_percentage = tf.where(self.better_difference > 0, tf.ones_like(self.better_difference), tf.identity(tf.zeros_like(self.better_difference)))  # [B,1]
        self.discount_better_percentage = tf.where(self.discount_better_difference > 0, tf.ones_like(self.discount_better_difference), tf.identity(tf.zeros_like(self.discount_better_difference)))  # [B,1]
        self.inference_order_better_percentage = tf.where(self.inference_order_better_difference > 0, tf.ones_like(self.inference_order_better_difference), tf.identity(tf.zeros_like(self.inference_order_better_difference)))  # [B,1]
        self.training_order_sorting_better_percentage = tf.where(self.training_order_sorting_better_difference > 0, tf.ones_like(self.training_order_sorting_better_difference), tf.identity(tf.zeros_like(self.training_order_sorting_better_difference)))  # [B,1]
        self.better_ratio = tf.divide(self.better_difference, self.reranking_order_pv_score + self.base_order_pv_score)  # [B,1]

        # reward
        self.build_evaluator_reward()

    def loss_op(self):
        self.evaluate()  # reward:[B,N]
        self.build_target_weights()  # [B,N]
        self.target_weights *= self.evaluator_reward  # [B,10]

        with tf.name_scope("{}_Loss_Op".format(self.name)):
            self.loss = 0.0

            self.reg_loss_f()

            # self.training_attention_distribution: [B,N] * (N+1)
            # labels: [B,N] * (N)
            logits = tf.stack(self.training_attention_distribution[:-1], axis=1)  # [B,10,20]
            labels = tf.one_hot(self.training_prediction_order, self.candidate_size)  # [B,10,20]

            # hill climbing
            if self.use_hill_climbing:
                reward_boolean_flag = self.get_max_reward_index()  # [B*hill_climbing_candidate_num]
                self.target_weights = tf.Print(self.target_weights, [tf.shape(self.target_weights), tf.shape(logits), tf.shape(labels), self.target_weights, logits, labels], message="before mask shape", first_n=10, summarize=10000)
                logits = tf.boolean_mask(logits, reward_boolean_flag)  # [B]
                labels = tf.boolean_mask(labels, reward_boolean_flag)  # [B]
                self.target_weights = tf.boolean_mask(self.target_weights, reward_boolean_flag)  # [B]
                self.target_weights = tf.Print(self.target_weights, [tf.shape(self.target_weights), tf.shape(logits), tf.shape(labels), self.target_weights, logits, labels], message="after mask shape", first_n=10, summarize=10000)

            if self.loss_type == "cross_entropy":
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # [B,10]
                if self.clip_loss == "tanh":
                    loss = tf.tanh(loss)
                self.loss += tf.reduce_sum(loss * self.target_weights, axis=-1)  # [B]
            elif self.loss_type == "inner_product":
                loss = -tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), labels), axis=-1)  # [B,10]
                self.loss += tf.reduce_sum(loss * self.target_weights, axis=-1)  # [B]
            elif self.loss_type == "probability":
                probabilities = tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), labels), axis=-1)  # [B,10]
                probabilities = probabilities * 20
                probabilities = tf.reduce_prod(probabilities, axis=-1)  # [B]
                self.loss += -probabilities * tf.reduce_mean(self.target_weights, axis=-1)  # [B]
            elif self.loss_type == "optimize_reward_directly":
                self.loss = 1 - self.reranking_training_order_sorting_evaluator_score  # [B,1]
            elif self.loss_type == "optimize_reward_l1":
                self.loss = tf.losses.absolute_difference(labels=tf.ones_like(self.reranking_training_order_sorting_evaluator_score), predictions=self.reranking_training_order_sorting_evaluator_score)    # [1]
            elif self.loss_type == "optimize_reward_l2":
                self.loss = tf.losses.mean_squared_error(labels=tf.ones_like(self.reranking_training_order_sorting_evaluator_score), predictions=self.reranking_training_order_sorting_evaluator_score) # [1]
            elif self.loss_type == "optimize_reward_cross_entropy":
                self.loss = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(self.reranking_training_order_sorting_evaluator_score), y_pred=self.reranking_training_order_sorting_evaluator_score)   # [B]
            self.loss = tf.reduce_mean(self.loss)
            tf.summary.scalar("main_loss", self.loss)

            # 叠加reinforce loss
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # [B,10]
            # if self.clip_loss == "tanh":
            #     loss = tf.tanh(loss)
            # loss = tf.reduce_sum(loss * self.target_weights, axis=-1)  # [B]
            # loss = tf.reduce_mean(loss)
            # self.loss += loss
            # tf.summary.scalar("reinforce_loss", loss)

            if "attention_weight_entropy_loss" in self.extra_loss:
                attention_weight = tf.nn.softmax(logits) + self.epsilon
                if self.extra_loss == "attention_weight_entropy_loss_1":
                    self.attention_weight_entropy_loss_weight /= 20.0
                    attention_weight_entropy_loss = tf.reduce_sum(tf.log(attention_weight), axis=-1)  # [B,10]
                elif self.extra_loss == "attention_weight_entropy_loss_2":
                    attention_weight_entropy_loss = -tf.reduce_sum(attention_weight * tf.log(attention_weight), axis=-1)  # [B,10]
                attention_weight_entropy_loss = -tf.reduce_mean(attention_weight_entropy_loss)
                tf.summary.scalar("origin_attention_weight_entropy_loss", attention_weight_entropy_loss)
                loss_weight_decay = 1.0
                global_step = tf.cast(self.global_step, tf.float32)
                self.max_global_step = float(self.max_global_step)
                min_value = 0.01
                if self.loss_weight_decay_method == "linear_decay":
                    self.max_global_step = float(1000000)
                    loss_weight_decay = self.get_global_step_linear_decay(global_step, self.max_global_step, min_value)
                elif self.loss_weight_decay_method == "exponential_decay":
                    decay_speed = 2.5  # 这决定了衰减的初始速率，值越大初始下降越快
                    linear_decay = self.get_global_step_linear_decay(global_step, self.max_global_step, 0.0)
                    loss_weight_decay = min_value + (1.0 - min_value) * tf.pow(linear_decay, decay_speed)
                elif self.loss_weight_decay_method == "sigmoid_decay":
                    scale = 15.0 / self.max_global_step  # scale决定了曲线的“宽度”
                    loss_weight_decay = min_value + (1.0 - min_value) / (1.0 + tf.exp((global_step - self.max_global_step / 2.0) * scale))
                elif self.loss_weight_decay_method == "cosine_decay":
                    cycles = 4  # 周期数
                    linear_decay = self.get_global_step_linear_decay(global_step, self.max_global_step, min_value)
                    loss_weight_decay = linear_decay * tf.maximum(tf.cos(global_step * 2 * np.pi * cycles / self.max_global_step), min_value)
                else:
                    self.attention_weight_entropy_loss_weight /= 3.0    # 恒定weight
                tf.summary.scalar("loss_weight_decay", loss_weight_decay)
                attention_weight_entropy_loss = self.attention_weight_entropy_loss_weight * loss_weight_decay * attention_weight_entropy_loss
                tf.summary.scalar("final_attention_weight_entropy_loss", attention_weight_entropy_loss)
                self.loss += attention_weight_entropy_loss

            is_adamw = (self.optimizer == "AdamAsync" and self.weight_decay != 0)
            if not is_adamw:
                print 'use reg_loss.'
                self.loss = self.loss + self.reg_loss

    def build_evaluator_input(self):
        position_feature = self.get_position_feature(self.pv_size)  # [B,N,1]

        exposure_input = self.get_candidates_non_sequential_feature_representation(top_n=self.pv_size)
        exposure_input = tf.concat([exposure_input, position_feature], axis=-1)
        base_order_input = exposure_input  # [B,pv_size,input_size]

        reranking_order_input = self.get_candidates_non_sequential_feature_representation(order=self.training_prediction_order)
        reranking_order_input = tf.concat([reranking_order_input, position_feature], axis=-1)

        reranking_inference_order_input = self.get_candidates_non_sequential_feature_representation(order=self.inference_prediction_order)
        reranking_inference_order_input = tf.concat([reranking_inference_order_input, position_feature], axis=-1)

        reranking_training_order_sorting_input = self.get_candidates_non_sequential_feature_representation(use_neural_sort=True)    # [B,N,D]
        reranking_training_order_sorting_input = tf.concat([reranking_training_order_sorting_input, position_feature], axis=-1)

        if len(set(self.fg.seq_feature_conf_map.keys())) > 0:
            seq_feature_representation = tf.tile(self.seq_feature_representation, [1, self.pv_size, 1])  # [B,pv_size,D]
            base_order_input = tf.concat([seq_feature_representation, exposure_input], axis=-1)
            reranking_order_input = tf.concat([seq_feature_representation, reranking_order_input], axis=-1)
            reranking_inference_order_input = tf.concat([seq_feature_representation, reranking_inference_order_input], axis=-1)
            reranking_training_order_sorting_input = tf.concat([seq_feature_representation, reranking_training_order_sorting_input], axis=-1)

        return base_order_input, reranking_order_input, reranking_inference_order_input, reranking_training_order_sorting_input

    def get_candidates_non_sequential_feature_representation(self, top_n=None, order=None, use_neural_sort=False):
        block_layer_dict = {}
        for block_name in self._column_blocks:
            block_layer_dict[block_name] = self.block_layer_dict[block_name]

            if block_name == 'context_columns':
                block_layer_dict[block_name] = self.block_layer_dict[block_name] + self.block_layer_dict['bias_columns']

            if top_n is not None:
                block_layer_dict[block_name] = block_layer_dict[block_name][:, :top_n, :]

            if order is not None:
                block_layer_dict[block_name] = self.sort_by_indices(indices=order, values=block_layer_dict[block_name])

            if use_neural_sort:
                attention_weight = tf.stack(self.training_attention_distribution[:-1], axis=1)  # [B,10,20]
                attention_weight = tf.nn.softmax(attention_weight)
                if self.neural_sort_version == "original_neural_sort":
                    # from developing_models.private_sphere.tuanyu.utils.neural_sort_for_reranking_v3 import SortPageByScore
                    from developing_models.private_sphere.tuanyu.utils.neural_sort_v2_for_reranking_v3 import SortPageByScore
                    spbs = SortPageByScore()
                    spbs.temperature_factor = self.temperature_factor
                    block_layer_dict[block_name] = spbs.neural_sort_page_by_score(block_layer_dict[block_name], attention_weight, self.training_prediction_order)   # [B,10,D]
                else:
                    from developing_models.private_sphere.tuanyu.utils.sort_by_score import SortPageByScore
                    spbs = SortPageByScore()
                    block_layer_dict[block_name] = spbs.sort_page_by_score(block_layer_dict[block_name], attention_weight, self.training_prediction_order)  # [B,10,D]
                # 这里没有做截断再算augment特征，是有问题的

        feature_representation = [
            self.augment_context_features(block_layer_dict[block_name]) if block_name in self.augmenting_blocks
            else block_layer_dict[block_name]
            for block_name in self.main_column_blocks]
        feature_representation = tf.concat(values=feature_representation, axis=-1)  # [B,N,input_size]
        feature_representation = tf.nn.tanh(feature_representation, name="FEATURE_TANH")

        return feature_representation

    def metrics_op(self):
        super(V3, self).metrics_op()

        self.metrics['scalar/training_order_sorting_better_percentage'] = tf.reduce_mean(self.training_order_sorting_better_percentage)  # [1]
        self.metrics['scalar/training_order_sorting_evaluator_reward'] = tf.reduce_mean(self.reranking_training_order_sorting_evaluator_score)  # [1]
        self.metrics['scalar/training_order_sorting_evaluator_reward_advantage'] = tf.reduce_mean(self.training_order_sorting_better_difference)

    def gumbel_sampling(self, sampling_shape):
        epsilon = 1e-10
        U = tf.random_uniform(sampling_shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + epsilon) + epsilon)

    def parse_hyper_params(self, hyper_params):
        super(V3, self).parse_hyper_params(hyper_params)

        algo_conf = get_config_value(hyper_params.algo_configs, self.name, hyper_params.algo_configs)

        model_conf = get_config_value(algo_conf, "modelx", {})
        self.neural_sort_version = model_conf.get("neural_sort_version", "original_neural_sort")
        self.temperature_factor = model_conf.get("temperature_factor", 0.03)
        self.sample_noise = model_conf.get("sample_noise", False)
        print "neural_sort_version = ", self.neural_sort_version
        print "temperature_factor = ", self.temperature_factor
        print "sample_noise = ", self.sample_noise