# coding=utf-8
from common.base_model_reranking import BaseModelReranking
from developing_models.private_sphere.tuanyu.subscribe_rerank_actor_critic_v1 import V3 as V3Base
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
    def build_model_input(self):
        super(V3, self).build_model_input()

        if '1.4' in tf.__version__:
            feed_source = tf.sparse_tensor_to_dense(self.features["context__feed_source"], '')
            feed_type = tf.sparse_tensor_to_dense(self.features["context__real_feedtype"], '')
        else:
            feed_source = self.features["context__feed_source"]
            feed_type = self.features["context__real_feedtype"]
        feed_source = tf.squeeze(feed_source, axis=-1)  # [B,N]
        feed_type = tf.squeeze(feed_type, axis=-1)  # [B,N]
        self.top_contents_flag = tf.where(tf.equal(feed_source, 'topContents'), tf.ones_like(feed_source, tf.int32), tf.zeros_like(feed_source, tf.int32))  # [B,N]
        self.second_topping_flag = tf.where(tf.equal(feed_type, '900104'), tf.ones_like(feed_type, tf.int32), tf.zeros_like(feed_type, tf.int32))  # [B,N]
        self.top_second_topping_flag = self.top_contents_flag * self.second_topping_flag

    def get_sampling_function(self):
        self.inference_prediction_order = []
        self.inference_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        if self.data_type == "top_item":
            pos_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.max)  # [B,N]
            pos_second_topping = tf.ones([self.batch_size, self.N]) * 10000000  # [B,N]

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights * self.sample_polarization_degree
            origin_attention_weights = tf.nn.softmax(attention_weights)
            # attention_weights = tf.Print(attention_weights, [_,attention_weights[0]], message="inference attention_weights", first_n=1000000, summarize=1000000)
            if self.use_masking:
                if self.data_type == "top_item":
                    attention_weights = tf.where(self.top_contents_flag > 0, pos_inf, attention_weights)  # [B,N]
                    attention_weights = tf.where(self.top_second_topping_flag > 0, pos_second_topping, attention_weights)  # [B,N]
                attention_weights = tf.where(self.inference_sampled_symbol > 0, neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            with tf.name_scope('Inference_Step_{}_Max_Score'.format(_)):
                max_origin_attention_weight = tf.reduce_max(origin_attention_weights, axis=-1)  # [B]
                max_final_attention_weight = tf.reduce_max(attention_weights, axis=-1)  # [B]
                tf.summary.scalar('inference_step_{}_avg_max_origin_score'.format(_), tf.reduce_mean(max_origin_attention_weight))
                tf.summary.scalar('inference_step_{}_avg_max_final_score'.format(_), tf.reduce_mean(max_final_attention_weight))
                tf.summary.histogram('inference_step_{}_max_origin_score_histogram'.format(_), max_origin_attention_weight)
                tf.summary.histogram('inference_step_{}_max_final_score_histogram'.format(_), max_final_attention_weight)

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
            pos_second_topping = tf.ones([self.batch_size, self.N]) * 10000000  # [B,N]

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

    def build_pv_wise_evaluator_model(self, input):
        if "all_channels" in self.channels or "sum_pooling" in self.channels:
            # 1、sum_pooling_channel
            with tf.variable_scope(name_or_scope="{}_Sum_Pooling_Channel".format(self.name),
                                   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
                sum_pooling_layer = tf.reduce_sum(input, axis=1)
                logging.info("[DEBUG] sum_pooling_layer: {}".format(sum_pooling_layer))

        if "all_channels" in self.channels or "concatenation" in self.channels:
            # 2、concatenation_channel
            with tf.variable_scope(name_or_scope="{}_Concatenation_Channel".format(self.name),
                                   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE) as scope:
                running_layer = layers.fully_connected(
                    input,
                    16,
                    utils.getActivationFunctionOp(self.activation),
                    scope=scope,
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output],
                    normalizer_fn=layers.batch_norm if self.use_BN else None,
                    normalizer_params={"scale": True, "is_training": False})
                concatenation_layer = tf.reshape(running_layer, [-1, self.pv_size * running_layer.get_shape().as_list()[2]])
                logging.info("[DEBUG] concatenation_layer: {}".format(concatenation_layer))

        if "all_channels" in self.channels or "multi_head_self_attention" in self.channels:
            # 3、multi_head_self_attention_channel
            with tf.variable_scope(name_or_scope="{}_Multi_Head_Self_Attention_Channel".format(self.name),
                                   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
                shape_list = input.get_shape().as_list()
                all_feature_concatenation = tf.reshape(input, [-1, self.pv_size, shape_list[2]])
                queries = all_feature_concatenation
                keys = all_feature_concatenation
                mask = tf.cast(tf.ones_like(keys[:, :, 0]), dtype=tf.bool)
                logging.info("[DEBUG] mask:{}".format(mask))
                outputs, _ = multihead_attention(queries=queries,
                                                 keys=keys,
                                                 num_heads=8,
                                                 num_units=128,
                                                 num_output_units=2 * 128,
                                                 activation_fn="lrelu",
                                                 scope="multi_head_att",
                                                 atten_mode="ln",
                                                 reuse=tf.AUTO_REUSE,
                                                 key_masks=mask,
                                                 query_masks=mask,
                                                 variables_collections=[self.collections_dnn_hidden_layer],
                                                 outputs_collections=[self.collections_dnn_hidden_layer],
                                                 atten_weights_collections=[self.collections_dnn_hidden_layer],
                                                 is_target_attention=False)
                multi_head_self_attention_layer = tf.reduce_sum(outputs, axis=1)
                logging.info("[DEBUG] multi_head_self_attention_layer: {}".format(multi_head_self_attention_layer))

        if "all_channels" in self.channels or "rnn" in self.channels:
            # 4、rnn_channel
            def _rnn_channel(rnn_inputs):
                encoder_cell = tf.nn.rnn_cell.GRUCell(64)
                rnn_inputs = tf.transpose(rnn_inputs, perm=[1, 0, 2])  # [N,B,E]
                rnn_inputs = tf.unstack(rnn_inputs, num=self.pv_size, axis=0)  # [B,E]*N
                outputs, final_state = rnn.static_rnn(encoder_cell, rnn_inputs, dtype=tf.float32)

                output = [tf.reshape(output, [-1, 1, encoder_cell.output_size]) for output in outputs]
                output = tf.concat(axis=1, values=output)
                return output

            with tf.variable_scope(name_or_scope="{}_RNN_Channel".format(self.name),
                                   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                                   reuse=tf.AUTO_REUSE):
                # one can reverse self.all_feature_concatenation and make it a Bi-GRU
                rnn_layer_forward = _rnn_channel(input)
                rnn_layer = tf.reduce_sum(rnn_layer_forward, axis=1)
                logging.info("[DEBUG] rnn_layer: {}".format(rnn_layer))

        if "all_channels" in self.channels or "pair_wise_comparison" in self.channels:
            # 5、pair_wise_comparison_channel
            with tf.variable_scope(name_or_scope="{}_Pair_Wise_Comparison_Channel".format(self.name),
                                   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
                input_transposed = tf.transpose(input, perm=[0, 2, 1])
                output = tf.matmul(input, input_transposed)
                pair_wise_comparison_layer = tf.reshape(output, [-1, self.pv_size * self.pv_size])
                logging.info("[DEBUG] pair_wise_comparison_layer: {}".format(pair_wise_comparison_layer))

        # 6、dnn_layer
        dnn_layer = []
        for channel in self.channels:
            if channel == "all_channels":
                dnn_layer = [sum_pooling_layer, concatenation_layer, multi_head_self_attention_layer, rnn_layer, pair_wise_comparison_layer]
                break
            elif channel == "none_channel":
                dnn_layer = [tf.reshape(input, [-1, self.pv_size * input.get_shape().as_list()[2]])]
                break
            elif channel == "sum_pooling":
                dnn_layer.append(sum_pooling_layer)
            elif channel == "concatenation":
                dnn_layer.append(concatenation_layer)
            elif channel == "multi_head_self_attention":
                dnn_layer.append(multi_head_self_attention_layer)
            elif channel == "rnn":
                dnn_layer.append(rnn_layer)
            elif channel == "pair_wise_comparison":
                dnn_layer.append(pair_wise_comparison_layer)
        dnn_layer = tf.concat(values=dnn_layer, axis=-1)
        logging.info("[DEBUG] dnn_layer: {}".format(dnn_layer))
        final_neurons = self._dnn_layer(dnn_layer)

        # 7、resnet
        final_neurons = self.resnet_layer(final_neurons)

        # 8、logits_layer
        logits = self._logits_layer(final_neurons)

        # 9、prediction layer
        with tf.name_scope("{}_Predictions_Op".format(self.name)):
            predictions = tf.sigmoid(logits)
            predictions = tf.reshape(predictions, [-1, 1])  # [B,1]

        return predictions