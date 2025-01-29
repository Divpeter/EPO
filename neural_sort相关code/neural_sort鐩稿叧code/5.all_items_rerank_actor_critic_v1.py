# -*- coding: utf-8 -*-
import tensorflow as tf
if '1.4' in tf.__version__:
	from common.base_model_reranking import BaseModelReranking
else:
	from common.base_model_reranking_new import BaseModelReranking
from common.model_utils import ops as base_ops, checkpoint_utils
from common.model_utils import utils
from tensorflow.contrib import layers
from common.base_model import BaseModel
from common.model_utils.tflog import tflogger as logging
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.ops import nn_ops
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from common.model_utils import optimizer_ops as myopt
import prada_model_ops.metrics as metrics
from common.model_zoo.din import BaseDIN
import numpy as np
from common.model_utils.general_util import *
from common.model_utils.prada_util_attention import multihead_attention
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
from developing_models.private_sphere.tuanyu.utils.LNGRU import LNGRU
from developing_models.private_sphere.tuanyu.utils.LNLSTM import LayerNormBasicLSTMCell as LNLSTM
from developing_models.private_sphere.tuanyu.utils.pointer_network_decoder_v2 import attention_based_decoder
from developing_models.private_sphere.tuanyu.all_items_rerank_evaluator_v1 import Simulator
from common.model_utils.transfomer import transformer_model

# 1、label给法 2、预测采样不用argmax 3、训练不用teacher forcing 4、attention去掉已经挑选 5、每一步的label对不对 6、encoder state attention
class PointerNetwork(BaseModelReranking):
	def __init__(self, hyper_params, name="CTR"):
		super(PointerNetwork, self).__init__(hyper_params, name)

		self.expansion = 1000000000

		self.encoder_inputs = []
		self.decoder_inputs = []
		self.decoder_targets = []
		self.target_weights = []

	def parse_hyper_params(self, hyper_params):
		super(PointerNetwork, self).parse_hyper_params(hyper_params)

		algo_conf = get_config_value(hyper_params.algo_configs, self.name, hyper_params.algo_configs)

		rnn_config = get_config_value(algo_conf, "rnn_config", {})
		self.rnn_cell_type = rnn_config.get("rnn_cell_type", "custom_gru")
		self.state_size = rnn_config.get("state_size", 128)
		self.rnn_cell_layer = rnn_config.get("rnn_cell_layer", 1)
		self.attention_head_nums = rnn_config.get("attention_head_nums", 1)
		self.training_sample_manner = rnn_config.get("training_sample_manner", "sample")
		self.sample_manner = rnn_config.get("sample_manner", "greedy")
		self.sample_polarization_degree = rnn_config.get("sample_polarization_degree", 1)
		self.teacher_forcing = rnn_config.get("teacher_forcing", True)
		self.label_type = rnn_config.get("label_type", "click")
		self.ctrscore_feature = rnn_config.get("ctrscore_feature", "S_vv")
		self.decoder_input_from = rnn_config.get("decoder_input_from", "encoder_input")
		self.target_weights_manner = rnn_config.get("target_weights_manner", "equal")
		self.use_masking = rnn_config.get("use_masking", False)
		self.use_bidirectional_encoder = rnn_config.get("use_bidirectional_encoder", False)
		self.feed_context_vector = rnn_config.get("feed_context_vector", True)
		self.rnn_input_reverse = rnn_config.get("rnn_input_reverse", False)
		self.update_embedding = rnn_config.get("update_embedding", False)
		print "rnn_cell_type = ", self.rnn_cell_type
		print 'state_size = ', self.state_size
		print 'rnn_cell_layer = ', self.rnn_cell_layer
		print 'attention_head_nums = ', self.attention_head_nums
		print 'training_sample_manner = ', self.training_sample_manner
		print 'sample_manner = ', self.sample_manner
		print 'sample_polarization_degree = ', self.sample_polarization_degree
		print 'teacher_forcing = ', self.teacher_forcing
		print 'label_type = ', self.label_type
		print 'ctrscore_feature = ', self.ctrscore_feature
		print 'decoder_input_from = ', self.decoder_input_from
		print 'target_weights_manner = ', self.target_weights_manner
		print 'use_masking = ', self.use_masking
		print 'use_bidirectional_encoder = ', self.use_bidirectional_encoder
		print 'feed_context_vector = ', self.feed_context_vector
		print 'rnn_input_reverse = ', self.rnn_input_reverse
		print 'update_embedding = ', self.update_embedding

		rl_config = get_config_value(algo_conf, "rl_config", {})
		self.reward_method = rl_config.get("reward_method", "reinforce")
		self.pv_value_evaluation = rl_config.get("pv_value_evaluation", "upper_bound")
		self.reward_normalization = rl_config.get("reward_normalization", False)
		self.evaluator_type = rl_config.get("evaluator_type", "item_wise")
		self.evaluator_version = rl_config.get("evaluator_version", "default")
		self.use_hill_climbing = rl_config.get("use_hill_climbing", False)
		self.hill_climbing_candidate_num = rl_config.get("hill_climbing_candidate_num", 10)
		self.baseline_reward_discount_ratio = rl_config.get("baseline_reward_discount_ratio", 1)
		self.use_dynamic_baseline_reward_discount_ratio = rl_config.get("use_dynamic_baseline_reward_discount_ratio", False)
		print "reward_method = ", self.reward_method
		print "pv_value_evaluation = ", self.pv_value_evaluation
		print "reward_normalization = ", self.reward_normalization
		print "evaluator_type = ", self.evaluator_type
		print "evaluator_version = ", self.evaluator_version
		print "use_hill_climbing = ", self.use_hill_climbing
		print "hill_climbing_candidate_num = ", self.hill_climbing_candidate_num
		print "baseline_reward_discount_ratio = ", self.baseline_reward_discount_ratio
		print "use_dynamic_baseline_reward_discount_ratio = ", self.use_dynamic_baseline_reward_discount_ratio

		model_conf = get_config_value(algo_conf, "modelx", {})
		self.clip_loss = model_conf.get("clip_loss", "")
		self.loss_type = model_conf.get("loss_type", "cross_entropy")
		self.encoder_type = model_conf.get("encoder_type", "rnn")
		self.second_task_weight = model_conf.get("second_task_weight", 1)
		self.extra_loss = model_conf.get("extra_loss", "")
		self.loss_weight_decay_method = model_conf.get("loss_weight_decay_method", "")
		self.attention_weight_entropy_loss_weight = model_conf.get("attention_weight_entropy_loss_weight", 0.02)
		self.max_global_step = model_conf.get("max_global_step", 1200000)
		print "clip_loss = ", self.clip_loss
		print "loss_type = ", self.loss_type
		print "encoder_type = ", self.encoder_type
		print "second_task_weight = ", self.second_task_weight
		print "extra_loss = ", self.extra_loss
		print "loss_weight_decay_method = ", self.loss_weight_decay_method
		print "attention_weight_entropy_loss_weight = ", self.attention_weight_entropy_loss_weight
		print "max_global_step = ", self.max_global_step

	def build_model(self):
		if self.restore_embedding:
			self.feature_columns = self.setup_feature_columns()
		self.embedding_layer()
		if '1.4' in tf.__version__:
			self.reshape_input()	# self.features[feature_name]=[B,N,1], self.block_layer_dict[block_name]=[B,N,feature_num]
		else:
			self.cutting_input()
		self.build_model_input()

		self.init_rnn()
		if "rnn" in self.encoder_type:
			self.rnn_encode()
		elif self.encoder_type == "multi_head_attention":
			self.multi_head_attention_encode()
		elif self.encoder_type == "transformer":
			self.transformer_encode()
		elif "deep_set" in self.encoder_type:
			self.deep_set_encode()
		self.rnn_decode()

	def deep_set_encode(self):
		with tf.variable_scope("encoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			self.encoder_states = layers.fully_connected(
				self.model_input,
				self.state_size,
				utils.getActivationFunctionOp(self.activation),
				# scope=dnn_hidden_layer_scope,
				variables_collections=[self.collections_dnn_hidden_layer],
				outputs_collections=[self.collections_dnn_hidden_output]
			)
			self.final_state = tf.reduce_sum(self.encoder_states, axis=1)

			deep_set_function_name = self.encoder_type.split("_")[-1]
			print "deep_set_function_name = ", deep_set_function_name
			if deep_set_function_name != "dnn":
				self.final_state = utils.getActivationFunctionOp(deep_set_function_name)(self.final_state)
			else:
				self.final_state = layers.fully_connected(
					self.final_state,
					self.state_size,
					utils.getActivationFunctionOp(self.activation),
					# scope=dnn_hidden_layer_scope,
					variables_collections=[self.collections_dnn_hidden_layer],
					outputs_collections=[self.collections_dnn_hidden_output]
				)
			logging.info("[DEBUG] final_state: {}".format(self.final_state))

	def transformer_encode(self):
		# with tf.variable_scope("{}_self_attention_net".format(self.name), reuse=tf.AUTO_REUSE):
		with tf.variable_scope("encoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			# self_att_mask = tf.matmul(tf.expand_dims(sequence_mask, axis=2),
			# 						  tf.expand_dims(sequence_mask, axis=1))
			# logging.info("DEBUG: self_att_mask:{}".format(self_att_mask))
			keys = self.model_input
			if keys.get_shape().as_list()[2] != self.state_size:
				keys = layers.fully_connected(
					keys,
					self.state_size,
					utils.getActivationFunctionOp(self.activation),
					# scope=dnn_hidden_layer_scope,
					variables_collections=[self.collections_dnn_hidden_layer],
					outputs_collections=[self.collections_dnn_hidden_output]
				)
			self.encoder_states = transformer_model(input_tensor=keys,
										 # attention_mask=self_att_mask,
										 # hidden_size=keys.get_shape().as_list()[2],
										 hidden_size=self.state_size,
										 num_hidden_layers=2,
										 # num_attention_heads=8,
										 num_attention_heads=8 if self.state_size != 363 else 3,
										 intermediate_size=512,
										 attention_probs_dropout_prob=0.0,
										 hidden_dropout_prob=0.0)
			self.final_state = tf.reduce_sum(self.encoder_states, axis=1)
			logging.info("[DEBUG] final_state: {}".format(self.final_state))

	def multi_head_attention_encode(self):
		# with tf.variable_scope(name_or_scope="{}_Multi_Head_Self_Attention_Channel".format(self.name),
		# 					   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
		with tf.variable_scope("encoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			queries = self.model_input
			keys = self.model_input
			mask = tf.cast(tf.ones_like(keys[:, :, 0]), dtype=tf.bool)
			logging.info("[DEBUG] mask:{}".format(mask))
			self.encoder_states, _ = multihead_attention(queries=queries,
								keys=keys,
								num_heads=8,
								num_units=128,
								num_output_units=self.state_size,
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
			self.final_state = tf.reduce_sum(self.encoder_states, axis=1)
			logging.info("[DEBUG] final_state: {}".format(self.final_state))

	def get_a_cell(self, state_size):
		if self.rnn_cell_type == "official_gru":
			cell = tf.nn.rnn_cell.GRUCell(state_size)
		elif self.rnn_cell_type == "custom_gru":
			cell = LNGRU(state_size)
		elif self.rnn_cell_type == "official_lstm":
			cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=False)
		elif self.rnn_cell_type == "custom_lstm":
			cell = LNLSTM(state_size)

		return cell

	def init_rnn(self):
		self.encoder_cell = self.get_a_cell(self.state_size)
		self.decoder_cell = self.get_a_cell(self.state_size)
		# self.state = self.encoder_cell.zero_state(self.batch_size, dtype=tf.float32)

		if self.rnn_cell_layer > 1:
			self.encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.state_size) for _ in range(self.rnn_cell_layer)])
			self.decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell(self.state_size) for _ in range(self.rnn_cell_layer)])
		if self.use_bidirectional_encoder:
			self.encoder_cell = self.get_a_cell(self.state_size)
			self.encoder_backward_cell = self.get_a_cell(self.state_size)
			self.decoder_cell = self.get_a_cell(self.state_size*2)

	def rnn_encode(self):
		with tf.variable_scope("encoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			if self.encoder_type == "rnn":
				# build encoder input
				self.encoder_inputs = tf.transpose(self.model_input, perm=[1, 0, 2])  # [N,B,input_size]
				self.encoder_inputs = tf.unstack(self.encoder_inputs, num=self.candidate_size, axis=0)  # [[B,input_size]*N]
				if self.rnn_input_reverse:
					self.encoder_inputs = self.encoder_inputs[::-1]

				if not self.use_bidirectional_encoder:
					# [[B,state_size]*N], [B,state_size]
					# self.encoder_outputs, self.final_state = tf.contrib.rnn.static_rnn(self.encoder_cell, self.encoder_inputs, dtype=tf.float32)
					self.encoder_outputs, self.final_state = rnn.static_rnn(self.encoder_cell, self.encoder_inputs, dtype=tf.float32)
				else:
					# [[B,state_size*2]*N], [B,state_size*2], [B,state_size*2]
					self.encoder_outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(self.encoder_cell, self.encoder_backward_cell, self.encoder_inputs, dtype=tf.float32)
					self.final_state = tf.concat([output_state_fw, output_state_bw], axis=-1)	# [B,state_size*2]

				if not self.use_bidirectional_encoder:
					encoder_outputs = [tf.reshape(output, [-1, 1, self.encoder_cell.output_size]) for output in self.encoder_outputs]
				else:
					encoder_outputs = [tf.reshape(output, [-1, 1, self.encoder_cell.output_size * 2]) for output in self.encoder_outputs]
				# [B,N,state_size] or [B,N+1,state_size]
				self.encoder_states = tf.concat(axis=1, values=encoder_outputs)
			elif self.encoder_type == "dynamic_rnn":
				self.encoder_states, self.final_state = tf.nn.dynamic_rnn(cell=self.encoder_cell, dtype=tf.float32,
															 sequence_length=tf.ones([self.batch_size],dtype=tf.int32)*self.N, inputs=self.model_input)
			print "encoder_states = ", self.encoder_states
			print "final_state = ", self.final_state

	def rnn_decode(self):
		# build decoder input
		# training
		self.decoder_inputs = self.build_decoder_input()

		# build sampling function
		training_sampling_function = self.get_training_sampling_function()
		sampling_function = self.get_sampling_function()

		with tf.variable_scope("decoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			training_attention_distribution, states, _ = attention_based_decoder(
				self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
				sampling_function=training_sampling_function, attention_head_nums=self.attention_head_nums,
				feed_context_vector=self.feed_context_vector)

		with tf.variable_scope("decoder",
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=True):
			inference_attention_distribution, _, prediction_score = attention_based_decoder(
				self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
				sampling_function=sampling_function, attention_head_nums=self.attention_head_nums,
				feed_context_vector=self.feed_context_vector)

		self.training_attention_distribution = training_attention_distribution
		self.training_prediction_order = tf.stack(self.training_prediction_order, axis=1)  # [B,N]
		self.inference_attention_distribution = inference_attention_distribution
		self.inference_prediction_order = tf.stack(self.inference_prediction_order, axis=1)  # [B,N]
		self.predictions = prediction_score	# [B,N]

	def build_decoder_input(self):
		if self.decoder_input_from == "encoder_input":
			# add <BOS>
			decoder_inputs = [tf.zeros([self.batch_size, self.input_size])] * (self.pv_size + 1)  # [[B,input_size]*(N+1)]
		elif self.decoder_input_from == "encoder_output":
			# add <BOS>
			if not self.use_bidirectional_encoder:
				decoder_inputs = [tf.zeros([self.batch_size, self.state_size])] * (self.pv_size + 1)  # [[B,state_size]*(N+1)]
			else:
				decoder_inputs = [tf.zeros([self.batch_size, self.state_size*2])] * (self.pv_size + 1)  # [[B,state_size]*(N+1)]

		return decoder_inputs

	def predictions_op(self):
		with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
			# self.inference_prediction_order = tf.stack(self.inference_prediction_order, axis=1)	# [B,N]
			# self.training_prediction_order = tf.stack(self.training_prediction_order, axis=1)	# [B,N]
			# self.predictions = tf.Print(self.predictions, [self.inference_prediction_order, self.predictions], message="check training_prediction_order/inference_prediction_order/inference_prediction_score", first_n=100, summarize=10000)
			return self.predictions

	def build_model_input(self):
		self.model_input = self.get_candidates_non_sequential_feature_representation()
		if len(set(self.fg.seq_feature_conf_map.keys())) > 0:
			self.seq_feature_representation = self.get_candidates_sequential_feature_representation()  # [B,1,D]
			seq_feature_representation = tf.tile(self.seq_feature_representation, [1, self.N, 1])  # [B,N,D]
			self.model_input = tf.concat([seq_feature_representation, self.model_input], axis=-1)  # [B,N,D]
		self.batch_size = tf.shape(self.model_input)[0]

		# 监控特征被tanh打满的比例
		delta = 0.039
		feature_reach_upper_bound = tf.where(self.model_input > (1 - delta), tf.ones_like(self.model_input), tf.zeros_like(self.model_input))	# [B,N,D]
		tf.summary.scalar("feature_reach_upper_bound", tf.reduce_mean(tf.reduce_mean(feature_reach_upper_bound, axis=-1)))
		feature_reach_lower_bound = tf.where(self.model_input < (-1 + delta), tf.ones_like(self.model_input), tf.zeros_like(self.model_input))  # [B,N,D]
		tf.summary.scalar("feature_reach_lower_bound", tf.reduce_mean(tf.reduce_mean(feature_reach_lower_bound, axis=-1)))

		# position feature.
		if self.add_position_info == "position_feature":
			position_feature = self.get_position_feature(self.N)  # [B,N,1]
			self.model_input = tf.concat([self.model_input, position_feature], axis=-1)

		self.input_size = self.model_input.get_shape()[-1].value
		print "[print model input]: ", self.model_input

	def metrics_op(self):
		with tf.name_scope("{}_Metrics".format(self.name)):
			if self.is_local:
				self.current_auc, self.total_auc = metrics.auc(labels=self.label,
															   predictions=self.predictions,
															   num_thresholds=2000)
			else:
				with tf.device(self.hyper_params.worker_device):
					self.current_auc, self.total_auc = metrics.auc(labels=self.label,
																   predictions=self.predictions,
																   num_thresholds=2000)
			# scalar
			self.metrics['scalar/auc'] = self.current_auc
			self.metrics['scalar/total_auc'] = self.total_auc

			self.metrics['scalar/loss'] = self.loss
			self.metrics['scalar/reg_loss'] = self.reg_loss

			self.metrics['scalar/better_percentage'] = tf.reduce_mean(self.better_percentage)  # [1]
			self.metrics['scalar/discount_better_percentage'] = tf.reduce_mean(self.discount_better_percentage)  # [1]
			self.metrics['scalar/inference_order_better_percentage'] = tf.reduce_mean(self.inference_order_better_percentage)  # [1]
			self.metrics['scalar/base_order_evaluator_reward'] = tf.reduce_mean(self.base_order_evaluator_score)	# [1]
			self.metrics['scalar/training_order_evaluator_reward'] = tf.reduce_mean(self.reranking_order_evaluator_score)	# [1]
			self.metrics['scalar/inference_order_evaluator_reward'] = tf.reduce_mean(self.reranking_inference_order_evaluator_score)  # [1]
			self.metrics['scalar/training_order_evaluator_reward_advantage'] = tf.reduce_mean(self.better_difference)	# [1]
			self.metrics['scalar/discount_training_order_evaluator_reward_advantage'] = tf.reduce_mean(self.discount_better_difference)	# [1]
			self.metrics['scalar/inference_order_evaluator_reward_advantage'] = tf.reduce_mean(self.inference_order_better_difference)	# [1]
		self.attention_weight_monitor()

	def attention_weight_monitor(self):
		with tf.name_scope('All_Step_Mean_Metrics'):
			attention_weight = tf.nn.softmax(tf.stack(self.inference_attention_distribution[:-1], axis=1))	# [B,N,N]
			tf.summary.histogram('attention_weight_all_step_histogram', attention_weight)

			attention_weight_mean = tf.reduce_mean(attention_weight, axis=1)	# [B,N]
			attention_weight_mean = tf.reduce_mean(attention_weight_mean, axis=0)	# [N]
			attention_weight_mean = tf.unstack(attention_weight_mean, self.candidate_size)	# [1]*N
			for i in range(len(attention_weight_mean)):
				tf.summary.scalar('attention_weight_all_step_mean_index_{}'.format(i), attention_weight_mean[i])

		inference_prediction_order = tf.unstack(self.inference_prediction_order,num=self.pv_size,axis=-1)	# [B]*N
		for i in range(len(inference_prediction_order)):
			with tf.name_scope('Step_{}_Mean_Metrics'.format(i)):
				tf.summary.histogram('index_histogram_step_{}'.format(i), inference_prediction_order[i])

				step_attention_weight = tf.nn.softmax(self.inference_attention_distribution[i])	# [B,N]
				step_attention_weight = tf.reduce_mean(step_attention_weight,axis=0,keep_dims=True)	# [1,N]

				# image = tf.tile(step_attention_weight,[self.pv_size,1])	# [N,N]
				# image = tf.reshape(image, [-1,self.pv_size,self.pv_size,1])	# [1,N,N,1]
				# tf.summary.image('attention_weight_heatmap_step_{}'.format(i), image)

				scores = tf.unstack(tf.squeeze(step_attention_weight),num=self.candidate_size,axis=0)	# [1]*N
				for j in range(len(scores)):
					tf.summary.scalar('index_{}'.format(j), scores[j])

	def set_labels(self):
		self.label = self.features['click_label']	# [B,N,1]

	def collect_execution_ops(self):
		self.train_ops = {
			"run_ops":{
				"train_op": self.train_op
				, "localvar": self.localvar
				, "metrics": self.metrics
				, "label": self.training_prediction_order	# [B,N][0,1,2,3,4,5]
				, "loss": self.loss
				, "global_step": self.global_step
			},
			"reset_auc_ops": self.reset_auc_ops,
			"auc": self.current_auc,
			"total_auc": self.total_auc,
			"loss": self.loss
		}

		self.predict_ops = {
			"run_ops": {"predictions": self.predictions * self.expansion,
						"origin_predictions": self.predictions,
						"clk_label":self.features['click_label'],
						"buy_label":self.features['buy_label'],
						"id": self.id,
						"prediction_order": self.inference_prediction_order,
						"better_difference": self.inference_order_better_difference,	# [B,1]
						"better_percentage": self.inference_order_better_percentage,	# [B,1]
						"better_ratio": self.better_ratio,	# [B,1]
						"reranking_order_evaluator_score": self.reranking_order_evaluator_score,	# [B,N]
						"base_order_evaluator_score": self.base_order_evaluator_score,	# [B,N]
						"reranking_inference_order_evaluator_score": self.reranking_inference_order_evaluator_score	# [B,1]
						}
		}

	def build_target_weights(self):
		clk_label = tf.squeeze(self.label, axis=-1)  # [B,N]

		# target_weights: [B,N] -> [[B,1]*N]
		self.target_weights = tf.ones([self.batch_size, self.pv_size])	# [B,N]

		if self.target_weights_manner == "equal":
			pass
		elif self.target_weights_manner == "selective_zero_weight":
			values, indices = tf.nn.top_k(clk_label, k=self.pv_size, sorted=True)	# [B,N]
			self.target_weights = values
		elif self.target_weights_manner == "selective_small_weight":
			values, indices = tf.nn.top_k(clk_label, k=self.pv_size, sorted=True)	# [B,N]
			self.target_weights = tf.where(tf.equal(values, tf.zeros_like(values)), tf.add(values, 0.1), values)	# [B,N]
		elif self.target_weights_manner == "selective_big_weight":
			values, indices = tf.nn.top_k(clk_label, k=self.pv_size, sorted=True)  # [B,N]
			# self.target_weights = tf.add(values, 1.0)	# [B,N]
			self.target_weights = tf.where(tf.equal(values, tf.ones_like(values)), tf.add(values, 5.0), tf.add(values, 1.0))  # [B,N]
		elif isinstance(self.target_weights_manner, int) or isinstance(self.target_weights_manner, float):
			target_weights_multiple = self.target_weights * self.target_weights_manner	# [B,N]
			self.target_weights = tf.where(self.evaluator_reward>0, target_weights_multiple, self.target_weights)	# [B,N]

		# self.target_weights = tf.Print(self.target_weights, [self.target_weights], message="check target_weights", first_n=10, summarize=10000)

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
				reward_boolean_flag = self.get_max_reward_index()	# [B*hill_climbing_candidate_num]
				self.target_weights = tf.Print(self.target_weights, [tf.shape(self.target_weights), tf.shape(logits), tf.shape(labels), self.target_weights, logits, labels], message="before mask shape", first_n=10, summarize=10000)
				logits = tf.boolean_mask(logits, reward_boolean_flag)	# [B]
				labels = tf.boolean_mask(labels, reward_boolean_flag)	# [B]
				self.target_weights = tf.boolean_mask(self.target_weights, reward_boolean_flag)	# [B]
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
				self.loss += -probabilities * tf.reduce_mean(self.target_weights, axis=-1)	# [B]
			self.loss = tf.reduce_mean(self.loss)
			tf.summary.scalar("main_loss", self.loss)

			if "attention_weight_entropy_loss" in self.extra_loss:
				attention_weight = tf.nn.softmax(logits) + self.epsilon
				if self.extra_loss == "attention_weight_entropy_loss_1":
					self.attention_weight_entropy_loss_weight /= 20.0
					attention_weight_entropy_loss = tf.reduce_sum(tf.log(attention_weight), axis=-1)	# [B,10]
				elif self.extra_loss == "attention_weight_entropy_loss_2":
					attention_weight_entropy_loss = -tf.reduce_sum(attention_weight * tf.log(attention_weight), axis=-1)	# [B,10]
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
					self.attention_weight_entropy_loss_weight /= 3.0
				tf.summary.scalar("loss_weight_decay", loss_weight_decay)
				attention_weight_entropy_loss = self.attention_weight_entropy_loss_weight * loss_weight_decay * attention_weight_entropy_loss
				tf.summary.scalar("final_attention_weight_entropy_loss", attention_weight_entropy_loss)
				self.loss += attention_weight_entropy_loss

			is_adamw = (self.optimizer == "AdamAsync" and self.weight_decay != 0)
			if not is_adamw:
				print 'use reg_loss.'
				self.loss = self.loss + self.reg_loss

	def get_global_step_linear_decay(self, global_step, max_global_step, min_value):
		return tf.maximum(1.0 - global_step / max_global_step, min_value)

	def linear_increase(self):
		min_value = self.baseline_reward_discount_ratio
		max_value = 1.0

		global_step = tf.cast(self.global_step, tf.float32)
		max_global_step = tf.cast(self.max_global_step, tf.float32)

		result = min_value + (max_value - min_value) * (global_step / max_global_step)
		result = tf.clip_by_value(result, min_value, max_value)

		return result

	def get_max_reward_index(self):
		reward = tf.reduce_mean(self.evaluator_reward, axis=-1, keep_dims=True)	# [B*hill_climbing_candidate_num,1]
		reward = tf.reshape(reward, [-1, self.hill_climbing_candidate_num])	# [B,hill_climbing_candidate_num]
		reward_boolean_flag = tf.equal(reward, tf.reduce_max(reward, axis=-1, keep_dims=True))	# [B,hill_climbing_candidate_num]
		self.target_weights = tf.Print(self.target_weights,[reward,reward_boolean_flag],message="check hill climbing 1",first_n=10,summarize=10000)
		reward_boolean_flag = tf.reshape(reward_boolean_flag, [-1])	# [B*hill_climbing_candidate_num]

		hill_climbing_reward_difference = tf.reduce_max(reward, axis=-1) - tf.reduce_min(reward, axis=-1)  # [B]
		self.metrics['scalar/hill_climbing_reward_difference'] = tf.reduce_mean(hill_climbing_reward_difference)  # [1]

		return reward_boolean_flag

	def sort_by_indices(self, indices, values):	# ([B,N,input_size],[B,N])->[B,N,input_size]
		batch_ids = tf.ones_like(indices) * tf.expand_dims(
			tf.range(tf.shape(input=indices)[0]), 1)
		nd_indices = tf.stack([batch_ids, indices], axis=-1)		# [B,N,2]
		return tf.gather_nd(values, nd_indices)

	def symbol_to_index_pair(self, index_matrix):
		# [[3,1,2], [2,3,1]] -> [[[0 3] [0 1] [0 2]],
		#                        [[1 2] [1 3] [1 1]]]
		replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
		rank = len(index_matrix.get_shape())
		if rank == 2:
			replicated_first_indices = tf.tile(
				tf.expand_dims(replicated_first_indices, dim=1),
				[1, tf.shape(index_matrix)[1]])
		return tf.stack([replicated_first_indices, index_matrix], axis=rank)

	def get_sampling_function(self):
		self.inference_sampled_symbol = tf.zeros([self.batch_size, self.N])	# [B,N]
		self.inference_prediction_order = []
		self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]

		def sampling_function(attention_weights, _):
			attention_weights = attention_weights * self.sample_polarization_degree
			# attention_weights = tf.Print(attention_weights, [_,attention_weights[0]], message="inference attention_weights", first_n=1000000, summarize=1000000)
			if self.use_masking:
				attention_weights = tf.where(self.inference_sampled_symbol > 0, self.neg_inf, attention_weights)	# [B,N]
			attention_weights = tf.nn.softmax(attention_weights)

			if self.sample_manner == "greedy":
				# 1、greedy
				sampling_symbol = math_ops.argmax(attention_weights, 1) # [B,N] -> [B]
			else:
				greedy_result = math_ops.argmax(attention_weights, 1)  # [B,N] -> [B]
				# 2、sample
				sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights),1),axis=-1)	# [B,N] -> [B]
				# sampling_symbol = tf.distributions.Categorical(probs=attention_weights).sample()  # [B,N] -> [B]
				# sampling_symbol = tf.Print(sampling_symbol, [sampling_symbol, greedy_result], message="check inference sample", first_n=100, summarize=1000)
			sampling_symbol = tf.cast(sampling_symbol, tf.int32)	# [B]
			self.inference_prediction_order.append(sampling_symbol)

			if self.use_masking:
				sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)	# [B,N]
				# ***** #
				sampling_symbol_onehot = tf.where(self.inference_sampled_symbol > 0, tf.zeros_like(sampling_symbol_onehot), sampling_symbol_onehot)  # [B,N]
				# ***** #
				self.inference_sampled_symbol += sampling_symbol_onehot	# [B,N]

				sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot	# [B,N]
			# sampling_symbol = tf.Print(sampling_symbol,
			# 						   [_, sampling_symbol[0], sampling_symbol_score[0], self.inference_sampled_symbol[0],
			# 							attention_weights[0]],
			# 						   message="check sampling_symbol/sampling_symbol_score/inference_sampled_symbol/attention_weights",
			# 						   first_n=10000000, summarize=1000000000)

			if self.decoder_input_from == "encoder_input":
				embedding_matrix = self.model_input	# [B,N,input_size]
			elif self.decoder_input_from == "encoder_output":
				embedding_matrix = self.encoder_states	# [B,N,state_size]
			sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(sampling_symbol)) # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
			if not self.update_embedding:
				sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
			return sampling_symbol_embedding, sampling_symbol_score

		return sampling_function

	def get_training_sampling_function(self):
		self.training_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
		self.training_prediction_order = []
		self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]

		def sampling_function(attention_weights, _):
			attention_weights = attention_weights * self.sample_polarization_degree
			# attention_weights = tf.Print(attention_weights, [_, attention_weights[0]], message="training attention_weights", first_n=1000000, summarize=1000000)
			if self.use_masking:
				attention_weights = tf.where(self.training_sampled_symbol > 0, self.neg_inf, attention_weights)  # [B,N]
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
			if not self.update_embedding:
				sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
			return sampling_symbol_embedding, sampling_symbol_score

		return sampling_function

	def mark_output(self, predictions):	# [B,N,1]
		with tf.name_scope("{}_Mark_Output".format(self.name)):
			predictions = tf.reshape(predictions, [-1,1])
			# predictions = predictions * self.expansion
			predictions = tf.identity(predictions, name="rtp_trace_reranking_score")
			if '1.4' in tf.__version__:
				tf.identity(predictions, name="rank_predict")  # NEED TO BE [N,1]
			else:
				tf.identity(predictions, name="aop_predict")	# NEED TO BE [N,1]

	def build_evaluator_input(self):
		position_feature = self.get_position_feature(self.pv_size)  # [B,N,1]

		exposure_input = self.get_candidates_non_sequential_feature_representation(top_n=self.pv_size)
		exposure_input = tf.concat([exposure_input, position_feature], axis=-1)
		base_order_input = exposure_input  # [B,pv_size,input_size]

		reranking_order_input = self.get_candidates_non_sequential_feature_representation(order=self.training_prediction_order)
		reranking_order_input = tf.concat([reranking_order_input, position_feature], axis=-1)

		reranking_inference_order_input = self.get_candidates_non_sequential_feature_representation(order=self.inference_prediction_order)
		reranking_inference_order_input = tf.concat([reranking_inference_order_input, position_feature], axis=-1)

		if len(set(self.fg.seq_feature_conf_map.keys())) > 0:
			seq_feature_representation = tf.tile(self.seq_feature_representation, [1, self.pv_size, 1])  # [B,pv_size,D]
			base_order_input = tf.concat([seq_feature_representation, exposure_input], axis=-1)
			reranking_order_input = tf.concat([seq_feature_representation, reranking_order_input], axis=-1)
			reranking_inference_order_input = tf.concat([seq_feature_representation, reranking_inference_order_input], axis=-1)

		return base_order_input, reranking_order_input, reranking_inference_order_input

	def evaluate(self):
		# input
		base_order_input, reranking_order_input, reranking_inference_order_input = self.build_evaluator_input()	# [B,N,input_size]

		# evaluate
		if self.evaluator_type == "item_wise":
			self.reranking_order_evaluator_score = self.build_item_wise_evaluator_model(reranking_order_input)	# [B,N]
			self.base_order_evaluator_score = self.build_item_wise_evaluator_model(base_order_input)	# [B,N]
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
			self.reranking_order_pv_score = self.reranking_order_evaluator_score = self.build_pv_wise_evaluator_model(reranking_order_input)	# [B,1]
			self.base_order_pv_score = self.base_order_evaluator_score = self.build_pv_wise_evaluator_model(base_order_input)	# [B,1]
			self.reranking_inference_order_pv_score = self.reranking_inference_order_evaluator_score = self.build_pv_wise_evaluator_model(reranking_inference_order_input)  # [B,1]

		baseline_reward_discount_ratio = self.baseline_reward_discount_ratio
		if self.use_dynamic_baseline_reward_discount_ratio:
			baseline_reward_discount_ratio = self.linear_increase()
		tf.summary.scalar("baseline_reward_discount_ratio", baseline_reward_discount_ratio)

		# better percentage
		self.better_difference = self.reranking_order_pv_score - self.base_order_pv_score	# [B,1]
		self.discount_better_difference = self.reranking_order_pv_score - baseline_reward_discount_ratio * self.base_order_pv_score	# [B,1]
		self.inference_order_better_difference = self.reranking_inference_order_pv_score - self.base_order_pv_score	# [B,1]
		if "combined_model_and_prior_reward" in self.pv_value_evaluation:
			if self.pv_value_evaluation == "combined_model_and_prior_reward":
				prior_total_reward = self.calculate_list_prior_reward(self.training_prediction_order)	# [B,1]
			elif self.pv_value_evaluation == "combined_model_and_prior_reward_2":
				prior_total_reward = self.calculate_list_prior_reward_2(self.training_prediction_order)	# [B,1]
			self.better_difference = self.better_difference	+ prior_total_reward	# [B,1]
			self.metrics['scalar/combined_reward'] = tf.reduce_mean(self.better_difference)
		self.better_percentage = tf.where(self.better_difference > 0, tf.ones_like(self.better_difference), tf.identity(tf.zeros_like(self.better_difference)))  # [B,1]
		self.discount_better_percentage = tf.where(self.discount_better_difference > 0, tf.ones_like(self.discount_better_difference), tf.identity(tf.zeros_like(self.discount_better_difference)))  # [B,1]
		self.inference_order_better_percentage = tf.where(self.inference_order_better_difference > 0, tf.ones_like(self.inference_order_better_difference), tf.identity(tf.zeros_like(self.inference_order_better_difference)))  # [B,1]
		self.better_ratio = tf.divide(self.better_difference, self.reranking_order_pv_score + self.base_order_pv_score)	# [B,1]

		# reward
		self.build_evaluator_reward()

	def calculate_list_prior_reward(self, reranking_order):
		base_order = tf.tile(tf.expand_dims(tf.range(self.pv_size), axis=0), [self.batch_size, 1])  # [B,pv_size]
		# [B,1]
		base_order_buy_label_num, base_order_buy_label_avg_pos = self.calculate_list_label_info("base_order", base_order, "prior_buy_reward", "buy_label")
		reranking_order_buy_label_num, reranking_order_buy_label_avg_pos = self.calculate_list_label_info("reranking_order", reranking_order, "prior_buy_reward", "buy_label")
		prior_buy_reward = reranking_order_buy_label_num - base_order_buy_label_num	# [B,1]
		candidate_buy_label_num = tf.count_nonzero(tf.squeeze(self.features["buy_label"], axis=-1), axis=-1, keep_dims=True, dtype=tf.float32)  # [B,1]
		prior_buy_reward += 0.5 * (reranking_order_buy_label_num - candidate_buy_label_num)	# [B,1]
		self.metrics['scalar/prior_buy_reward'] = tf.reduce_mean(prior_buy_reward)	# [B,1]

		base_order_click_label_num, base_order_click_label_avg_pos = self.calculate_list_label_info("base_order", base_order, "prior_click_reward", "click_label")
		reranking_order_click_label_num, reranking_order_click_label_avg_pos = self.calculate_list_label_info("reranking_order", reranking_order, "prior_click_reward", "click_label")
		prior_click_reward = reranking_order_click_label_num - base_order_click_label_num	# [B,1]
		candidate_click_label_num = tf.count_nonzero(tf.squeeze(self.features["click_label"], axis=-1), axis=-1, keep_dims=True, dtype=tf.float32)  # [B,1]
		prior_click_reward += 0.5 * (reranking_order_click_label_num - candidate_click_label_num)	# [B,1]
		self.metrics['scalar/prior_click_reward'] = tf.reduce_mean(prior_click_reward)

		prior_total_reward = prior_buy_reward + prior_click_reward / 50.0
		self.metrics['scalar/prior_total_reward'] = tf.reduce_mean(prior_total_reward)

		return prior_total_reward

	def calculate_list_prior_reward_2(self, reranking_order):
		base_order = tf.tile(tf.expand_dims(tf.range(self.pv_size), axis=0), [self.batch_size, 1])  # [B,pv_size]
		# [B,1]
		base_order_buy_label_num, base_order_buy_label_avg_pos = self.calculate_list_label_info("base_order", base_order, "prior_buy_reward", "buy_label")
		reranking_order_buy_label_num, reranking_order_buy_label_avg_pos = self.calculate_list_label_info("reranking_order", reranking_order, "prior_buy_reward", "buy_label")
		prior_buy_reward = tf.nn.tanh(reranking_order_buy_label_num - base_order_buy_label_num)	# [B,1]
		candidate_buy_label_num = tf.count_nonzero(tf.squeeze(self.features["buy_label"], axis=-1), axis=-1, keep_dims=True, dtype=tf.float32)  # [B,1]
		prior_buy_reward += 0.5 * tf.nn.tanh(reranking_order_buy_label_num - candidate_buy_label_num)	# [B,1]
		self.metrics['scalar/prior_buy_reward'] = tf.reduce_mean(prior_buy_reward)	# [B,1]

		base_order_click_label_num, base_order_click_label_avg_pos = self.calculate_list_label_info("base_order", base_order, "prior_click_reward", "click_label")
		reranking_order_click_label_num, reranking_order_click_label_avg_pos = self.calculate_list_label_info("reranking_order", reranking_order, "prior_click_reward", "click_label")
		prior_click_reward = 2 * tf.nn.tanh(reranking_order_click_label_num - base_order_click_label_num)	# [B,1]
		candidate_click_label_num = tf.count_nonzero(tf.squeeze(self.features["click_label"], axis=-1), axis=-1, keep_dims=True, dtype=tf.float32)  # [B,1]
		prior_click_reward += tf.nn.tanh(reranking_order_click_label_num - candidate_click_label_num)	# [B,1]
		self.metrics['scalar/prior_click_reward'] = tf.reduce_mean(prior_click_reward)

		prior_total_reward = prior_buy_reward + prior_click_reward / 30.0
		self.metrics['scalar/prior_total_reward'] = tf.reduce_mean(prior_total_reward)

		return prior_total_reward

	def calculate_list_label_info(self, list_name, list_order, prior_reward_name, label_name):
		list_label = tf.squeeze(self.sort_by_indices(indices=list_order, values=self.features[label_name]), axis=-1)	# [B,pv_size]
		list_label_num = tf.count_nonzero(list_label, axis=-1, keep_dims=True, dtype=tf.float32)	# [B,1]

		pos_index = tf.tile(tf.expand_dims(tf.range(1, self.pv_size + 1, dtype=tf.float32), axis=0), [self.batch_size, 1])	# [B,pv_size]
		list_label_pos = tf.where(list_label > 0, pos_index, tf.zeros_like(pos_index))	# [B,pv_size]
		list_label_pos_sum = tf.reduce_sum(list_label_pos, axis=-1, keep_dims=True)	# [B,1]
		list_label_avg_pos = list_label_pos_sum / (list_label_num + self.epsilon)	# [B,1]

		self.metrics['scalar/'+list_name+'_'+prior_reward_name+'_'+'label_num'] = tf.reduce_mean(list_label_num)
		self.metrics['scalar/'+list_name+'_'+prior_reward_name+'_'+'label_avg_pos'] = tf.reduce_sum(list_label_avg_pos) / (tf.count_nonzero(list_label_avg_pos, dtype=tf.float32) + self.epsilon)

		return list_label_num, list_label_avg_pos

	def build_item_wise_evaluator_model(self, input):
		# 1、dnn layer
		input = self._dnn_layer(input)

		# 2、logit layer
		logits = self._logits_layer(input)

		# 3、prediction layer
		with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
			predictions = tf.sigmoid(logits)
			predictions = tf.reshape(predictions, [-1, self.pv_size])  # [B,N]

		return predictions

	def build_pv_wise_evaluator_model(self, input):
		# 1、sum_pooling_channel
		with tf.variable_scope(name_or_scope="{}_Sum_Pooling_Channel".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
			sum_pooling_layer = tf.reduce_sum(input, axis=1)
			logging.info("[DEBUG] sum_pooling_layer: {}".format(sum_pooling_layer))

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

		# 3、multi_head_self_attention_channel
		with tf.variable_scope(name_or_scope="{}_Multi_Head_Self_Attention_Channel".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
			shape_list = input.get_shape().as_list()
			all_feature_concatenation = tf.reshape(input, [-1, self.pv_size, shape_list[2] ])
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

		# 5、pair_wise_comparison_channel
		with tf.variable_scope(name_or_scope="{}_Pair_Wise_Comparison_Channel".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
			input_transposed = tf.transpose(input, perm=[0, 2, 1])
			output = tf.matmul(input, input_transposed)
			pair_wise_comparison_layer = tf.reshape(output, [-1, self.pv_size * self.pv_size])
			logging.info("[DEBUG] pair_wise_comparison_layer: {}".format(pair_wise_comparison_layer))

		# 6、dnn_layer
		dnn_layer = [sum_pooling_layer, concatenation_layer,
					 multi_head_self_attention_layer, rnn_layer, pair_wise_comparison_layer]
		dnn_layer = tf.concat(values=dnn_layer, axis=-1)
		logging.info("[DEBUG] dnn_layer: {}".format(dnn_layer))
		final_neurons = self._dnn_layer(dnn_layer)

		# 7、logits_layer
		logits = self._logits_layer(final_neurons)

		# 8、prediction layer
		with tf.name_scope("{}_Predictions_Op".format(self.name)):
			predictions = tf.sigmoid(logits)
			predictions = tf.reshape(predictions, [-1, 1])  # [B,1]

		return predictions

	# 构造evaluator的dnn专用
	def _dnn_layer(self, input_dnn_net):
		with tf.variable_scope(name_or_scope="{}_Score_Network".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			running_layer = input_dnn_net
			with arg_scope(base_ops.model_arg_scope(weight_decay=self.dnn_l2_reg)):
				for layer_id, num_hidden_units in enumerate(self.dnn_hidden_units):
					with variable_scope.variable_scope("hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
						if self.use_BN:
							running_layer = layers.fully_connected(
								running_layer,
								num_hidden_units,
								utils.getActivationFunctionOp(self.activation),
								scope=dnn_hidden_layer_scope,
								variables_collections=[self.collections_dnn_hidden_layer],
								outputs_collections=[self.collections_dnn_hidden_output],
								normalizer_fn=layers.batch_norm,
								normalizer_params={"scale": True, "is_training": False})
						else:
							running_layer = layers.fully_connected(
								running_layer,
								num_hidden_units,
								utils.getActivationFunctionOp(self.activation),
								scope=dnn_hidden_layer_scope,
								variables_collections=[self.collections_dnn_hidden_layer],
								outputs_collections=[self.collections_dnn_hidden_output]
							)
							if self.use_dropout:
								running_layer = layers.dropout(running_layer,
															   keep_prob=self.dropout_keep_prob,
															   is_training=self.is_training)
				final_neurons = running_layer
				return final_neurons

	def _logits_layer(self, final_neurons):
		logging.info("[DEBUG] logits_layer starts: {}".format(final_neurons))
		with tf.variable_scope(name_or_scope="{}_Logits".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num,
																self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE) as dnn_logits_scope:
			with arg_scope(base_ops.model_arg_scope(weight_decay=self.dnn_l2_reg)):
				logits = layers.linear(
					final_neurons,
					1,
					scope=dnn_logits_scope,
					variables_collections=[self.collections_dnn_hidden_layer],
					outputs_collections=[self.collections_dnn_hidden_output])
		return logits

	def resnet_layer(self, input_tensor):
		with tf.variable_scope(name_or_scope="{}_Residual_Network".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			resnet_last_hidden_units_num = -1
			if len(self.dnn_hidden_units) > 0:
				# resnet默认接在dnn之后
				resnet_last_hidden_units_num = self.dnn_hidden_units[-1]
			with arg_scope(base_ops.model_arg_scope(weight_decay=self.dnn_l2_reg)):
				for layer_id, num_hidden_units in enumerate(self.resnet_hidden_units):
					with variable_scope.variable_scope("residual_block_{}".format(layer_id)) as resnet_hidden_layer_scope:
						is_identity = (num_hidden_units == resnet_last_hidden_units_num)
						print '[resnet] layer_id = ', layer_id, ', resnet_last_hidden_units_num = ', resnet_last_hidden_units_num, ', num_hidden_units = ', num_hidden_units, ', is_identity = ', is_identity
						input_tensor = self.residual_block(
							input_tensor=input_tensor,
							units=num_hidden_units,
							is_identity=is_identity,
							is_training=False
						)
						resnet_last_hidden_units_num = num_hidden_units
				final_neurons = input_tensor
				return final_neurons

	def normalize_reward(self, reward):
		reward_mean, reward_var = tf.nn.moments(reward, axes=[0,1])
		reward_std = tf.sqrt(reward_var)
		reward = (reward - reward_mean) / (reward_std + self.epsilon)	# [B,N]

		# reward = tf.Print(reward,[reward],message="reward_normalization",first_n=10,summarize=10000)
		self.metrics['scalar/batch_reward_mean'] = reward_mean
		self.metrics['scalar/batch_reward_std'] = reward_std

		return reward

	def min_max_normalize_reward(self, reward):
		reward_min = tf.reduce_min(reward)
		reward_max = tf.reduce_max(reward)
		reward = (reward - reward_min) / (reward_max - reward_min + self.epsilon)	# [B,N]

		# reward = tf.Print(reward, [reward], message="reward_normalization", first_n=10, summarize=10000)
		self.metrics['scalar/batch_reward_min'] = reward_min
		self.metrics['scalar/batch_reward_max'] = reward_max

		return reward

	def build_evaluator_reward(self):
		if self.reward_method == "reinforce":
			gamma = 0.3

			reinforce_weight = tf.range(self.pv_size, dtype=tf.float32)
			reinforce_weight = tf.reshape(reinforce_weight, [-1, 1])	# [N,1]
			reinforce_weight = tf.tile(reinforce_weight, [1, self.pv_size])	# [N,N]
			reinforce_weight = reinforce_weight - tf.transpose(reinforce_weight)	# [N,N]
			reinforce_weight = tf.where(reinforce_weight>=0, tf.pow(gamma, reinforce_weight), tf.zeros_like(reinforce_weight))	# [N,N]

			self.evaluator_reward = tf.matmul(self.reranking_order_evaluator_score, reinforce_weight)	# [B,N]
			# self.evaluator_reward = tf.Print(self.evaluator_reward,
			# 												[tf.reduce_mean(self.better_percentage)],
			# 												message="better percentage", first_n=10, summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward,
			# 												[self.reranking_order_pv_score],
			# 												message="reranking_order_pv_score", first_n=10,
			# 												summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward,
			# 												[self.base_order_pv_score],
			# 												message="base_order_pv_score", first_n=10,
			# 												summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward,
			# 												[self.reranking_order_evaluator_score],
			# 												message="reranking_order_evaluator_score", first_n=10,
			# 												summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward,
			# 												[self.evaluator_reward],
			# 												message="evaluator_reward", first_n=10, summarize=10000)

			if self.reward_normalization:
				self.evaluator_reward = self.min_max_normalize_reward(self.evaluator_reward)	# [B,N]

		elif "lzd_with_baseline" in self.reward_method:
			if self.reward_method == "lzd_with_baseline":
				reward = self.discount_better_difference
			elif self.reward_method == "lzd_with_baseline_v2":
				reward = tf.clip_by_value(self.better_ratio,-10,10)
			elif self.reward_method == "lzd_with_baseline_v3":
				reward = tf.sign(self.better_difference) * 2 * tf.pow(tf.abs(self.better_difference), 1.5)
			self.evaluator_reward = tf.tile(reward, [1, self.pv_size])	# [B,N]

			# self.evaluator_reward = tf.Print(self.evaluator_reward, [tf.reduce_mean(self.better_percentage)],
			# 								 message="better percentage", first_n=10, summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward, [self.reranking_order_pv_score],
			# 								 message="reranking_order_pv_score", first_n=10, summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward, [self.base_order_pv_score],
			# 								 message="base_order_pv_score", first_n=10, summarize=10000)
			# self.evaluator_reward = tf.Print(self.evaluator_reward, [reward],
			# 								 message="reward", first_n=10, summarize=10000)

			if self.reward_normalization:
				self.evaluator_reward = self.normalize_reward(self.evaluator_reward)	# [B,N]

		elif self.reward_method == "reinforce_v2":
			self.evaluator_reward = self.reranking_order_evaluator_score	# [B,N]

			if self.reward_normalization:
				self.evaluator_reward = self.min_max_normalize_reward(self.evaluator_reward)	# [B,N]

		elif self.reward_method == "reinforce_v2_with_baseline":
			item_score_mean = tf.reduce_mean(self.reranking_order_evaluator_score, axis=-1, keep_dims=True)	# [B,1]
			self.evaluator_reward = self.reranking_order_evaluator_score - item_score_mean	# [B,N]

			if self.reward_normalization:
				self.evaluator_reward = self.normalize_reward(self.evaluator_reward)	# [B,N]

		elif "real_pv_label" in self.reward_method:
			if self.reward_method == "real_pv_label":
				pv_label = tf.reduce_max(self.label, axis=1)	# [B,1]
				reward = tf.where(pv_label > 0, tf.ones_like(pv_label), -tf.ones_like(pv_label))	# [B,1]
			elif self.reward_method == "real_pv_label_with_baseline":
				pv_label = tf.reduce_max(self.label, axis=1)	# [B,1]
				exposed_ctr_feature = tf.where(self.exposure_label > 0, self.features["context__ctr_score"], tf.zeros_like(self.exposure_label))	# [B,candidate_size,1]
				expected_pv_label = 1 - tf.reduce_prod(1 - exposed_ctr_feature, axis=1)	# [B,1]
				reward = pv_label - expected_pv_label  # [B,1]
				# self.label = tf.Print(self.label, [self.exposure_label[0], self.features["context__ctr_score"][0], exposed_ctr_feature[0], pv_label[0], expected_pv_label[0], reward[0]], message="check reward", first_n=100, summarize=100000)
			elif self.reward_method == "real_pv_label_count":
				pv_label_count = tf.reduce_sum(self.label, axis=1)	# [B,1]
				reward = tf.where(pv_label_count > 0, pv_label_count / 10.0, -tf.ones_like(pv_label_count))	# [B,1]
			elif self.reward_method == "real_pv_label_count_with_baseline":
				pv_label_count = tf.reduce_sum(self.label, axis=1)	# [B,1]
				exposed_ctr_feature = tf.where(self.exposure_label > 0, self.features["context__ctr_score"], tf.zeros_like(self.exposure_label))	# [B,candidate_size,1]
				expected_pv_label_count = tf.reduce_sum(exposed_ctr_feature, axis=1)	# [B,1]
				reward = pv_label_count - expected_pv_label_count	# [B,1]
				# self.label = tf.Print(self.label, [self.exposure_label[0], self.features["context__ctr_score"][0], exposed_ctr_feature[0], pv_label_count[0], expected_pv_label_count[0], reward[0]], message="check reward", first_n=100, summarize=100000)
			self.evaluator_reward = tf.tile(reward, [1, self.pv_size])	# [B,N]
			# self.evaluator_reward = tf.Print(self.evaluator_reward, [self.label[0], self.evaluator_reward[0], self.better_percentage[0], self.inference_order_better_percentage[0], self.training_prediction_order[0], self.inference_prediction_order[0]], message="check evaluator_reward", first_n=1000, summarize=100000)

	def get_pv_value_lower_bound(self, score_list):
		pv_value = 1 - tf.reduce_prod(1 - score_list, axis=-1, keep_dims=True)	# [B,1]
		return pv_value

	def get_pv_value_upper_bound(self, score_list):
		pv_value = tf.reduce_sum(score_list, axis=-1, keep_dims=True)	# [B,1]
		return pv_value

	def warm_start(self):
		with tf.name_scope(name="Partially_recovery"):
			recovery_score = ["{}_Concatenation_Channel/", "{}_Multi_Head_Self_Attention_Channel/", "{}_RNN_Channel/",
							  "{}_Score_Network/", "{}_Residual_Network/", "{}_Logits/", "input_from_feature_columns/", "{}_Extra_Logits/",
							  "{}_sequential_feature_representation/"]
			for scope in recovery_score:
				src_name = scope.format(self.name)
				target_name = scope.format(self.name)
				logging.info("Restore params from {} to {}".format(src_name, target_name))
				checkpoint_utils.init_from_checkpoint(self.restore_dir, {src_name: target_name})

	def training_op(self):
		global_variables = tf.global_variables()
		print "global_variables = ", len(global_variables), global_variables
		print "distinct variable scope = ", set([var.name.split("/")[0] for var in global_variables])

		trainable_var = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
		print "[Before selection] trainable_var = ", len(trainable_var), trainable_var
		trainable_var = [var for var in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES) if
						 (self.name not in var.name and "input_from_feature_columns" not in var.name)]
		print "[After selection] trainable_var = ", len(trainable_var), trainable_var

		update_ops = self.update_op()
		with tf.variable_scope(name_or_scope="Optimize", reuse=tf.AUTO_REUSE):
			gs = tf.train.get_or_create_global_step()
			logging.info("[training_op] Global_step:{},{}".format(self.name, str(gs)))
			self.train_op, _, _ = myopt.optimize_loss(
				loss=self.loss,
				global_step=self.global_step,
				learning_rate=self.initial_learning_rate,
				optimizer=utils.get_optimizer(self.get_optimizer_conf(),
											  global_step=gs,
											  learning_rate=self.initial_learning_rate,
											  learning_rate_decay_fn=self.learning_rate_decay_cs),
				update_ops=update_ops,
				clip_gradients=self.clip_gradients,
				variables=trainable_var,
				increment_global_step=True,
				summaries=myopt.OPTIMIZER_SUMMARIES)

			self.train_ops.update({"train_op": self.train_op})