# -*- coding: utf-8 -*-
import tensorflow as tf
from common.model_utils import ops as base_ops, checkpoint_utils
from common.model_utils import utils
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns
from prada_model_ops import list_wise_feature_column_ops
from common.base_model import BaseModel
from common.model_utils.tflog import tflogger as logging
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.ops import nn_ops
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
import prada_model_ops.metrics as metrics
from common.model_zoo.din import BaseDIN
import numpy as np
from common.model_utils.general_util import *
from common.model_utils.prada_util_attention import multihead_attention

class BaseModelReranking(BaseModel):
	def __init__(self, hyper_params, name="CTR"):
		super(BaseModelReranking, self).__init__(hyper_params, name)
		
		self.epsilon = 0.00001
		self.real_list_size_feature_name = "user__list_size"

	def build(self, features, feature_columns, labels):
		"""Creates all ops for training and evaluation."""
		self.build_placeholder()
		self.set_is_online()
		self.build_inputs(features, feature_columns, labels)
		self.get_pv_size()

		self.build_model()

		self.set_global_step()
		self.set_labels()
		self.loss_op()

		self.training_op()

		predictions = self.predictions_op()
		self.mark_output(predictions)

		self.summary_op()

		self.setup_reset_op()

		self.collect_execution_ops()

		if self.restore_dir != "" and not self.is_local and self.run_mode.lower() == "train":
			self.warm_start()

	def min_max_normalization(self, raw_feature, tensor_global_min_tile, tensor_global_max_tile):
		matrix_f_global = tf.where(tensor_global_max_tile - tensor_global_min_tile < self.epsilon,
								   tf.fill(tf.shape(raw_feature), 0.5),
								   tf.div(tf.subtract(raw_feature, tensor_global_min_tile),
										  tf.subtract(tensor_global_max_tile, tensor_global_min_tile) + self.epsilon))	# (B, N, emb_dims)
		return matrix_f_global

	def divide_mean_normalization(self, raw_feature):
		raw_feature_mean = tf.reduce_mean(raw_feature, axis=1, keep_dims=True)  # (B, 1, emb_dims)
		matrix_f_global = tf.divide(raw_feature, raw_feature_mean + self.epsilon)  # (B, N, emb_dims)
		return matrix_f_global

	def augment_context_features(self, raw_feature):
		with tf.name_scope("{}_Context_Augmentation".format(self.name)):
			N = tf.shape(raw_feature)[1]
			tensor_global_max = tf.reduce_max(raw_feature, axis=1, keep_dims=True)  # (B, 1, d2)
			tensor_global_min = tf.reduce_min(raw_feature, axis=1, keep_dims=True)  # (B, 1, d2)
			tensor_global_max_tile = tf.tile(tensor_global_max, [1, N, 1])  # (B, N, d2)
			tensor_global_min_tile = tf.tile(tensor_global_min, [1, N, 1])  # (B, N, d2)

			if self.augment_feature_normalization == "divide_mean":
				matrix_f_global = self.divide_mean_normalization(raw_feature)
			elif self.augment_feature_normalization == "min_max":
				matrix_f_global = self.min_max_normalization(raw_feature, tensor_global_min_tile,
															 tensor_global_max_tile)
				matrix_f_global = matrix_f_global - 0.5

			tensor_global_mean = tf.divide(tf.reduce_sum(raw_feature, axis=1, keep_dims=True),
										   tf.cast(N, dtype=tf.float32))  # (B, 1, emb_dims)
			tensor_global_mean_tile = tf.tile(tensor_global_mean, [1, N, 1])  # (B, 17, d2)

			tensor_global_sigma = tf.reduce_mean(tf.square(raw_feature - tensor_global_mean_tile), axis=1,
												 keep_dims=True)
			tensor_global_sigma_tile = tf.tile(tensor_global_sigma, [1, N, 1])  # (B, 17, d2)
			tensor_global_sigma_tile = tf.where(tf.equal(tensor_global_sigma_tile,0),
												tensor_global_sigma_tile+self.epsilon,
												tensor_global_sigma_tile)

			raw_feature_pv_norm = tf.where(tf.sqrt(tensor_global_sigma_tile) < self.epsilon,
										   tf.fill(tf.shape(raw_feature), 0.0),
										   (raw_feature - tensor_global_mean_tile) / (
													   tf.sqrt(tensor_global_sigma_tile) + self.epsilon))  # [B,N,D]

			augmented_feature_list = [raw_feature, tensor_global_mean_tile, tensor_global_sigma_tile,
									  tensor_global_max_tile, tensor_global_min_tile, matrix_f_global,
									  raw_feature_pv_norm]

		return tf.concat(augmented_feature_list, axis=-1)

	def set_is_online(self):
		self.is_online = tf.equal(self.is_training, False)
		if self.run_mode.lower() == 'predict':
			self.is_online = False

	def get_pv_size(self):
		if self.real_list_size_feature_name in self.features:
			self.N = tf.cast(tf.reduce_max(self.features[self.real_list_size_feature_name]), tf.int32)
		else:
			self.N = self.candidate_size

	def set_labels(self):
		'''
		One needs to assign value to self.label here
		:return:
		'''
		raise NotImplementedError("Please Implement Function: set_labels")

	def embedding_layer(self):
		with tf.variable_scope(name_or_scope="input_from_feature_columns",
							   partitioner=base_ops.partitioner(self.ps_num, self.embedding_partition_size),
							   reuse=tf.AUTO_REUSE) as scope:
			for block_name in self._column_blocks:
				# 非 seq, output shape is [batch_size, list_size, dim_size]
				self.block_layer_dict[block_name] = list_wise_feature_column_ops.input_from_feature_columns(
					columns_to_tensors=self.features,
					feature_columns=self.feature_columns[block_name],
					weight_collections=None,
					trainable=True,
					scope=scope,
					list_size=self.candidate_size
				)
			for block_name in set(self.fg.seq_feature_conf_map.keys()):
				# seq, output shape is [batch_size, list_size, seq_len, dim_size]
				self.block_layer_dict[block_name] = list_wise_feature_column_ops.input_from_seq_feature_columns(
					columns_to_tensors=self.features,
					feature_columns=self.feature_columns[block_name],
					weight_collections=None,
					trainable=True,
					scope=scope,
					list_size=self.candidate_size,
					seq_len=self.fg.get_seq_len_by_sequence_name(block_name)
				)

	def cutting_input(self):
		with tf.name_scope("{}_Cutting_Input".format(self.name)):
			seq_feature_name = []
			for seq_column_name in set(self.fg.seq_feature_conf_map.keys()):
				true_seq_feature_name = [seq_column_name+'_'+name for name in self.mc.get_column_names_by_block_name(seq_column_name)]
				seq_feature_name.extend(true_seq_feature_name)

			for k, v in self.features.items():
				if isinstance(v, tf.Tensor):
					if k in seq_feature_name:
						self.features[k] = v[:, :self.N, :, :]	# [B,N,seq_len,dims]
					else:
						self.features[k] = v[:, :self.N, :]	# [B,N,dims]
				elif isinstance(v, tf.SparseTensor):
					if k in seq_feature_name:
						self.features[k] = tf.sparse_tensor_to_dense(v, "")[:, :self.N, :, :]	# [B,N,seq_len,dims]
					else:
						self.features[k] = tf.sparse_tensor_to_dense(v, "")[:, :self.N, :]	# [B,N,dims]

			for k, v in self.block_layer_dict.items():
				if k in set(self.fg.seq_feature_conf_map.keys()):
					self.block_layer_dict[k] = v[:, :self.N, :, :]	# [B,N,seq_len,dims]
				else:
					self.block_layer_dict[k] = v[:, :self.N, :]	# [B,N,dims]

	def build_inputs(self, features, feature_columns, labels):
		with tf.name_scope("{}_Build_Inputs".format(self.name)):
			self.id = features['id']
			self.features = {key:val for key, val in features.items() if key not in ["id", "label", "uniqueId"]}
			self.feature_columns = feature_columns
			self.label = labels	# not really in use

	def build_model(self):
		if self.restore_embedding and not self.is_local:
			self.feature_columns = self.setup_feature_columns()
		self.embedding_layer()
		self.cutting_input()

		self.dnn_layer()
		self.logits_layer()

	def dnn_layer(self):
		dnn_layer = self.get_candidates_non_sequential_feature_representation()
		if len(set(self.fg.seq_feature_conf_map.keys())) > 0:
			seq_feature_representation = self.get_candidates_sequential_feature_representation()	# [B,1,D]
			seq_feature_representation = tf.tile(seq_feature_representation, [1, self.N, 1])	# [B,N,D]
			dnn_layer = tf.concat([seq_feature_representation, dnn_layer], axis=-1)	# [B,N,D]
		self.batch_size = tf.shape(dnn_layer)[0]

		# position feature.
		if self.add_position_info == "position_feature":
			position_feature = self.get_position_feature(self.N)	# [B,N,1]
			dnn_layer = tf.concat([dnn_layer, position_feature], axis=-1)

		# It may be useful for collect_execution_ops in prediction mode if you want to print dnn input.
		self.dnn_input = dnn_layer
		self._dnn_layer(dnn_layer)

	def get_candidates_sequential_feature_representation(self):
		for block_name in set(self.fg.seq_feature_conf_map.keys()):
			seq_feature_representation = self.block_layer_dict[block_name]	# [B,list_size,seq_len,D]
			seq_feature_representation = tf.reduce_mean(seq_feature_representation, axis=1)	# [B,seq_len,D]

		with tf.variable_scope(name_or_scope="{}_sequential_feature_representation".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
							   reuse=tf.AUTO_REUSE):
			seq_feature_representation = layers.fully_connected(
				seq_feature_representation,
				128,
				utils.getActivationFunctionOp(self.activation),
				variables_collections=[self.collections_dnn_hidden_layer],
				outputs_collections=[self.collections_dnn_hidden_output]
			)
			seq_feature_representation = tf.reduce_sum(seq_feature_representation, axis=1, keep_dims=True)	# [B,1,128]
			seq_feature_representation = utils.getActivationFunctionOp('relu')(seq_feature_representation)
			return seq_feature_representation

	def get_position_feature(self, length):
		position_feature = tf.range(1, length + 1, 1.0) / tf.cast(self.pv_size, tf.float32)
		position_feature = tf.reshape(position_feature, [-1, length, 1])
		position_feature = tf.tile(position_feature, [self.batch_size, 1, 1])  # (B, N, 1)
		return position_feature

	def get_position_embedding(self, length, width):
		position_embedding = tf.get_variable(
			name="position_embedding",
			shape=[length, width],
			initializer=tf.truncated_normal_initializer(stddev=0.02))
		position_embedding = tf.reshape(position_embedding, [-1, length, width])
		return position_embedding

	def loss_op(self):
		with tf.name_scope("{}_Loss_Op".format(self.name)):
			self.reg_loss_f()
			if self.loss_type == "pair_wise":
				self.loss = self.pair_wise_loss(self.label, self.logits)
			elif self.loss_type == "point_wise":
				self.loss = self.point_wise_loss(self.label, self.logits)
			elif self.loss_type == "bpr_loss":
				self.loss = self.bpr_loss(self.label, self.logits)
			is_adamw = (self.optimizer == "AdamAsync" and self.weight_decay != 0)
			if not is_adamw:
				print 'use reg_loss.'
				self.loss = self.loss + self.reg_loss

	def mark_output(self, predictions):	# [B,N,1]
		with tf.name_scope("{}_Mark_Output".format(self.name)):
			pseudo_doc_length = self.candidate_size - self.N
			pseudo_doc_score = tf.zeros([self.batch_size, pseudo_doc_length, 1])
			predictions = tf.reshape(predictions, [-1, self.N, 1])
			predictions = tf.concat([predictions, pseudo_doc_score], axis=1)	# [B,candidate_size,1]
			predictions = tf.reshape(predictions, [-1,1])
			tf.identity(predictions, name="aop_predict")	# NEED TO BE [N,1]

	def summary_op(self):
		self.metrics_op()
		with tf.name_scope("{}_Metrics_Scalar".format(self.name)):
			for key, metric in self.metrics.items():
				tf.summary.scalar(name=key, tensor=metric)

		with tf.name_scope("{}_Layer_Summary".format(self.name)):
			base_ops.add_norm2_summary(self.collections_dnn_hidden_layer)
			base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)
			base_ops.add_weight_summary(self.collections_dnn_hidden_layer)

		with tf.name_scope('{}_Embedding_Summary'.format(self.name)):
			for block_name, layer in self.block_layer_dict.items():
				if not self.mc.has_block(block_name):
					continue
				self.add_embed_layer_norm(layer, self.feature_columns[block_name])

	def add_embed_layer_norm(self, layer_tensor, columns):
		if layer_tensor is None:
			return
		i = 0
		for column in sorted(set(columns), key=lambda x: x.key):
			try:
				dim = column.dimension
			except:
				dim = column.embedding_dimension
			layer_tensor_rank = len(layer_tensor.shape)
			layer_tensor_slice = layer_tensor
			if layer_tensor_rank == 3:
				layer_tensor_slice = layer_tensor[:, :, i:i + dim]
			elif layer_tensor_rank == 2:
				layer_tensor_slice = layer_tensor[:, i:i + dim]
			elif layer_tensor_rank == 1:
				layer_tensor_slice = tf.reshape(layer_tensor, [-1,1])
			tf.summary.scalar(name=column.name, tensor=tf.reduce_mean(tf.norm(layer_tensor_slice, axis=-1)))
			i += dim

	def point_wise_loss(self, label, logits):
		loss_click = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits=tf.reshape(logits, [-1]),
				labels=tf.reshape(label, [-1]))
		)
		return loss_click

	def pair_wise_loss(self, label, logits):
		with tf.name_scope("loss_op"):
			n = self.N
			probabilities = tf.sigmoid(logits)

			sj = tf.tile(tf.reshape(probabilities, [-1, 1, n]), [1, n, 1])
			si = tf.tile(tf.reshape(probabilities, [-1, n, 1]), [1, 1, n])
			probabilities_difference = sj - si
			probabilities_difference = tf.reshape(probabilities_difference, [-1, n * n])
			# Avoid tf.norm's input is zero tensor, may cause a "nan in summary" error.
			probabilities_difference = tf.add(probabilities_difference,
											  tf.random_normal(shape=tf.shape(probabilities_difference), mean=0.0,
															   stddev=self.epsilon))

			lj = tf.tile(tf.reshape(label, [-1, 1, n]), [1, n, 1])
			li = tf.tile(tf.reshape(label, [-1, n, 1]), [1, 1, n])
			labels_difference = lj - li
			labels_difference = tf.reshape(labels_difference, [-1, n * n])

			# cosine similarity
			loss_matrix = probabilities_difference * labels_difference
			loss_matrix = tf.div(tf.reduce_sum(loss_matrix, 1, keep_dims=True),
								 tf.norm(probabilities_difference, axis=1, keep_dims=True) * tf.norm(labels_difference,
																									 axis=1,
																									 keep_dims=True) + self.epsilon)

			loss = (- tf.reduce_sum(loss_matrix)) / tf.add(tf.count_nonzero(loss_matrix, dtype=tf.float32), self.epsilon)
		return loss

	def bpr_loss(self, label, logits):
		# label and logits: [B,N,1]
		with tf.name_scope("loss_op"):
			n = self.N
			probabilities = logits

			# 1 1 1
			# 2 2 2
			# 3 3 3
			si = tf.tile(tf.reshape(probabilities, [-1, n, 1]), [1, 1, n])  # [B,N,1] -> [B,N,N]
			# 1 2 3
			# 1 2 3
			# 1 2 3
			sj = tf.tile(tf.reshape(probabilities, [-1, 1, n]), [1, n, 1])	# [B,1,N] -> [B,N,N]
			probabilities_difference = si - sj	# [B,N,N]

			# 1 1 1
			# 2 2 2
			# 3 3 3
			li = tf.tile(tf.reshape(label, [-1, n, 1]), [1, 1, n])
			# 1 2 3
			# 1 2 3
			# 1 2 3
			lj = tf.tile(tf.reshape(label, [-1, 1, n]), [1, n, 1])
			labels_difference = li - lj

			bpr_mask = tf.greater(labels_difference, 0)
			# [B,N,N] + [B,N,N] -> [valid_num]
			bpr_logits = tf.boolean_mask(probabilities_difference, bpr_mask)
			# [valid_num]
			bpr_pseudo_labels = tf.ones_like(bpr_logits)
			# weight = self.features['reserve_double1']
			# weight_tile = tf.tile(weight,[1, 1, n])
			# bpr_weight_tile = tf.boolean_mask(weight_tile, bpr_mask)

			# pairwise loss [valid_num]
			weighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=bpr_logits, labels=bpr_pseudo_labels)
			# pointwise loss [B,N,1]
			weight_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label)
			# pairwise + pointwise loss
			loss = tf.reduce_mean(weighted_loss) + tf.reduce_mean(weight_label_loss)

		return loss

	def warm_start(self):
		raise NotImplementedError("Please Implement Function: warm_start")

	def get_candidates_non_sequential_feature_representation(self, top_n=None, order=None):
		block_layer_dict = {}
		for block_name in self._column_blocks:
			block_layer_dict[block_name] = self.block_layer_dict[block_name]

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

	def run_hidden_layers(self, input, hidden_units, variable_scope_prefix, is_training=None, use_BN=None, use_LN=False):
		if is_training is None:
			is_training = self.is_training
		if use_BN is None:
			use_BN = self.use_BN
		running_layer = input
		for layer_id, num_hidden_units in enumerate(hidden_units):
			with variable_scope.variable_scope("{}_hiddenlayer_{}".format(variable_scope_prefix, layer_id)) as dnn_hidden_layer_scope:
				if use_BN:
					running_layer = layers.fully_connected(
						running_layer,
						num_hidden_units,
						utils.getActivationFunctionOp(self.activation),
						scope=dnn_hidden_layer_scope,
						variables_collections=[self.collections_dnn_hidden_layer],
						outputs_collections=[self.collections_dnn_hidden_output],
						normalizer_fn=layers.batch_norm,
						normalizer_params={"scale": True, "is_training": is_training})
				elif use_LN:
					running_layer = layers.fully_connected(
						running_layer,
						num_hidden_units,
						utils.getActivationFunctionOp(self.activation),
						scope=dnn_hidden_layer_scope,
						variables_collections=[self.collections_dnn_hidden_layer],
						outputs_collections=[self.collections_dnn_hidden_output],
						normalizer_fn=layers.layer_norm,
						# layer-norm是对样本维度计算均值和方差，归一化时不同样本使用的统计均值和统计方差肯定不一样，不是对batch维度的每一个特征计算均值和方差，然后公式就是原始输入减去某批样本的统计均值除以统计标准差，再乘以可学习的缩放参数，再加上可学习的偏差参数
						# begin_norm_axis这个参数表示从输入tensor的第几维开始到最后一维一起算出一个统计均值和方差，假设输入是个3维tensor，如果begin_norm_axis=1，得到的统计均值就是[batch_size,1,1]，如果begin_norm_axis=-1，得到的统计均值就是[batch_size,list_size,1]，统计方差同理
						# begin_params_axis这个参数表示需要从输入tensor的第几维开始到最后一维具有可学习的缩放参数和偏差参数，然后前面的维度就是共享的，假设输入是个3维tensor，如果begin_params_axis=1，需要的缩放参数就是[1,list_size,dimension]，如果begin_params_axis=-1，需要的缩放参数就是[1,1,dimension]，偏差参数同理
						normalizer_params={"begin_norm_axis": self.layer_norm_begin_norm_axis, "begin_params_axis": self.layer_norm_begin_params_axis})
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
		return running_layer

	def get_logits(self, input, logits_dims, variable_scope_or_prefix=None):
		if isinstance(variable_scope_or_prefix, str):
			with tf.variable_scope("{}_Logits".format(variable_scope_or_prefix)) as dnn_logits_scope:
				logits = layers.linear(
					input,
					logits_dims,
					scope=dnn_logits_scope,
					variables_collections=[self.collections_dnn_hidden_layer],
					outputs_collections=[self.collections_dnn_hidden_output])
		else:
			logits = layers.linear(
				input,
				logits_dims,
				scope=variable_scope_or_prefix,
				variables_collections=[self.collections_dnn_hidden_layer],
				outputs_collections=[self.collections_dnn_hidden_output])
		return logits

	def get_expectation_from_logits(self, logits):
		pv_click_num_probability_distribution = tf.nn.softmax(logits)  # [B,pv_size+1]
		click_num_template = tf.range(pv_click_num_probability_distribution.get_shape()[-1].value, dtype=tf.float32)
		click_num_template = tf.expand_dims(click_num_template, axis=0)
		expectation = tf.reduce_sum(pv_click_num_probability_distribution * click_num_template, axis=-1)  # [B]
		return expectation

	def resnet_layer(self, input_tensor):
		with tf.variable_scope(name_or_scope="{}_Residual_Network".format(self.name),
							   partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
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
							is_training=self.is_training
						)
						resnet_last_hidden_units_num = num_hidden_units
				self.final_neurons = input_tensor
				return self.final_neurons

	def residual_block(self, input_tensor, units, is_identity=True, is_training=True):
		"""构建一个残差块，可以是恒等块或卷积块。"""
		# x = tf.contrib.layers.fully_connected(input_tensor, units, activation_fn=None,
		# 									  weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
		# 									  variables_collections=[self.collections_dnn_hidden_layer],
		# 									  outputs_collections=[self.collections_dnn_hidden_output],
		# 									  scope='fc1')
		# x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None, scope='bn1')
		x = layers.fully_connected(
			input_tensor,
			units,
			activation_fn=None,
			weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			variables_collections=[self.collections_dnn_hidden_layer],
			outputs_collections=[self.collections_dnn_hidden_output],
			normalizer_fn=layers.batch_norm,
			normalizer_params={"scale": True, "is_training": is_training},
			scope='fc1'
		)
		x = tf.nn.relu(x)

		# x = tf.contrib.layers.fully_connected(x, units, activation_fn=None,
		# 									  weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
		# 									  variables_collections=[self.collections_dnn_hidden_layer],
		# 									  outputs_collections=[self.collections_dnn_hidden_output],
		# 									  scope='fc2')
		# x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None, scope='bn2')
		x = layers.fully_connected(
			x,
			units,
			activation_fn=None,
			weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			variables_collections=[self.collections_dnn_hidden_layer],
			outputs_collections=[self.collections_dnn_hidden_output],
			normalizer_fn=layers.batch_norm,
			normalizer_params={"scale": True, "is_training": is_training},
			scope='fc2'
		)
		x = tf.nn.relu(x)

		if not is_identity:
			# 如果是卷积块，用一个全连接层调整输入维度以匹配输出维度
			# input_tensor = tf.contrib.layers.fully_connected(input_tensor, units, activation_fn=None,
			# 												 weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
			# 												 variables_collections=[self.collections_dnn_hidden_layer],
			# 												 outputs_collections=[self.collections_dnn_hidden_output],
			# 												 scope='fc3')
			# input_tensor = tf.contrib.layers.batch_norm(input_tensor, scale=True, is_training=is_training,
			# 											updates_collections=None, scope='bn3')
			input_tensor = layers.fully_connected(
				input_tensor,
				units,
				activation_fn=None,
				weights_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
				variables_collections=[self.collections_dnn_hidden_layer],
				outputs_collections=[self.collections_dnn_hidden_output],
				normalizer_fn=layers.batch_norm,
				normalizer_params={"scale": True, "is_training": is_training},
				scope='fc3'
			)

		x = tf.add(x, input_tensor)
		x = tf.nn.relu(x)
		return x

	def parse_hyper_params(self, hyper_params):
		super(BaseModelReranking, self).parse_hyper_params(hyper_params)

		algo_conf = get_config_value(hyper_params.algo_configs, self.name, hyper_params.algo_configs)
		job_conf = get_config_value(hyper_params.job_configs, self.name, hyper_params.job_configs)
		self.augmenting_blocks = [v.strip() for v in get_config_value(algo_conf, "augmenting_blocks", 'item_columns;context_columns').split(";") if v in self.main_column_blocks]
		self.pv_size = get_config_value(algo_conf, "pv_size", 10)
		self.candidate_size = get_config_value(algo_conf, "candidate_size", 10)
		self.run_mode = job_conf["run_mode"]
		self.use_LN = get_config_value(algo_conf, "use_LN", False)
		print "augmenting_blocks = ", self.augmenting_blocks
		print "use_LN = ", self.use_LN

		model_conf = get_config_value(algo_conf, "modelx", {})
		self.loss_type = model_conf.get("loss_type", "pair_wise")
		self.augment_feature_normalization = model_conf.get("augment_feature_normalization", "divide_mean")
		self.add_position_info = model_conf.get("add_position_info", "")
		self.channels = [channel.strip() for channel in get_config_value(model_conf, "channels", 'all_channels').split(";")]
		self.resnet_hidden_units = get_config_value(model_conf, "resnet_hidden_units", [])
		self.layer_norm_begin_norm_axis = get_config_value(model_conf, "layer_norm_begin_norm_axis", -1)
		self.layer_norm_begin_params_axis = get_config_value(model_conf, "layer_norm_begin_params_axis", -1)
		print "channels = ", self.channels
		print "resnet_hidden_units = ", self.resnet_hidden_units
		print "layer_norm_begin_norm_axis = ", self.layer_norm_begin_norm_axis
		print "layer_norm_begin_params_axis = ", self.layer_norm_begin_params_axis

		pipeline_restore_dir = get_config_value(hyper_params.job_configs, "restore_dir", "")
		if pipeline_restore_dir != "":
			self.restore_dir = pipeline_restore_dir
		print "restore_dir = ", self.restore_dir