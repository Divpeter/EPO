from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util
import model_utils.optimizer_ops as myopt
import prada_model_ops.metrics as metrics

from common.model_utils import ops as base_ops
from common.model_utils import utils
from common.model_utils.tflog import tflogger as logging
from common.model_utils.general_util import *
from common.model_column import ModelColumn
from common.fg_parser import FgParser
import warnings

class BaseModel(object):

    def __init__(self, hyper_params, name="CTR"):
        self.name = name
        self.parse_hyper_params(hyper_params)

        # assumed model competences
        self.block_layer_dict = OrderedDict()   # Feature values after apply tf.feature_column (e.g. after looking up embedding) and orgnized in blocks
        self.final_neurons = None               # The hidden status value before the last linear layer for the logits
        self.logits = None                      # A float32 Tensor with shape [batch_size, 1].
        self.predictions = None                 # A float32 Tensor with shape [batch_size, 1]. after sigmoid
        self.loss = None                        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.reg_loss = None                    # Regularization loss, (optionally) added to the final loss
        self.metrics = {}                       # A dictionary recording various metrics such as training CTR.
        self.is_training = None                 # Is training flag for ops such as batch norm.

        # required dicts for execution
        self.train_ops = {}
        self.predict_ops = {}
        self.evaluate_ops = {}
        self.validate_ops = {}

    def build(self, features, feature_columns, labels):
        """Creates all ops for training and evaluation."""
        self.build_inputs(features, feature_columns, labels)
        self.build_placeholder()

        self.build_model()

        self.set_global_step()
        self.loss_op()

        self.training_op()

        predictions = self.predictions_op()
        self.mark_output(predictions)

        self.summary_op()

        self.setup_reset_op()

        self.collect_execution_ops()

    def embedding_layer(self):
        with tf.variable_scope(name_or_scope="input_from_feature_columns",
                               partitioner=base_ops.partitioner(self.ps_num, self.embedding_partition_size),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in set(self.mc.get_column_block_dict().keys()) - set(self.fg.seq_feature_conf_map.keys()):
                self.block_layer_dict[block_name] = layers.input_from_feature_columns(self.features,
                                                                                      self.feature_columns[block_name],
                                                                                      scope=scope)

    def dnn_layer(self):
        dnn_layer = [self.block_layer_dict[block_name] for block_name in self.main_column_blocks]
        self._dnn_layer(tf.concat(values=dnn_layer, axis=1))

    def _dnn_layer(self, input_dnn_net):
        with tf.variable_scope(name_or_scope="{}_Score_Network".format(self.name),
                               partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size)):
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
                                normalizer_params={"scale": True, "is_training": self.is_training})
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
                self.final_neurons = running_layer
                return self.final_neurons

    def logits_layer(self):
        logging.info("[DEBUG] logits_layer starts: {}".format(self.final_neurons))
        with tf.variable_scope(name_or_scope="{}_Logits".format(self.name),
                               partitioner=base_ops.partitioner(self.ps_num,
                                                                self.dnn_partition_size)) as dnn_logits_scope:
            with arg_scope(base_ops.model_arg_scope(weight_decay=self.dnn_l2_reg)):
                self.logits = layers.linear(
                    self.final_neurons,
                    1,
                    scope=dnn_logits_scope,
                    variables_collections=[self.collections_dnn_hidden_layer],
                    outputs_collections=[self.collections_dnn_hidden_output])

    def build_model(self):
        if self.restore_embedding and not self.is_local:
            self.feature_columns = self.setup_feature_columns()
        self.embedding_layer()
        self.dnn_layer()
        self.logits_layer()

    def update_op(self):
        update_ops = []
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            if update_op.name.startswith(self.name):
                update_ops.append(update_op)
        return update_ops

    def training_op(self):
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
                variables=ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES),
                increment_global_step=True,
                summaries=myopt.OPTIMIZER_SUMMARIES)

            self.train_ops.update({"train_op": self.train_op})

        global_variables = tf.global_variables()
        print "global_variables = ", len(global_variables), global_variables
        trainable_var = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        print "trainable_var = ", len(trainable_var), trainable_var

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
            self.metrics['scalar/label_mean'] = tf.reduce_mean(self.label)
            self.metrics['scalar/logits_mean'] = tf.reduce_mean(self.logits)
            self.metrics['scalar/predictions_mean'] = tf.reduce_mean(self.predictions)

    def summary_op(self):
        self.metrics_op()
        with tf.name_scope("{}_Metrics_Scalar".format(self.name)):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)

        with tf.name_scope("{}_Layer_Summary".format(self.name)):
            base_ops.add_norm2_summary(self.collections_dnn_hidden_layer)
            # base_ops.add_dense_output_summary(self.collections_dnn_hidden_output)
            base_ops.add_weight_summary(self.collections_dnn_hidden_layer)

        with tf.name_scope('{}_Embedding_Summary'.format(self.name)):
            for block_name, layer in self.block_layer_dict.items():
                if not self.mc.has_block(block_name):
                    continue
                base_ops.add_embed_layer_norm(layer, self.feature_columns[block_name])

    @property
    def model_name(self):
        return self.name

    def setup_reset_op(self):
        self.reset_auc_ops, self.local_auc_variables = utils.reset_variables(tf.GraphKeys.LOCAL_VARIABLES, 'Metrics/auc')

    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def setup_cl_global_step(self):
        # single task step not in tf.GraphKeys.GLOBAL_STEP collection
        global_step = tf.Variable(
            initial_value=0,
            name="{}_global_step".format(self.name),
            trainable=False,
            dtype=tf.int64,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def mark_output(self, predictions):
        with tf.name_scope("{}_Mark_Output".format(self.name)):
            tf.identity(predictions, name="rank_predict")

    def predictions_op(self):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.predictions = tf.sigmoid(self.logits)
            return self.predictions

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_losses = []
        for reg_loss in reg_losses:
            if reg_loss.name.startswith(self.name):
                self.reg_losses.append(reg_loss)
        self.reg_loss = tf.reduce_sum(self.reg_losses)

    def loss_op(self):
        with tf.name_scope("{}_Loss_Op".format(self.name)):
            self.reg_loss_f()
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.label))
            self.loss = self.loss + self.reg_loss

    def _setup_feature_columns(self, feature_name_list, feature_column_list, fg_config_dict, target_embedding_name):
        for i in range(len(feature_name_list)):
            feature_name = feature_name_list[i]
            if feature_name in fg_config_dict \
                    and "shared_name" in fg_config_dict[feature_name] \
                    and fg_config_dict[feature_name]["shared_name"] == target_embedding_name:
                # new_embedding_column is a list
                new_embedding_column = tf.contrib.layers.shared_embedding_columns(
                    sparse_id_columns=[feature_column_list[i].sparse_id_column],
                    dimension=fg_config_dict[feature_name]["embedding_dimension"],
                    shared_embedding_name=target_embedding_name,
                    ckpt_to_load_from=self.restore_dir,
                    tensor_name_in_ckpt=self.restore_embedding_names[target_embedding_name],
                    trainable=self.restore_embedding_trainable)
                new_embedding_column = new_embedding_column[0]
                logging.info("DEBUG: updating embedding from {} to {}".format(feature_column_list[i], new_embedding_column))
                feature_column_list[i] = new_embedding_column

    def setup_feature_columns(self):
        for block_name in  set(self.mc.get_column_block_dict().keys()) - set(self.fg.seq_feature_conf_map.keys()):
            # sequence feature is excluded here which facilitates easy function reuse.
            # assume block_name == seq_name == seq_feature_name
            for target_embedding_shared_name in self.restore_embedding_names.keys():
                self._setup_feature_columns(self.mc.get_column_block_dict()[block_name],
                                            self.feature_columns[block_name],
                                            self.fg.feature_conf_map, target_embedding_shared_name)
        return self.feature_columns

    def collect_execution_ops(self):
      self.train_ops = {
            "run_ops":{
                "train_op": self.train_op
                , "localvar": self.local_auc_variables
                , "metrics": self.metrics
                , "label": self.label
                , "loss": self.loss
                , "global_step": self.global_step
            },
            "reset_auc_ops": self.reset_auc_ops,
            "auc": self.current_auc,
            "total_auc": self.total_auc,
            "loss": self.loss
        }

      self.predict_ops = {
                "run_ops": {"predictions": self.predictions, "label": self.label, "id": self.id}
        }

    def build_placeholder(self):
        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def build_inputs(self, features, feature_columns, labels):
        self.id = self.features['id']
        self.features = features
        self.feature_columns = feature_columns
        self.label = labels

        recursive_log(self.features, "[DEBUG] self.features:")
        recursive_log(self.feature_columns, "[DEBUG] self.feature_columns:")
        recursive_log(self.label, "[DEBUG] self.labels:")

    def parse_hyper_params(self, hyper_params):
        '''
        All used parameters are required to parse to the Model instance directly,
        i.e. self.foo = hyper_params.algo_conf['foo'].
        '''

        '''
        First level parsing. Parameters parsed here are assumed to be used for all child models.
        '''
        self.hyper_params = hyper_params

        algo_conf = get_config_value(hyper_params.algo_configs, self.name, hyper_params.algo_configs)
        job_conf = get_config_value(hyper_params.job_configs, self.name, hyper_params.job_configs)
        fg_conf = get_config_value(hyper_params.fg_configs, self.name, hyper_params.fg_configs)

        self.fg_conf = fg_conf

        self.ps_num = job_conf["ps_num"]
        self.work_num = job_conf["worker_num"]
        self.is_local = True if str(get_config_value(job_conf, "is_local", True)).lower() == 'true' else False

        self.restore_embedding = False if str(
            get_config_value(algo_conf, "restore_embedding", "false")).lower() == "false" else True
        self.restore_dir = get_config_value(algo_conf, "restore_dir", "")
        self.restore_embedding_names = get_config_value(algo_conf, "restore_embedding_names", {})
        self.restore_embedding_trainable = True if str(
            get_config_value(algo_conf, "restore_embedding_trainable", "true")).lower() == "true" else False

        self.dnn_partition_size = get_config_value(algo_conf, "dnn_partition_size", 64 * 1024)
        self.embedding_partition_size = get_config_value(algo_conf, "embedding_partition_size", 8 * 1024 * 1024)

        model_conf = get_config_value(algo_conf, "modelx", None)
        self.activation = get_config_value(model_conf, "activation", "lrelu")
        self.dnn_l2_reg = get_config_value(model_conf, "dnn_l2_reg", 1e-6)
        self.dnn_hidden_units = get_config_value(model_conf, "dnn_hidden_units", [512, 256, 128])
        self.bias_dnn_hidden_units = get_config_value(model_conf, "bias_dnn_hidden_units", [512, 256, 128])

        self.use_BN = get_config_value(algo_conf, "use_BN", True)
        self.use_dropout = get_config_value(algo_conf, "use_dropout", False)
        self.dropout_keep_prob = get_config_value(algo_conf, "dropout_keep_prob", 0.5)

        self.main_column_blocks = [v.strip() for v in algo_conf['main_columns'].split(";") if v]

        self.collections_dnn_hidden_layer = "{}_dnn_hidden_layer".format(self.name)
        self.collections_dnn_hidden_output = "{}_dnn_hidden_output".format(self.name)

        if isinstance(get_config_value(algo_conf, "optimizer", None), dict):
            for opt_name, opt_conf in get_config_value(algo_conf, "optimizer", None).items():
                self.optimizer = opt_name
                # Params for learning_rate
                self.initial_learning_rate = get_config_value(opt_conf, "initial_learning_rate", 0.01)
                self.lrcs_init_lr = get_config_value(opt_conf, "lrcs_init_lr", 0.001)
                self.lrcs_init_step = get_config_value(opt_conf, "lrcs_init_step", 1000000)

                # Params for AdagradDecayOptimizer
                self.decay_step = get_config_value(opt_conf, "decay_step", 5000000)
                self.decay_rate = get_config_value(opt_conf, "decay_rate", 0.95)

                # Params for AdagradOptimizer
                self.initial_accumulator_value = get_config_value(opt_conf, "initial_accumulator_value", 0.1)
                self.optimizer_use_lock = get_config_value(opt_conf, "optimizer_use_lock", False)

                self.clip_gradients = get_config_value(opt_conf, "clip_gradients", 5.0)

                # Params for AdamAsyncOptimizer
                self.weight_decay = get_config_value(opt_conf, "weight_decay", 0)

        self.input_columns = job_conf['input_columns']
        self.mc = ModelColumn(get_config_value(self.input_columns, self.name, self.input_columns))
        self.fg = FgParser(self.fg_conf)

        # deprecated, for compatibility only
        self._column_blocks = set(self.mc.get_column_block_dict().keys()) - set(self.fg.seq_feature_conf_map.keys())

        '''
        Second level parsing. Parameters parsed here are assumed to be used for the single model.
        '''
        self.parse_model_specific_parameters()

    def parse_model_specific_parameters(self):
        '''
        All used parameters are required to parse to the Model instance directly,
        i.e. self.foo = self.hyper_params.algo_conf['foo'].

        It is recommended to give default values to all the core hyper parameters,
        especially for a shared model in the model zoo.
        :return:
        '''
        warnings.warn("self.parse_model_specific_parameters is deprecated. Please use self.parse_hyper_params instead", DeprecationWarning)
        pass

    def get_optimizer_conf(self):
        optimizer_conf = DottedDict()  # used by optimizer utility
        parameters = ["optimizer", "initial_learning_rate", "lrcs_init_lr", "lrcs_init_step",
                      "decay_step", "decay_rate", "initial_accumulator_value", "optimizer_use_lock", "clip_gradients",
                      "weight_decay"]
        for para in parameters:
            optimizer_conf[para] = getattr(self, para)
        return optimizer_conf

    def learning_rate_decay_cs(self, lr, gs):
        return utils.lr_cold_start(lr, gs, self.lrcs_init_lr, self.lrcs_init_step)

    '''
    Functions used to mark deprecation starts.
    '''
    @property
    def dnn_net(self):
        warnings.warn("self.dnn_net is deprecated. Please use self.final_neurons instead", DeprecationWarning)
        return self.final_neurons

    @dnn_net.setter
    def dnn_net(self, value):
        self.final_neurons = value

    @property
    def column_blocks(self):
        warnings.warn("self.column_blocks is deprecated. Please use self.mc instead", DeprecationWarning)
        return self._column_blocks

    @column_blocks.setter
    def column_blocks(self, value):
        self._column_blocks = value

    @property
    def localvar(self):
        warnings.warn("self.localvar is deprecated. Please use self.local_auc_variable instead", DeprecationWarning)
        return self.local_auc_variables

    @localvar.setter
    def localvar(self, value):
        self.local_auc_variable = value

    def trace_sample_op(self):
        warnings.warn("trace_sample_op is deprecated. It does nothing.", DeprecationWarning)
        pass

    '''
   Functions used to mark deprecation ends.
   '''