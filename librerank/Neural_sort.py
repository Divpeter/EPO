from librerank.rl_reranker import *
from librerank.reranker import *
from tensorflow.contrib import layers
from tensorflow.python.ops import rnn
from librerank.prada_util_attention import *
from librerank.utils import neural_sort
from librerank.utils import gumbel_sampling
# import tensorflow.python.framework.ops as ops
from tensorflow.python.framework import ops


class NS_generator(RLModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, acc_prefer=1.0, is_controllable=False, sample_val=0.2, gamma=0.01,
                 rep_num=1, loss_type='ce', model_name='NS_evaluator'):
        self.model_name = model_name
        super(NS_generator, self).__init__(feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                                           profile_num, max_norm, acc_prefer=acc_prefer,
                                           is_controllable=is_controllable)

    def build_ft_chosen(self, data_batch, chosen):
        itm_spar_ph, itm_dens_ph, length_seq = data_batch[2], data_batch[3], data_batch[6]
        batch_size, item_size = len(itm_spar_ph), len(itm_spar_ph[0])
        ret_spar, ret_dens = [], []
        for i in range(batch_size):
            spar_i, dens_i = [], []
            for j in range(item_size):
                if j < length_seq[i]:
                    spar_i.append(itm_spar_ph[i][chosen[i][j]])
                    dens_i.append(itm_dens_ph[i][chosen[i][j]])
                else:
                    spar_i.append(itm_spar_ph[i][length_seq[i]])
                    dens_i.append(itm_dens_ph[i][length_seq[i]])
            ret_spar.append(spar_i)
            ret_dens.append(dens_i)
        return np.array(ret_spar), np.array(ret_dens)

    def attention_based_decoder(self, decoder_inputs, initial_state, attention_states, cell, sampling_function,
                                attention_head_nums=1, feed_context_vector=True, dtype=dtypes.float32, scope=None):

        # if not decoder_inputs:
        #     raise ValueError("Must provide at least 1 input to attention decoder.")
        if attention_head_nums < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        # TODO: recover
        # if not attention_states.get_shape()[1:2].is_fully_defined():
        #     raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
        #                      % attention_states.get_shape())

        with vs.variable_scope(scope or "point_decoder"):
            batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            input_size = decoder_inputs[0].get_shape()[1].value  # input_size or state_size
            # TODO: recover
            # attn_length = attention_states.get_shape()[1].value # N+1 or N
            attn_length = tf.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value  # state_size——rnn output size

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            # [B,N+1,1,state_size] or [B,N,1,state_size]——encoder outputs
            hidden = tf.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

            attention_vec_size = attn_size  # state_size——Size of "query" vectors for attention.
            # size is CORRECT! Because both W1 and W2 are square matrix
            hidden_features = []
            v = []
            for a in range(attention_head_nums):
                k = vs.get_variable("AttnW_%d" % a,
                                    [1, 1, attn_size, attention_vec_size])  # [1,1,state_size,state_size]
                hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1],
                                                     "SAME"))  # transformation of encoder outputs, <BOS> in the front of encoder_outputs——W1ej in paper
                v.append(
                    vs.get_variable("AttnV_%d" % a, [attention_vec_size]))  # [state_size]

            states = [initial_state]  # list of all N's decoder state——[B,state_size], may be length of "N+1"

            def attention(query):  # query——[B,state_size], new state produced by decoder in current N
                """Point on hidden using hidden_features(W1ej)——[B,N+1,1,state_size] or [B,N,1,state_size] and query(decoder state)."""
                attention_weights = []  # Results of attention reads will be stored here.
                context_vector_list = []
                for a in range(attention_head_nums):
                    with vs.variable_scope("Attention_%d" % a):
                        if not self.is_controllable:
                            y = core_rnn_cell._linear(query, attention_vec_size,
                                                      True)  # [B,state_size], W2di in paper, linear transform, same shape as decoder state and encoder states
                        else:
                            y = self.get_hyper_dnn(query, [query.get_shape()[-1].value, 200, attention_vec_size],
                                                   [tf.nn.relu, None], "hyper_dec_dnn")
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])  # [B,1,1,state_size]
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(
                            v[a] * tf.tanh(hidden_features[a] + y),
                            [2, 3])  # [B,N+1,1,state_size]->[B,N+1] or [B,N,1,state_size]->[B,N]
                        # a = tf.nn.softmax(s)
                        attention_weights.append(s)

                        context_vector = tf.reduce_sum(tf.reshape(s, [-1, attn_length, 1, 1]) * hidden,
                                                       [1, 2])
                        context_vector = tf.reshape(context_vector, [-1, attn_size])
                        context_vector_list.append(context_vector)
                return attention_weights, context_vector_list

            outputs = []  # outputs: list of [B,N+1] or [B,N], may be length of "N+1", attention weight(ACTUAL OUTPUT) of each N's decoder state and all encoder output
            prev = None  # sampled vector
            batch_attn_size = tf.stack([batch_size, attn_size])
            attns = [
                tf.zeros(
                    batch_attn_size, dtype=dtype) for _ in range(attention_head_nums)
            ]  # [B,state_size]

            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])
            inps = []  # list of [B,input_size], decoder inputs, may be length of "N"(except for the first N) or "0"(it depends on "feed_prev")
            prediction_score = tf.zeros([batch_size, attn_length])  # [B,N]
            for i, inp in enumerate(decoder_inputs):  # [(N+1)*[B,input_size]], input into decoder N by N
                if i > 0:
                    vs.get_variable_scope().reuse_variables()

                # If sampling_function is set, we use it instead of decoder_inputs.
                if sampling_function is not None and prev is not None:
                    # TODO:reuse=True
                    with vs.variable_scope("sampling_function", reuse=tf.AUTO_REUSE):
                        inp, sampling_symbol_score = sampling_function(prev, i)
                        inps.append(inp)
                        prediction_score += sampling_symbol_score  # [B,N]
                        # self.dd.append(sampling_symbol_score)

                # Merge input and previous attentions into one vector of the right size.
                # projection dimension should be cell.input_size(input_size), but not cell.output_size (because it should be same with encoder)
                x = inp
                if feed_context_vector:
                    x = core_rnn_cell._linear([inp] + attns, input_size,
                                              True)  # [B,input_size]——union of input(origin decoder input[B,input_size] or weighted sum of decoder input[B,input_size]) and attns, finally, decoder input of each N is [B,input_size] nor [B,state_size]
                # Run the RNN.
                cell_output, new_state = cell(x, states[-1])  # [B,state_size], [B,state_size]
                states.append(new_state)
                # Run the attention mechanism.
                # TODO: attns should be weighted-sum of attention_states depends on new_state
                # TODO: and output should be linear combination of cell_output and new attns, and prev should be set as output, and then generate new inp?(if sampling_function is not none)
                output, attns = attention(
                    new_state)  # ([B,N+1] or [B,N]) * attention_head_nums, attention information of new decoder state and all encoder output
                output = tf.stack(output, axis=1)  # [B,attention_head_nums,N]
                output = tf.reduce_mean(output, axis=1)  # [B,N]

                if sampling_function is not None:
                    prev = output
                # The output of the pointer network is actually the attention weight!
                outputs.append(output)

        return outputs, states, prediction_score

    def deep_set_encode(self):
        self.enc_input = self.all_feature_concatenation
        self.encoder_states = self.get_dnn(self.enc_input, [200], [tf.nn.relu], "enc_dnn_1")  # [B*N or B, N, 200]
        self.final_state = tf.reduce_sum(self.encoder_states, axis=1)  # [B*N or B, 1, 200]
        self.final_state = self.get_dnn(self.final_state, [self.lstm_hidden_units], [tf.nn.relu],
                                        "enc_dnn_2")  # [B*N or B, 1, 200]

    def build_evaluator_input(self, itm_spar_ph, itm_dens_ph):
        self.itm_spar_emb = tf.gather(self.emb_mtx, tf.cast(itm_spar_ph, tf.int32))  # [?, 10, 5, 16]
        self.item_seq = tf.concat(
            [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]), itm_dens_ph],
            axis=-1)  # [?, 10, ft_num]
        self.itm_enc_input = tf.reshape(self.item_seq, [-1, self.item_size, self.ft_num])  # [B, N, ft_num]
        self.usr_enc_input = tf.reshape(self.usr_seq, [-1, 1, self.profile_num * self.emb_dim])
        raw_evaluator_input = tf.concat([self.itm_enc_input, tf.tile(self.usr_enc_input, [1, self.item_size, 1])],
                                        axis=-1)
        return raw_evaluator_input

    def rnn_decode(self):
        # build decoder input
        # training
        self.decoder_inputs = self.build_decoder_input()

        # build sampling function
        training_sampling_function = self.get_training_sampling_function()
        sampling_function = self.get_sampling_function()

        with tf.variable_scope("decoder",
                               # partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                               reuse=tf.AUTO_REUSE):
            training_attention_distribution, states, _ = self.attention_based_decoder(
                self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
                sampling_function=training_sampling_function, attention_head_nums=self.attention_head_nums,
                feed_context_vector=self.feed_context_vector)

        with tf.variable_scope("decoder",
                               # partitioner=base_ops.partitioner(self.ps_num, self.dnn_partition_size),
                               reuse=True):
            inference_attention_distribution, _, prediction_score = self.attention_based_decoder(
                self.decoder_inputs, self.final_state, self.encoder_states, self.decoder_cell,
                sampling_function=sampling_function, attention_head_nums=self.attention_head_nums,
                feed_context_vector=self.feed_context_vector)

        self.training_attention_distribution = training_attention_distribution
        self.training_prediction_order = tf.stack(self.training_prediction_order, axis=1)  # [B,N]
        self.inference_attention_distribution = inference_attention_distribution
        self.predictions = prediction_score  # [B,N]
        self.inference_prediction_order_record = tf.stack(self.inference_prediction_order, axis=1)  # [B,N]

        self.act_idx_out = self.training_prediction_order
        tmp_idx_out = tf.cast(tf.reshape(self.act_idx_out, [-1, self.item_size, 1]), dtype=tf.int32)
        tmp_idx_range = tf.tile(tf.reshape(tf.range(0, tf.shape(tmp_idx_out)[0]), [-1, 1, 1]),
                                [1, self.item_size, 1])
        tmp_idx_range = tf.cast(tf.concat([tmp_idx_range, tmp_idx_out], axis=2), dtype=tf.int32)
        self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2)
        self.cate_chosen = tf.gather_nd(self.cate_seq, tmp_idx_range)
        self.cate_seq = tf.gather(self.itm_spar_ph, 1, axis=2)
        mask = tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N), tf.int32)  # [B,N]
        self.cate_chosen = self.cate_chosen * mask

    def build_decoder_input(self):
        # decoder_inputs = tf.zeros_like(self.enc_input)
        decoder_inputs = [tf.zeros([self.batch_size, self.enc_input.shape[-1].value])] * (
                self.pv_size + 1)  # [[B,input_size]*(N+1)]
        return decoder_inputs

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
        # self.inference_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        self.inference_sampled_symbol = 1 - tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N),
                                                    tf.float32)  # [B,N]
        self.inference_prediction_order = []
        self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights
            if self.use_masking:
                attention_weights = tf.where(self.inference_sampled_symbol > 0, self.neg_inf,
                                             attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            if self.sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                greedy_result = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
                # sampling_symbol = tf.cond(self.feed_inference_order, lambda: tf.transpose(self.inference_order)[_ - 1, :],
                #                           lambda: sampling_symbol)
            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.inference_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                sampling_symbol_onehot = tf.where(self.inference_sampled_symbol > 0,
                                                  tf.zeros_like(sampling_symbol_onehot),
                                                  sampling_symbol_onehot)  # [B,N]
                # ***** #
                self.inference_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            embedding_matrix = self.enc_input  # [B,N,input_size]
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)
            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def get_training_sampling_function(self):
        # self.training_sampled_symbol = tf.zeros([self.batch_size, self.N])  # [B,N]
        self.training_sampled_symbol = 1 - tf.cast(tf.sequence_mask(self.seq_length_ph, maxlen=self.N),
                                                   tf.float32)  # [B,N]
        self.training_prediction_order = []
        self.neg_inf = tf.ones([self.batch_size, self.N]) * (tf.float32.min)  # [B,N]
        # self.print_loss = tf.print("training_sampled_symbol: ", self.training_sampled_symbol, output_stream=sys.stderr)
        # self.dd, self.ee = [], []

        self.neural_sort_outputs = []

        # 回头写进超参数里面
        self.temperature_factor = 1.0
        self.beta = 1.0

        def sampling_function(attention_weights, _):
            attention_weights = attention_weights
            if self.use_masking:
                attention_weights = tf.where(self.training_sampled_symbol > 0, self.neg_inf, attention_weights)  # [B,N]
            attention_weights = tf.nn.softmax(attention_weights)

            attention_weights = tf.log(attention_weights + 1e-10) + gumbel_sampling(tf.shape(attention_weights), self.beta)    # [B,M]

            if self.training_sample_manner == "greedy":
                # 1、greedy
                sampling_symbol = tf.argmax(attention_weights, 1)  # [B,N] -> [B]
            else:
                # 2、sample
                sampling_symbol = tf.squeeze(tf.multinomial(tf.log(attention_weights), 1), axis=-1)  # [B,N] -> [B]
                sampling_symbol = tf.cond(self.feed_train_order, lambda: tf.transpose(self.train_order)[_ - 1, :],
                                          lambda: sampling_symbol)

            sampling_symbol = tf.cast(sampling_symbol, tf.int32)  # [B]
            self.training_prediction_order.append(sampling_symbol)

            if self.use_masking:
                sampling_symbol_onehot = tf.one_hot(sampling_symbol, self.N)  # [B,N]
                # ***** #
                # self.print_loss = tf.print("sampling_symbol", sampling_symbol,
                #                            "\nsampling_symbol_onehot", sampling_symbol_onehot,
                #                            output_stream=sys.stderr)
                sampling_symbol_onehot = tf.where(self.training_sampled_symbol > 0,
                                                  tf.zeros_like(sampling_symbol_onehot),
                                                  sampling_symbol_onehot)  # [B,N]
                # self.print_loss = tf.print("sampling_symbol", sampling_symbol,
                #                            "\nsampling_symbol_onehot", sampling_symbol_onehot,
                #                            output_stream=sys.stderr)
                # ***** #
                self.training_sampled_symbol += sampling_symbol_onehot  # [B,N]

                sampling_symbol_score = (self.pv_size - _ + 1) * 0.1 * sampling_symbol_onehot  # [B,N]
            # self.dd.append([tf.transpose(self.train_order)[_ - 1, :], sampling_symbol_onehot])
            # self.ee.append(sampling_symbol_score)
            # self.print_loss = tf.print("sampling_symbol_score: ", sampling_symbol_score,
            #                            "\nsampling_symbol: ", sampling_symbol_onehot,
            #                            "\nmask", self.training_sampled_symbol,
            #                            output_stream=sys.stderr)
            embedding_matrix = self.enc_input
            sampling_symbol_embedding = tf.gather_nd(params=embedding_matrix, indices=self.symbol_to_index_pair(
                sampling_symbol))  # [B,N,input_size]->[B,input_size] or [B,N,state_size]->[B,state_size]
            sampling_symbol_embedding = tf.stop_gradient(sampling_symbol_embedding)

            neural_sort_outputs = neural_sort(self.raw_evaluator_input, attention_weights, self.temperature_factor)
            self.neural_sort_outputs.append(neural_sort_outputs[:, 0, :])

            return sampling_symbol_embedding, sampling_symbol_score

        return sampling_function

    def _build_graph(self):
        self.lstm_hidden_units = 32

        with tf.variable_scope("input"):
            self.train_phase = self.is_train
            self.sample_phase = tf.placeholder(tf.bool, name="sample_phase")  # True
            self.only_evaluator = tf.placeholder(tf.bool, name="only_evaluator")  # True
            self.mask_in_raw = tf.placeholder(tf.float32, [None])
            self.div_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            self.auc_label = tf.placeholder(tf.float32, [None, self.max_time_len])
            # self.idx_out_act = tf.placeholder(tf.int32, [None, self.max_time_len])
            self.item_input = self.item_seq
            self.item_label = self.label_ph  # [B, N]
            item_features = self.item_input

            self.item_size = self.max_time_len
            self.mask_in = tf.reshape(self.mask_in_raw, [-1, self.item_size])  # [B*N, N]

            self.itm_enc_input = tf.reshape(item_features, [-1, self.item_size, self.ft_num])  # [B, N, ft_num]
            self.usr_enc_input = tf.reshape(self.usr_seq, [-1, 1, self.profile_num * self.emb_dim])
            self.full_item_spar_fts = self.itm_spar_ph
            self.full_item_dens_fts = self.itm_dens_ph
            self.pv_item_spar_fts = tf.reshape(self.full_item_spar_fts, (-1, self.full_item_spar_fts.shape[-1]))
            self.pv_item_dens_fts = tf.reshape(self.full_item_dens_fts, (-1, self.full_item_dens_fts.shape[-1]))

            self.raw_dec_spar_input = tf.placeholder(tf.float32, [None, self.itm_spar_num])
            self.raw_dec_dens_input = tf.placeholder(tf.float32, [None, self.itm_dens_num])
            self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
            self.raw_dec_input = tf.concat(
                [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]),
                 self.itm_dens_ph], axis=-1)
            self.dec_input = self.raw_dec_input
            # self.batch_size = tf.shape(self.dec_input)[0]
            self.batch_size = self.dec_input.get_shape()[0].value
            self.N = self.item_size
            self.use_masking = True
            self.training_sample_manner = 'sample'
            self.sample_manner = 'greedy'
            self.pv_size = self.N
            self.attention_head_nums = 2
            self.feed_context_vector = True
            self.feed_train_order = tf.placeholder(tf.bool)
            self.feed_inference_order = tf.placeholder(tf.bool)
            self.generator_name = 'NS_generator'
            self.train_order = tf.placeholder(tf.int64, [None, self.item_size])
            self.inference_order = tf.placeholder(tf.int64, [None, self.item_size])

            # 引入一些关于evaluator的参数
            self.dnn_hidden_units = [512, 256, 128]
            self.all_feature_concatenation_evaluator = None
            self.sum_pooling_layer = None
            self.concatenation_layer = None
            self.multi_head_self_attention_layer = None
            self.rnn_layer = None
            self.pair_wise_comparison_layer = None
            self.evaluator_name = 'NS_evaluator'
            self.label_type = 'zero_one'
            self.feature_batch_norm = True
            self.N = self.item_size = self.pv_size = self.max_time_len
            self.use_BN = True
            self.evaluator_path = '/root/LAST/model/save_model_ad/10/202303091111_lambdaMART_LAST_evaluator_16_0.0005_0.0002_64_16_0.8_1.0'
            self.is_training = tf.placeholder(tf.bool)
            self.batch_size = tf.shape(self.itm_enc_input)[0]
            self.score_format = 'pv'
            self.raw_evaluator_input = self.build_evaluator_input(self.itm_spar_ph, self.itm_dens_ph)

        self.feature_augmentation()

        with tf.variable_scope("encoder"):
            self.deep_set_encode()

        with tf.variable_scope("encoder_state"):
            self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.lstm_hidden_units)

        with tf.variable_scope("decoder"):
            self.rnn_decode()

        # judge of the evaluator's input is actor's (when training actor) or the raw data (when training evaluator)
        self.new_evaluator_input = tf.stack(self.neural_sort_outputs, axis=1)  # [B,N,D]
        self.enc_input_evaluator = tf.cond(self.only_evaluator, lambda: self.raw_evaluator_input,
                                           lambda: self.new_evaluator_input)

        # 应该是这一步之后将使用Neural_Sort排序后的结果传递
        with tf.variable_scope("evaluator"):
            self.build_evaluator()

        with tf.variable_scope("loss"):
            self.build_actor_loss()

        with tf.variable_scope("evaluator_loss"):
            self.build_evaluator_loss()

    def build_actor_loss(self):
        self.loss = tf.reduce_mean(-self.logits_pv)

        self.actor_opt()

    def actor_opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)
        trainable_var = [var for var in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES) if
                         (self.evaluator_name not in var.name and "emb_mtx" not in var.name)]
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=trainable_var)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss, var_list=trainable_var)

    def train(self, batch_data, lr, reg_lambda, keep_prop=0.8, train_prefer=1):
        with self.graph.as_default():
            _, total_loss = \
                self.sess.run(
                    [self.train_step, self.loss],
                    feed_dict={
                        self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                        self.itm_spar_ph: batch_data[2],
                        self.itm_dens_ph: batch_data[3],
                        self.seq_length_ph: batch_data[6],
                        self.reg_lambda: reg_lambda,
                        self.lr: lr,
                        self.keep_prob: keep_prop,
                        self.is_train: True,
                        self.only_evaluator: False,
                        self.feed_train_order: False,
                        self.train_order: np.zeros_like(batch_data[4]),
                        self.controllable_auc_prefer: train_prefer,
                        self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
                    })
            return total_loss

    def eval(self, batch_data, reg_lambda, eval_prefer=0, keep_prob=1, no_print=True):
        with self.graph.as_default():
            rerank_predict = self.sess.run(self.predictions,
                                           feed_dict={
                                               self.usr_profile: np.reshape(np.array(batch_data[1]),
                                                                            [-1, self.profile_num]),
                                               self.itm_spar_ph: batch_data[2],
                                               self.itm_dens_ph: batch_data[3],
                                               self.seq_length_ph: batch_data[6],
                                               self.is_train: False,
                                               self.only_evaluator: False,
                                               self.sample_phase: False,
                                               self.controllable_auc_prefer: eval_prefer,
                                               self.controllable_prefer_vector: [[eval_prefer, 1 - eval_prefer]],
                                               self.keep_prob: 1})
            return rerank_predict, 0

    # 上面是generator部分，下面是拆解后的evaluator部分
    def build_evaluator_loss(self):
        with tf.name_scope("CMR_evaluator_Loss_Op"):
            if self.score_format == 'pv':
                loss_weight = tf.ones([self.batch_size, 1])  # [B,1]
                if self.label_type == "total_num":  # label_ph: [B, N(0 or 1)]
                    loss_weight = tf.reduce_sum(self.label_ph, axis=1)
                    # loss_weight = tf.where(loss_weight > 1, loss_weight, tf.ones_like(loss_weight))  # [B,1]
                one = tf.ones_like(loss_weight, dtype=tf.float32)
                zero = tf.zeros_like(loss_weight, dtype=tf.float32)
                self.pv_pos_loss = tf.losses.log_loss(one, self.logits_pv, loss_weight, reduction="weighted_mean")
                self.pv_neg_loss = tf.losses.log_loss(zero, self.logits_pv, one, reduction="weighted_mean")
                self.evaluator_loss = self.pv_pos_loss + self.pv_neg_loss
                self.loss_weight = loss_weight
                # label = tf.reshape(tf.reduce_sum(self.label_ph, axis=1), [-1, 1])  # [B,1]
                # self.label = label
                # self.print_loss = tf.print("label: ", tf.reshape(label, [1, -1]),
                #                            "\nlogits", tf.reshape(self.logits, [1, -1]),
                #                            summarize=-1, output_stream=sys.stderr)
                # self.raw_evaluator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                #     logits=self.logits_pv_before_sigmoid,
                #     labels=label)
                # self.evaluator_loss = self.raw_evaluator_loss * loss_weight  # [B,1]
                self.evaluator_loss = tf.reduce_mean(self.evaluator_loss)
                self.gap = self.evaluator_loss
            elif self.score_format == 'iv':
                self.evaluator_loss = tf.losses.log_loss(self.label_ph, self.logits)
                self.gap = self.evaluator_loss

        self.evaluator_opt()

    def evaluator_opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.evaluator_loss += self.reg_lambda * tf.nn.l2_loss(v)

        self.evaluator_optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.evaluator_optimizer.compute_gradients(self.evaluator_loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.evaluator_train_step = self.evaluator_optimizer.apply_gradients(grads_and_vars)
        else:
            self.evaluator_train_step = self.evaluator_optimizer.minimize(self.evaluator_loss)

    def dnn_layer(self):
        dnn_layer = [self.sum_pooling_layer, self.concatenation_layer,
                     self.multi_head_self_attention_layer, self.rnn_layer, self.pair_wise_comparison_layer]
        dnn_layer = tf.concat(values=dnn_layer, axis=-1)
        if self.feature_batch_norm:
            with tf.variable_scope(name_or_scope="{}_Input_BatchNorm".format(self.evaluator_name)):
                dnn_layer = tf.contrib.layers.batch_norm(dnn_layer, is_training=self.is_train, scale=True)
        self.dnn_input = dnn_layer
        self.final_neurons = self.get_dnn(self.dnn_input, self.dnn_hidden_units, [tf.nn.relu, tf.nn.relu, tf.nn.relu],
                                          "evaluator_dnn"),

    def build_evaluator(self):
        with tf.variable_scope("evaluator"):
            self.all_feature_concatenation_evaluator = self.enc_input_evaluator

            self.sum_pooling_channel()
            self.concatenation_channel()
            self.multi_head_self_attention_channel()
            self.rnn_channel()
            self.pair_wise_comparison_channel()

            self.dnn_layer()
            self.logits_layer()
            # 之后加载指定路径的evaluator参数
            # self.load_evaluator_params(self.evaluator_path)

    def load_evaluator_params(self, path):
        """
        加载evaluator的预训练参数
        """
        # 创建一个 scope 以包含 self.name
        scope = f"evaluator/{self.evaluator_name}"

        # 通过Saver来恢复evaluator网络层的参数，只加载名字中包含self.name的变量
        variables_to_restore = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                                if self.evaluator_name in var.name]

        # 创建一个Saver，来恢复变量
        self.saver = tf.train.Saver(var_list=variables_to_restore)

        # 初始化所有变量
        self.sess.run(tf.global_variables_initializer())

        # 恢复指定路径的模型参数
        self.saver.restore(self.sess, path)
        print(f"evaluator parameters loaded from {path}")

    def sum_pooling_channel(self):
        with tf.variable_scope(name_or_scope="{}_Sum_Pooling_Channel".format(self.evaluator_name)):
            self.sum_pooling_layer = tf.reduce_sum(self.all_feature_concatenation_evaluator, axis=1)
            # for var in tf.trainable_variables():
            #     if "sum_pooling" in var.name:
            #         var.trainable = False

    def concatenation_channel(self):
        with tf.variable_scope(name_or_scope="{}_Concatenation_Channel".format(self.evaluator_name)) as scope:
            running_layer = layers.fully_connected(
                self.all_feature_concatenation_evaluator,
                16,
                tf.nn.relu,
                scope=scope,
                normalizer_fn=layers.batch_norm if self.use_BN else None,
                normalizer_params={"scale": True, "is_training": self.is_train})
            self.concatenation_layer = tf.reshape(running_layer,
                                                  [-1, self.pv_size * running_layer.get_shape().as_list()[2]])
            # 冻结此层的变量
            # for var in tf.trainable_variables():
            #     if "concatenation_channel" in var.name:
            #         var.trainable = False

    def multi_head_self_attention_channel(self):
        with tf.variable_scope(name_or_scope="{}_Multi_Head_Self_Attention_Channel".format(self.evaluator_name)):
            shape_list = self.all_feature_concatenation_evaluator.get_shape().as_list()
            all_feature_concatenation_evaluator = tf.reshape(self.all_feature_concatenation_evaluator,
                                                             [-1, self.pv_size, shape_list[2]])
            queries = all_feature_concatenation_evaluator
            keys = all_feature_concatenation_evaluator
            mask = tf.cast(tf.ones_like(keys[:, :, 0]), dtype=tf.bool)
            outputs, _ = multihead_attention(queries=queries,
                                             keys=keys,
                                             num_heads=8,
                                             num_units=128,
                                             num_output_units=2 * 128,
                                             activation_fn="relu",
                                             scope="multi_head_att",
                                             atten_mode="ln",
                                             reuse=tf.AUTO_REUSE,
                                             key_masks=mask,
                                             query_masks=mask,
                                             is_target_attention=False)
            self.multi_head_self_attention_layer = tf.reduce_sum(outputs, axis=1)
            # 冻结此层的变量
            # for var in tf.trainable_variables():
            #     if "multi_head_self_attention" in var.name:
            #         var.trainable = False

    def rnn_channel(self):
        with tf.variable_scope(name_or_scope="{}_RNN_Channel".format(self.evaluator_name)):
            # one can reverse self.all_feature_concatenation_evaluator and make it a Bi-GRU
            encoder_cell = tf.nn.rnn_cell.GRUCell(64)
            rnn_inputs = tf.transpose(self.all_feature_concatenation_evaluator, perm=[1, 0, 2])  # [N,B,E]
            rnn_inputs = tf.unstack(rnn_inputs, num=self.pv_size, axis=0)  # [B,E]*N
            outputs, final_state = rnn.static_rnn(encoder_cell, rnn_inputs, dtype=tf.float32)

            output = [tf.reshape(output, [-1, 1, encoder_cell.output_size]) for output in outputs]
            output = tf.concat(axis=1, values=output)
            self.rnn_layer = tf.reduce_sum(output, axis=1)
            # 冻结此层的变量
            # for var in tf.trainable_variables():
            #     if "rnn_channel" in var.name:
            #         var.trainable = False

    def pair_wise_comparison_channel(self):
        with tf.variable_scope(name_or_scope="{}_Pair_Wise_Comparison_Channel".format(self.evaluator_name)):
            input_transposed = tf.transpose(self.all_feature_concatenation_evaluator, perm=[0, 2, 1])
            output = tf.matmul(self.all_feature_concatenation_evaluator, input_transposed)
            self.pair_wise_comparison_layer = tf.reshape(output, [-1, self.pv_size * self.pv_size])
            # 冻结此层的变量
            # for var in tf.trainable_variables():
            #     if "pair_wise_comparison_channel" in var.name:
            #         var.trainable = False

    def get_dnn(self, x, layer_nums, layer_acts, name="NS_evaluator_dnn"):
        input_ft = x
        assert len(layer_nums) == len(layer_acts)
        with tf.variable_scope(name):
            for i, layer_num in enumerate(layer_nums):
                input_ft = tf.contrib.layers.fully_connected(
                    inputs=input_ft,
                    num_outputs=layer_num,
                    scope='layer_%d' % i,
                    activation_fn=layer_acts[i],
                    reuse=tf.AUTO_REUSE,
                    trainable=False)
        return input_ft

    def logits_layer(self, name="NS_evaluator_logits_layer"):
        # with tf.variable_scope(name_or_scope="{}_Logits".format(self.name)) as dnn_logits_scope:
        #     logits = layers.linear(self.final_neurons, 1, scope=dnn_logits_scope)
        with tf.variable_scope(name):
            if self.score_format == 'pv':
                logits = layers.linear(self.final_neurons, 1, trainable=False)
                self.logits_pv_before_sigmoid = tf.reshape(logits, [-1, 1])
                logits = tf.sigmoid(logits)
                predictions = tf.reshape(logits, [-1, 1])  # [B,1]
                self.logits_pv = predictions
                self.logits = tf.tile(self.logits_pv, [1, self.max_time_len])
            elif self.score_format == 'iv':
                logits = layers.linear(self.final_neurons, self.max_time_len, trainable=False)
                logits = tf.reshape(logits, [-1, self.max_time_len])
                logits = tf.nn.softmax(logits)
                seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
                predictions = seq_mask * logits
                self.logits = predictions
        return predictions

    def predict_evaluator(self, usr_ft, item_spar_fts, item_dens_fts, seq_len, batch_data):
        with self.graph.as_default():
            ctr_probs = self.sess.run(self.logits, feed_dict={
                self.usr_profile: np.reshape(usr_ft, [-1, self.profile_num]),
                self.itm_spar_ph: item_spar_fts.reshape([-1, self.max_time_len, self.itm_spar_num]),
                self.itm_dens_ph: item_dens_fts.reshape([-1, self.max_time_len, self.itm_dens_num]),
                self.seq_length_ph: seq_len,
                self.feed_train_order: False,
                self.train_order: np.zeros_like(batch_data[4]),
                self.only_evaluator: True,
                self.is_train: False,
                self.keep_prob: 1.0})
            return ctr_probs

    def train_evaluator(self, batch_data, lr, reg_lambda, keep_prob=0.8, train_prefer=1):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.evaluator_loss, self.evaluator_train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
                self.only_evaluator: True,
                self.feed_train_order: False,
                self.train_order: np.zeros_like(batch_data[4]),
                self.controllable_auc_prefer: train_prefer,
                self.controllable_prefer_vector: [[train_prefer, 1 - train_prefer]],
            })
            return loss

    def eval_evaluator(self, batch_data, reg_lambda, eval_prefer=1, keep_prob=1, no_print=True):
        with self.graph.as_default():
            pred, loss, logit, weight = self.sess.run([self.logits, self.evaluator_loss, self.logits_pv, self.loss_weight], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
                self.only_evaluator: True,
                self.feed_train_order: False,
                self.train_order: np.zeros_like(batch_data[4]),
                self.controllable_auc_prefer: eval_prefer,
                self.controllable_prefer_vector: [[eval_prefer, 1 - eval_prefer]],
            })
            return pred.tolist(), loss
