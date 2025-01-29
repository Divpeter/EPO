# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
if '1.4' in tf.__version__:
    from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
else:
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell as core_rnn_cell_impl

def attention_based_decoder(decoder_inputs, initial_state, attention_states, cell, sampling_function,
                    attention_head_nums=1, feed_context_vector=True, dtype=dtypes.float32, scope=None):

    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if attention_head_nums < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    # TODO: recover
    # if not attention_states.get_shape()[1:2].is_fully_defined():
    #     raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
    #                      % attention_states.get_shape())

    with vs.variable_scope(scope or "point_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        input_size = decoder_inputs[0].get_shape()[1].value # input_size or state_size
        # TODO: recover
        # attn_length = attention_states.get_shape()[1].value # N+1 or N
        attn_length = tf.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value   # state_size——rnn output size

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        # [B,N+1,1,state_size] or [B,N,1,state_size]——encoder outpus
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])

        attention_vec_size = attn_size  # state_size——Size of "query" vectors for attention.
        # size is CORRECT! Because both W1 and W2 are square matrix
        hidden_features = []
        v = []
        for a in xrange(attention_head_nums):
            k = vs.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])  # [1,1,state_size,state_size]
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))  # transformation of encoder outputs, <BOS> in the front of encoder_outputs——W1ej in paper
            v.append(
                vs.get_variable("AttnV_%d" % a, [attention_vec_size]))  # [state_size]

        states = [initial_state]    # list of all N's decoder state——[B,state_size], may be length of "N+1"

        def attention(query):   # query——[B,state_size], new state produced by decoder in current N
            """Point on hidden using hidden_features(W1ej)——[B,N+1,1,state_size] or [B,N,1,state_size] and query(decoder state)."""
            attention_weights = []  # Results of attention reads will be stored here.
            context_vector_list = []
            for a in xrange(attention_head_nums):
                with vs.variable_scope("Attention_%d" % a):
                    y = core_rnn_cell_impl._linear(query, attention_vec_size, True) # [B,state_size], W2di in paper, linear transform, same shape as decoder state and encoder states
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])    # [B,1,1,state_size]
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) # [B,N+1,1,state_size]->[B,N+1] or [B,N,1,state_size]->[B,N]
                    # a = tf.nn.softmax(s)
                    attention_weights.append(s)

                    context_vector = math_ops.reduce_sum(array_ops.reshape(s, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    context_vector = array_ops.reshape(context_vector, [-1, attn_size])
                    context_vector_list.append(context_vector)
            return attention_weights, context_vector_list

        outputs = []    # outputs: list of [B,N+1] or [B,N], may be length of "N+1", attention weight(ACTUAL OUTPUT) of each N's decoder state and all encoder output
        prev = None     # sampled vector
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in xrange(attention_head_nums)
        ]   # [B,state_size]

        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        inps = []   # list of [B,input_size], decoder inputs, may be length of "N"(except for the first N) or "0"(it depends on "feed_prev")
        prediction_score = tf.zeros([batch_size, attn_length])  # [B,N]
        for i, inp in enumerate(decoder_inputs):    # [(N+1)*[B,input_size]], input into decoder N by N
            if i > 0:
                vs.get_variable_scope().reuse_variables()

            # If sampling_function is set, we use it instead of decoder_inputs.
            # 如果不指定sampling_function，即自己指定好decoder每一步的输入，这个文件返回的第一个结果即原始的attention分数是能用的，但返回的第三个结果即rtp用的预测分是不能用的
            if sampling_function is not None and prev is not None:
                # TODO:reuse=True
                with vs.variable_scope("sampling_function", reuse=tf.AUTO_REUSE):
                    # 对原始的attention分数（即prev）做mask和softmax的逻辑都是放到sampling_function里面的
                    inp, sampling_symbol_score = sampling_function(prev, i)
                    inps.append(inp)
                    prediction_score += sampling_symbol_score   # [B,N]

            # feed_prev=false——training(it also can be true in training? because it will be same with prediction), feed_prev=true——predict
            # if feed_prev and i > 0:
            #     inp = tf.stack(decoder_inputs[1:])
            #     inp = tf.transpose(inp, perm=[1, 0, 2])
            #     inp = tf.reshape(inp, [-1, attn_length, input_size])    # [B,N+1,input_size] or [B,N,input_size]
            #     # TODO: here should not be inp? should be cell_output of last N? or it also may be make sense?(attention-based decode)
            #     # output——[B,N+1] or [B,N]: attetnion weight of last N's decoder state and all encoder output
            #     inp = tf.reduce_sum(inp * tf.reshape(output, [-1, attn_length, 1]), 1) # [B,input_size], weighted sum of inputs depends on last N's decoder state and all encoder output
            #     inp = tf.stop_gradient(inp)
            #     inps.append(inp)

            # Use the same inputs in inference, order internaly

            # Merge input and previous attentions into one vector of the right size.
            # projection dimension should be cell.input_size(input_size), but not cell.output_size (because it should be same with encoder)
            x = inp
            if feed_context_vector:
                x = core_rnn_cell_impl._linear([inp]+attns, input_size, True)    # [B,input_size]——union of input(origin decoder input[B,input_size] or weighted sum of decoder input[B,input_size]) and attns, finally, decoder input of each N is [B,input_size] nor [B,state_size]
            # Run the RNN.
            cell_output, new_state = cell(x, states[-1])    # [B,state_size], [B,state_size]
            states.append(new_state)
            # Run the attention mechanism.
            # TODO: attns should be weighted-sum of attention_states depends on new_state
            # TODO: and output should be linear combination of cell_output and new attns, and prev should be set as output, and then generate new inp?(if sampling_function is not none)
            output, attns = attention(new_state)   # ([B,N+1] or [B,N]) * attention_head_nums, attention information of new decoder state and all encoder output
            output = tf.stack(output,axis=1)    # [B,attention_head_nums,N]
            output = tf.reduce_mean(output, axis=1) # [B,N]

            if sampling_function is not None:
                prev = output
            # The output of the pointer network is actually the attention weight!
            # 这里的output就是每一步产出的原始的attention分数，没过softmax，也没过mask
            outputs.append(output)

    return outputs, states, prediction_score
