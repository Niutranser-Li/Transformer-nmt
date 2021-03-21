import os, sys
import numpy as np
import tensorflow as tf

class Transformer(tf.keras.Model):
    def __init__(self, inputs_vocab_size, target_vocab_size, encoder_count, decoder_count, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(Transformer, self).__init__()

        # Model hyper parameter variables
        self.inputs_vocab_size = inputs_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoder_count = encoder_count
        self.decoder_count = decoder_count
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.encoder_embedding_layer = EmbeddingLayer(self.inputs_vocab_size, self.d_model)
        self.encoder_embedding_dropout = tf.keras.layers.Dropout(self.dropout_prob)
        self.decoder_embedding_layer = EmbeddingLayer(self.outputs_vocab_size, self.d_model)
        self.decoder_embedding_dropout = tf.keras.layers.Dropout(self.dropout_prob)

        self.encoder_layers = [
            EncoderLayer(
                self.attention_head_count, self.d_model, self.d_point_wise_ff, self.dropout_prob
            ) for _ in range(self.encoder_count)
        ]

        self.decoder_layers = [
            DecoderLayer(
                self.attention_head_count, self.d_model, self.d_point_wise_ff, self.dropout_prob
            ) for _ in range(self.decoder_count)
        ]

        self.linear = tf.keras.layers.Dense(self.target_vocab_size)
    
    def call(self, inputs, target, inputs_padding_mask, look_ahead_mask, target_padding_mask, training):
        encoder_tensor = self.encoder_embedding_layer(inputs)
        encoder_tensor = self.encoder_embedding_dropout(encoder_tensor, training=training)

        for encoder_layer_idx in range(self.encoder_count):
            encoder_tensor, _ = self.encoder_layers[encoder_layer_idx](encoder_tensor, inputs_padding_mask, training)
        
        target = self.decoder_embedding_layer(target)
        decoder_tensor = self.decoder_embedding_dropout(target, training=training)

        for decoder_layer_idx in range(self.decoder_count):
            decoder_tensor, _, _ = self.decoder_layers[decoder_layer_idx](
                decoder_tensor, encoder_tensor, look_ahead_mask, target_padding_mask, training=training
            )
        
        return self.linear(decoder_tensor)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(EncoderLayer, self).__init__()
        
        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttention(self.attention_head_count, self.d_model)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwordLayer(
            self.d_point_wise_ff, self.d_model
        )
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, mask, training):
        output, attention = self.multi_head_attention(inputs, inputs, inputs, mask)
        output = self.dropout_1(output, training=training)
        output = self.layer_norm_1(tf.add(inputs, output))

        output = self.position_wise_feed_forward_layer(output)
        output = self.dropout_2(output, training=training)
        output = self.layer_norm_2(tf.add(inputs, output))

        return output, attention

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model, d_point_wise_ff, dropout_prob):
        super(DecoderLayer, self).__init__()

        self.attention_head_count = attention_head_count
        self.d_model = d_model
        self.d_point_wise_ff = d_point_wise_ff
        self.dropout_prob = dropout_prob

        self.masked_multi_head_attention = MultiHeadAttention(self.attention_head_count, self.d_model)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_decoder_attention = MultiHeadAttention(self.attention_head_count, self.d_model)
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.position_wise_feed_forward_layer = PositionWiseFeedForwordLayer(
            self.d_point_wise_ff, self.d_model
        )
        self.dropout_3 = tf.keras.layers.Dropout(self.dropout_prob)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, decoder_inputs, encoder_output, look_ahead_mask, padding_mask, training):
        output, attention_1 = self.masked_multi_head_attention(
            decoder_inputs, decoder_inputs, decoder_inputs, look_ahead_mask
        )
        output = self.dropout_1(output, training=training)
        query = self.layer_norm_1(tf.add(decoder_inputs, output))

        output, attention_2 = self.encoder_decoder_attention(
            query, encoder_output, encoder_output, padding_mask
        )
        output = self.dropout_2(output, training=training)
        encoder_decoder_attention_output = self.layer_norm_2(tf.add(output, query))

        output = self.position_wise_feed_forward_layer(encoder_decoder_attention_output)
        output = self.dropout_3(output, training=training)
        output = self.layer_norm_3(tf.add(encoder_decoder_attention_output, output))

        return output, attention_1, attention_2

class PositionWiseFeedForwordLayer(tf.keras.layers.Layer):
    def __init__(self, d_point_wise_ff, d_model):
        super(PositionWiseFeedForwordLayer, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_point_wise_ff)
        self.w_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, inputs):
        inputs = self.w_1(inputs)
        inputs = tf.nn.relu(inputs)
        return self.w_2(inputs)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, attention_head_count, d_model):
        super(MultiHeadAttention, self).__init__()

        self.attention_head_count = attention_head_count
        self.d_model = d_model

        if not (self.d_model%self.attention_head_count):
            raise ValueError(
                "d_model({}) % attention_head_count({}) is not zero. d_model must be multiple of attention_head_count.".format(
                    self.d_model, self.attention_head_count
                )
            )
        
        self.d_h = self.d_model // self.attention_head_count

        self.w_query = tf.keras.layers.Dense(self.d_model)
        self.w_key = tf.keras.layers.Dense(self.d_model)
        self.w_value = tf.keras.layers.Dense(self.d_model)

        self.scaled_dot_product = ScaledDotProductAttention(self.d_h)
        self.ff = tf.keras.layers.Dense(self.d_model)
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        output, attention = self.scaled_dot_product(query, key, value, mask)
        output = self.concat_head(output, batch_size)

        return self.ff(output), attention

    def split_head(self, tensor, batch_size):
        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, self.d_h)
            ),
            [0, 2, 1, 3]
        )
    
    def concat_head(self, tensor, batch_size):
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count*self.d_h)
        )

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h
    
    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)
        
        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)
        return tf.matmul(attention_weight, value), attention_weight

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
    
    def call(self, sequences):
        max_sequence_len = sequences.shape[1]
        output = self.embedding(sequences) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        output += self.position_encoding(max_sequence_len)
        return output

    def position_encoding(self, max_len):
        pos = np.expand_dims(np.array(0, max_len), axis=1)
        index = np.expand_dims(np.array(0, self.d_model), axis=0)

        pe = self.angle(pos, index)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float32)

    def angle(self, pos, index):
        return pos / np.power(10000, (index - index%2)/np.float32(self.d_model))
