from theano.tensor.shared_randomstreams import RandomStreams

srng2 = RandomStreams(seed=234)

from .utils import *


class BiLSTM(object):
    def __init__(self, emb, pos, nh=256, nc=2, de=100, p_drop=0.5):
        """
        Args:
            emb: Embedding Matrix
            pos: position matrix
            nh: hidden layer size
            nc: Number of classes
            # de: Dimensionality of word embeddings
            p_drop :: Dropout probability
        """

        def recurrence(xi, mask, h_tm1, c_tm1,
                       W_i, U_i, b_i, W_c, U_c, b_c, W_f, U_f, b_f, W_o2, U_o, b_o2,
                       mask_in, mask_rec):
            x = xi * T.neq(mask, 0).dimshuffle(0, 'x')
            x = dropout_scan(x, mask_in, dropout_switch, 0.2)

            x_i = T.dot(x, W_i) + b_i
            x_i = x_i * T.neq(mask, 0).dimshuffle(0, 'x')

            x_f = T.dot(x, W_f) + b_f
            x_f = x_f * T.neq(mask, 0).dimshuffle(0, 'x')

            x_c = T.dot(x, W_c) + b_c
            x_c = x_c * T.neq(mask, 0).dimshuffle(0, 'x')

            x_o = T.dot(x, W_o2) + b_o2
            x_o = x_o * T.neq(mask, 0).dimshuffle(0, 'x')

            h_tm1 = h_tm1 * T.neq(mask, 0).dimshuffle(0, 'x')
            h_tm1 = dropout_scan(h_tm1, mask_rec, dropout_switch, 0.2)

            i = hard_sigmoid(x_i + T.dot(h_tm1, U_i))
            f = hard_sigmoid(x_f + T.dot(h_tm1, U_f))
            c = f * c_tm1 + i * T.tanh(x_c + T.dot(h_tm1, U_c))
            o = hard_sigmoid(x_o + T.dot(h_tm1, U_o))
            h = o * T.tanh(c)
            return [h, c]

        # Source Embeddings
        self.emb = theano.shared(name='Words', value=emb.astype('float32'))

        self.pos = theano.shared(name='Pos', value=pos.astype('float32'))

        # Source Output Weights
        self.w_o = theano.shared(name='w_o', value=he_normal((nh + nh, nc)).astype('float32'))
        self.b_o = theano.shared(name='b_o', value=np.zeros((nc,)).astype('float32'))

        # input
        idxs = T.matrix()
        e1_pos_idxs = T.matrix()
        e2_pos_idxs = T.matrix()
        Y = T.ivector()
        dropout_switch = T.scalar()

        # get word embeddings based on indicies
        x_word = self.emb[T.cast(idxs, 'int32')]
        x_e1_pos = self.pos[T.cast(e1_pos_idxs, 'int32')]
        x_e2_pos = self.pos[T.cast(e2_pos_idxs, 'int32')]
        x_word = T.concatenate([x_word, x_e1_pos, x_e2_pos], axis=2)
        mask = T.neq(idxs, 0) * 1
        x_word = x_word * mask.dimshuffle(0, 1, 'x')

        de = emb.shape[1] + 2 * pos.shape[1]

        fwd_params, bck_params = bilstm_weights(de, nh)

        # Update these parameters
        self.params = [self.w_o, self.b_o, self.emb, self.pos]
        self.params += fwd_params + bck_params

        self.h0 = theano.shared(name='h0', value=np.zeros((nh,), dtype="float32"))

        maskd1 = srng.binomial((x_word.shape[0], x_word.shape[-1]), p=0.8, dtype='float32')
        maskd2 = srng.binomial((x_word.shape[0], nh), p=0.8, dtype='float32')
        [h_fwd, _], u = theano.scan(fn=recurrence,
                                     sequences=[x_word.dimshuffle(1, 0, 2), idxs.dimshuffle(1, 0)],
                                     non_sequences=fwd_params + [maskd1, maskd2],
                                     outputs_info=[T.alloc(self.h0, x_word.shape[0], nh),
                                                   T.alloc(self.h0, x_word.shape[0], nh)],
                                     n_steps=x_word.shape[1],
                                     strict=True)

        maskd3 = srng.binomial((x_word.shape[0], x_word.shape[-1]), p=0.8, dtype='float32')
        maskd4 = srng.binomial((x_word.shape[0], nh), p=0.8, dtype='float32')
        [h_bck, _], u = theano.scan(fn=recurrence,
                                    sequences=[x_word.dimshuffle(1, 0, 2)[::-1, :, :], idxs.dimshuffle(1, 0)[::-1, :]],
                                    non_sequences=bck_params + [maskd3, maskd4],
                                    outputs_info=[T.alloc(self.h0, x_word.shape[0], nh),
                                                  T.alloc(self.h0, x_word.shape[0], nh)],
                                    n_steps=x_word.shape[1],
                                    strict=True)

        h_bck = h_bck[::-1, :, :].dimshuffle(1, 0, 2)
        h_fwd = h_fwd.dimshuffle(1, 0, 2)
        h_priv = T.concatenate([h_fwd, h_bck], axis=2)
        h = h_priv.max(axis=1)
        h = dropout(h, dropout_switch, 0.2)

        Y_neg = T.ivector()
        pyx = T.nnet.nnet.softmax(T.dot(h, self.w_o) + self.b_o.dimshuffle('x', 0))
        pyx = T.clip(pyx, 1e-5, 1 - 1e-5)
        L = -T.mean(T.log(pyx)[T.arange(Y.shape[0]), Y]) + 1e-6 * sum([(x ** 2).sum() for x in self.params])

        updates, _ = Adam(L, self.params, lr2=0.001)

        self.train_batch = theano.function([idxs, e1_pos_idxs, e2_pos_idxs, \
                                            Y, dropout_switch],
                                           L, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
        self.predict_proba = theano.function([idxs, e1_pos_idxs, e2_pos_idxs, dropout_switch], \
                                             pyx, allow_input_downcast=True, on_unused_input='ignore')

    def __getstate__(self):
        values = [x.get_value() for x in self.params]
        return values

    def __setstate__(self, weights):
        for x, w in zip(self.params, weights):
            x.set_value(w)


def bilstm_weights(de, nh):
    """

    Args:
        de: Dimensionality of word embeddings
        nh: Hidden layer dimensionality

    Returns:
        forward weights, backward weights
    """
    # forward Bi-LSTM Weights
    Wf_i = theano.shared(name='wf_i', value=he_normal((de, nh)).astype("float32"))
    Uf_i = theano.shared(name='uf_i', value=he_normal((nh, nh)).astype("float32"))
    bf_i = theano.shared(name='bf_i', value=np.zeros((nh,), dtype="float32"))

    Wf_f = theano.shared(name='wf_f', value=he_normal((de, nh)).astype("float32"))
    Uf_f = theano.shared(name='uf_f', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bf_f = theano.shared(name='bf_f', value=np.ones((nh,), dtype="float32"))

    Wf_c = theano.shared(name='wf_c', value=he_normal((de, nh)).astype("float32"))
    Uf_c = theano.shared(name='uf_c', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bf_c = theano.shared(name='bf_c', value=np.zeros((nh), dtype="float32"))

    Wf_o2 = theano.shared(name='wfoo', value=he_normal((de, nh)).astype("float32"))
    Uf_o = theano.shared(name='ufoo', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bf_o2 = theano.shared(name='bfoo', value=np.zeros((nh,), dtype="float32"))

    # backward Bi-LSTM Weights
    Wb_i = theano.shared(name='wb_i', value=he_normal((de, nh)).astype("float32"))
    Ub_i = theano.shared(name='ub_i', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bb_i = theano.shared(name='bb_i', value=np.zeros((nh,), dtype="float32"))

    Wb_f = theano.shared(name='wb_f', value=he_normal((de, nh)).astype("float32"))
    Ub_f = theano.shared(name='ub_f', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bb_f = theano.shared(name='bb_f', value=np.ones((nh), dtype="float32"))

    Wb_c = theano.shared(name='wb_c', value=he_normal((de, nh)).astype("float32"))
    Ub_c = theano.shared(name='ub_c', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bb_c = theano.shared(name='bb_c', value=np.zeros((nh), dtype="float32"))

    Wb_o2 = theano.shared(name='wboo', value=he_normal((de, nh)).astype("float32"))
    Ub_o = theano.shared(name='uboo', value=orthogonal_tmp((nh, nh)).astype("float32"))
    bb_o2 = theano.shared(name='bboo', value=np.zeros((nh), dtype="float32"))

    params_forward = [Wb_i, Ub_i, bb_i,
                      Wb_c, Ub_c, bb_c,
                      Wb_f, Ub_f, bb_f,
                      Wb_o2, Ub_o, bb_o2]

    params_backward = [Wf_i, Uf_i, bf_i,
                       Wf_c, Uf_c, bf_c,
                       Wf_f, Uf_f, bf_f,
                       Wf_o2, Uf_o, bf_o2]

    return params_forward, params_backward
