"""
Usage:
    pred.py [options]

Options:
  -h --help                    show this help message and exit
  --word2vec=<file>            word vectors in gensim format
  --dataset=<file>             dataset (see data folder for example)
  --test_ids=<file>            ids of test examples (see data folder for example)
  --model=<file>               filename to use to save model 
  --mini_batch_size=<arg>      Minibatch size [default: 32]
  --num_classes=<arg>          Total number of classes for training [default: 5]
  --lstm_hidden_state=<arg>    lstm hidden state size [default: 256]
  --random_seed=<arg>          random seed [default: 42]

"""

import logging
import pickle
import random
import sys
from models.bilstm import BiLSTM
import docopt
import numpy as np
from sklearn.metrics import f1_score


def main(argv):
    argv = docopt.docopt(__doc__, argv=argv)

    random_seed = argv['--random_seed']
    np.random.seed(random_seed)
    random.seed(random_seed)

    mini_batch_size = argv['--mini_batch_size']

    def read_ids(file):
        ids = []
        with open(file, 'r') as fp:
            for row in fp:
                ids.append(row.strip())
        return ids

    test_ids = read_ids(argv['<test_ids>'])

    with open(argv['--model']) as fp:
        tmp = pickle.load(fp)

    ld = tmp['token']
    mod = BiLSTM(ld.embs, ld.pos, ld.pospeech, ld.chunk, nc=5, nh=2048, de=ld.embs.shape[1])
    mod.__setstate__(tmp['model_params'])

    pairs_idx, pos_e1_idx, pos_e2_idx, y, _, _, _, _, _, _  = ld.transform(argv['--dataset'], test_ids)

    test_idxs = list(range(len(pairs_idx)))

    all_test_preds = []
    scores = []
    for start, end in zip(range(0, len(test_idxs), mini_batch_size),
                          range(mini_batch_size, len(test_idxs) + mini_batch_size,
                                mini_batch_size)):
        if len(test_idxs[start:end]) == 0:
            continue
        tpairs = ld.pad_data([pairs_idx[i] for i in test_idxs[start:end]])
        te1 = ld.pad_data([pos_e1_idx[i] for i in test_idxs[start:end]])
        te2 = ld.pad_data([pos_e2_idx[i] for i in test_idxs[start:end]])
        preds = mod.predict_proba(tpairs, te1, te2, np.float32(1.))

        for x in preds:
            all_test_preds.append(x.argmax())

    test_f1 = f1_score(y, all_test_preds, average='micro')
    print("test_f1: %.4f" % (test_f1))
    sys.stdout.flush()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
