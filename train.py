"""
Usage:
  train.py [options]

Options:
  -h --help                    show this help message and exit
  --word2vec=<file>            word vectors in gensim format
  --dataset=<file>             dataset (see data folder for example)
  --train_ids=<file>           ids of training examples (see data folder for example)
  --dev_ids=<file>             ids of dev exapmles (see data folder for example)
  --model=<file>               filename to use to save model 
  --num_epochs=<arg>           Max number of epochs [default: 25]
  --mini_batch_size=<arg>      Minibatch size [default: 32]
  --num_classes=<arg>          Total number of classes for training [default: 5]
  --lstm_hidden_state=<arg>    lstm hidden state size [default: 256]
  --random_seed=<arg>          random seed [default: 42]

"""

import random
import sys
import logging

import docopt
import numpy as np
from sklearn.metrics import f1_score

from models.bilstm import BiLSTM
from load_data import LoadData
import pickle


def main(argv):
    argv = docopt.docopt(__doc__)

    num_epochs = int(argv['--num_epochs'])
    mini_batch_size = int(argv['--mini_batch_size'])
    val_mini_batch_size = 64
    num_classes = int(argv['--num_classes'])
    lstm_hidden_state_size = int(argv['--lstm_hidden_state'])
    random_seed = argv['--random_seed']

    np.random.seed(int(random_seed))
    random.seed(int(random_seed))

    def read_ids(filename):
        ids = []
        with open(filename, 'r') as fp:
            for row in fp:
                ids.append(row.strip())
        return ids

    train_ids = read_ids(argv['--train_ids'])
    val_ids = read_ids(argv['--dev_ids'])

    ld = LoadData(argv['--word2vec'])

    train_pairs, train_e1, train_e2, train_y, _, _, _, train_ids, _, _  = ld.fit_transform(argv['--dataset'], train_ids)
    dev_pairs, dev_e1, dev_e2, dev_y, _, _, _, val_ids, dev_e1_ids,dev_e2_ids  = ld.transform(argv['--dataset'], val_ids)

    idxs = list(range(len(train_pairs)))
    dev_idxs = list(range(len(dev_pairs)))

    last_loss = None
    avg_loss = []
    avg_f1 = []
    check_preds = None
    mod = BiLSTM(ld.embs, ld.pos, nc=int(num_classes), nh=int(lstm_hidden_state_size), de=ld.embs.shape[1])
    best_dev_f1 = 0
    for epoch in range(1, int(num_epochs)+1):
        mean_loss = []
        random.shuffle(idxs)
        for start, end in zip(range(0, len(idxs), mini_batch_size), range(mini_batch_size, len(idxs)+mini_batch_size,
                mini_batch_size)):
            idxs_sample = idxs[start:end]
            batch_labels = np.array(train_y[idxs_sample], dtype='int32')
            tpairs = ld.pad_data([train_pairs[i] for i in idxs_sample])
            te1 = ld.pad_data([train_e1[i] for i in idxs_sample])
            te2 = ld.pad_data([train_e2[i] for i in idxs_sample])
            cost = mod.train_batch(tpairs, te1, te2, train_y[idxs_sample].astype('int32'), np.float32(0.))
            mean_loss.append(cost)
            print("EPOCH: %d loss: %.4f train_loss: %.4f" % (epoch, cost, np.mean(mean_loss)))
            sys.stdout.flush()

        all_dev_preds = []
        scores = []
        for start, end in zip(range(0, len(dev_idxs), val_mini_batch_size), range(val_mini_batch_size, len(dev_idxs)+val_mini_batch_size,
                    val_mini_batch_size)):
            if len(dev_idxs[start:end]) == 0:
                continue
            vpairs = ld.pad_data([dev_pairs[i] for i in dev_idxs[start:end]])
            ve1 = ld.pad_data([dev_e1[i] for i in dev_idxs[start:end]])
            ve2 = ld.pad_data([dev_e2[i] for i in dev_idxs[start:end]])
            preds = mod.predict_proba(vpairs, ve1, ve2, np.float32(1.))
            for x in preds:
                all_dev_preds.append(x.argmax())

        dev_f1 = f1_score(dev_y, all_dev_preds, average='micro')
        print("EPOCH: %d train_loss: %.4f dev_f1: %.4f" % (epoch, np.mean(mean_loss), dev_f1))
        sys.stdout.flush()

        if dev_f1 > best_dev_f1:
            with open(argv['--model'], 'w') as fp:
                pickle.dump({'model_params':mod.__getstate__(), 'token':ld}, fp, pickle.HIGHEST_PROTOCOL)
            best_dev_f1 = dev_f1

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
