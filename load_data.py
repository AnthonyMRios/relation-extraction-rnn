import random
import pickle
from time import time
import sys
from collections import defaultdict
import gensim
import logging
from gensim.models.keyedvectors import KeyedVectors
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
from nltk import conlltags2tree, tree2conlltags

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

def dataRead(fname):
    print ("Input File Reading")
    fp = open(fname, 'r')
    #samples = fp.read().strip().split('\r\n\r\n')
    samples = fp.read().strip().split('\n\n')
    sent_lengths   = []        #1-d array
    sent_contents  = []        #2-d array [[w1,w2,....] ...]
    sent_lables    = []        #1-d array
    entity1_list   = []        #2-d array [[e1,e1_t] [e1,e1_t]...]
    entity2_list   = []        #2-d array [[e1,e1_t] [e1,e1_t]...]
    doc_ids = []
    idents = []
    for sample in samples:
        #sent, entities = sample.strip().split('\r\n')
        sent, entities = sample.strip().split('\n')
        doc_id, ident, e1, e2, relation = entities.split('\t') 
        sent_contents.append(sent.lower())
        entity1_list.append([e1, ident])
        entity2_list.append([e2, ident])
        sent_lables.append(relation)
        idents.append(ident)
        doc_ids.append(doc_id)

    return idents, sent_contents, entity1_list, entity2_list, sent_lables 


class LoadDataReturn(object):
    def __init__(self):
        self.pairs_idx = []
        self.pos_idx = []
        self.pairs_idx_rev = []
        self.domain_labels = []
        self.pos_e2_idx = []
        self.pos_e1_idx = []
        self.subj_labels = []
        self.pred_labels = []
        self.obj_labels = []
        self.e1_ids = []
        self.e2_ids = []
        self.y = []
        self.idents = []


class LoadData(object):
    def __init__(self, word2vec_file):
        self.word_index = {}
        self.pos_index = {}
        self.num_words = 1
        self.num_pos = 1
        self.embs = [np.zeros((300,))]
        self.pos  = [np.zeros((32,))]
        logging.debug('Loading %s', word2vec_file)
        #self.wv = gensim.models.Word2Vec.load('/home/amri228/i2b2_2016/ddi/word_vecs2/gensim_model_pubmed')
        self.wv = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
        logging.debug('Done')
        self.max_u = self.wv.syn0.max()
        self.min_u = self.wv.syn0.min()

    def fit(self, filename, ids):
        all_data = dataRead(filename)
        word_cnts = {}
        pos_cnts = {}
        missing = set()
        for ident, tr, tl, e1, e2 in zip(all_data[0], all_data[1], all_data[-1], all_data[2], all_data[3]):
            if ident not in ids:
                continue
            final_string = tr.split()
            final_string_pos = nltk.pos_tag(final_string)
            #tree = self.tagger.parse(final_string_pos)
            #iob_tags = tree2conlltags(tree)
            final_e1_string = ['druga']
            final_e2_string = ['drugb']
            e1_pos = None
            e2_pos = None
            cnt = 0
            for w in final_string:
                if w == 'druga':
                    e1_pos = cnt
                elif w == 'drugb':
                    e2_pos = cnt
                cnt += 1
            tmp = []
            final_e1_pos = []
            final_e2_pos = []
            cnt = 0
            error = False
            #print final_string
            for w in final_string:
                if cnt-e1_pos in pos_cnts:
                    pos_cnts[cnt-e1_pos] += 1
                else:
                    pos_cnts[cnt-e1_pos] = 1
                if cnt-e2_pos in pos_cnts:
                    pos_cnts[cnt-e2_pos] += 1
                else:
                    pos_cnts[cnt-e2_pos] = 1
                cnt += 1
            for w in final_string:
                if w in word_cnts:
                    word_cnts[w] += 1
                else:
                    word_cnts[w] = 1
            for w in final_e1_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1
            for w in final_e2_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1

        for w, cnt in word_cnts.iteritems():
            if cnt > 5:
                if w in self.wv:
                    self.embs.append(self.wv[w])
                    self.word_index[w] = self.num_words
                    self.num_words += 1
                else:
                    missing.add(w)
                    self.embs.append(np.random.uniform(-1., 1., (300,)))
                    self.word_index[w] = self.num_words
                    self.num_words += 1
        for w, cnt in pos_cnts.iteritems():
            if cnt > 5:
                self.pos.append(np.random.uniform(-1., 1., (32,)))
                self.pos_index[w] = self.num_pos
                self.num_pos += 1

        self.pos_index['NegUNK'] = self.num_pos
        self.num_pos += 1
        self.pos.append(np.random.uniform(-1., 1., (32,)))
        self.pos_index['PosUNK'] = self.num_pos
        self.num_pos += 1
        self.pos.append(np.random.uniform(-1., 1., (32,)))

        self.word_index['UNK'] = self.num_words
        self.embs.append(np.random.uniform(-1., 1., (300,)))
        self.num_words += 1

        del self.wv
        self.embs = np.array(self.embs, dtype='float32')
        self.pos = np.array(self.pos, dtype='float32')
        return

    def transform(self, filename, ids):
        all_data = dataRead(filename)
        pairs_idx = []
        pos_idx = []
        pairs_idx_rev = []
        domain_labels = []
        pos_e2_idx = []
        pos_e1_idx = []
        subj_labels = []
        pred_labels = []
        obj_labels = []
        e1_ids = []
        e2_ids = []
        y = []
        idents = []
        for ident, tr, tl, e1, e2 in zip(all_data[0], all_data[1],
                                         all_data[-1], all_data[2], all_data[3]):
            if ident not in ids:
                continue
            final_string = tr.split()
            final_string_pos = nltk.pos_tag(final_string)
            #tree = self.tagger.parse(final_string_pos)
            #iob_tags = tree2conlltags(tree)
            final_e1_string = ['druga']
            final_e2_string = ['drugb']
            e1_pos = None
            e2_pos = None
            cnt = 0
            for w in final_string:
                if w == 'druga':
                    e1_pos = cnt
                elif w == 'drugb':
                    e2_pos = cnt
                cnt += 1
            if e1_pos is None or e2_pos is None:
                continue
            tmp = []
            final_e1_pos = []
            final_e2_pos = []
            cnt = 0
            tmp_subj = []
            tmp_pred = []
            tmp_obj = []
            for w in final_string:
                final_e1_pos.append(cnt - e1_pos)
                final_e2_pos.append(cnt - e2_pos)
                cnt += 1
            idents.append(ident)
            e1_ids.append(e1[0])
            e2_ids.append(e2[0])
            y.append(tl)
            fstring = []
            for w in final_string:
                fstring.append(w)
            final_string = fstring
            str_idx = []
            for w in final_string:
                if w in self.word_index:
                    str_idx.append(self.word_index[w])
                else:
                    str_idx.append(self.word_index['UNK'])
            pairs_idx.append(str_idx)
            e1_idx = []
            for p in final_e1_pos:
                if p in self.pos_index:
                    e1_idx.append(self.pos_index[p])
                else:
                    if p < 0:
                        e1_idx.append(self.pos_index['NegUNK'])
                    else:
                        e1_idx.append(self.pos_index['PosUNK'])
            pos_e1_idx.append(e1_idx)
            e2_idx = []
            for p in final_e2_pos:
                if p in self.pos_index:
                    e2_idx.append(self.pos_index[p])
                else:
                    if p < 0:
                        e2_idx.append(self.pos_index['NegUNK'])
                    else:
                        e2_idx.append(self.pos_index['PosUNK'])
            pos_e2_idx.append(e2_idx)

        lab_lookup = {'OTHER':0, 'CLASS1':1}
        self.lab_lookup_rev = {0:'OTHER', 1:'CLASS1'}
        final_y = np.array([np.int32(lab_lookup[x]) for x in y])

        return pairs_idx, pos_e1_idx, pos_e2_idx, final_y, subj_labels, pred_labels, obj_labels, idents, e1_ids, e2_ids

    def fit_transform(self, filename, ids):
        self.fit(filename, ids)
        return self.transform(filename, ids)

    def pad_data(self, data, max_len = None):
        max_len = np.max([len(x) for x in data])
        padded_dataset = []
        for example in data:
            try:
                zeros = [0]*(max_len-len(example))
                padded_dataset.append(example+zeros)
            except:
                logging.exception('%s %s %s', max_len, len(example), example)
                exit(1)
        if max_len is None:
            return np.array(padded_dataset)
        else:
            return np.array(padded_dataset)[:,:max_len]
