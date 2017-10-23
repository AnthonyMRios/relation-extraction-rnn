# Bi-LSTM Relation Extraction Model

Implementation of a word-level bi-lstm relation extraction model (Kavuluru et al., 2017).

## Required Packages
- Python 2.7
- numpy 1.11.1+
- scipy 0.18.0+
- Theano
- gensim
- sklearn
- docopt
- nltk

## Usage

### Data Format

We use a custom data format as input to our model. Specifically, each example consists of two lines. The first line represents the sentences and the two entities must be marked as DRUGA or DRUGB, respectively. We use the DRUGA and DRUGB convention because our work focused on extracting drug-drug interactions. These entity markers must be used because they are used to find the positive vectors for each word in the sentence relative to each entity. The second line should contain the sentence id, document id, DRUGA id, DRUGB id, and the associated class for that instance. Each id should be separated by a tab.

```
Sentence start DRUGA sentence middle DRUGB sentence end .
sentence_id\tdoc_id\tdruga_id\tdrugb_id\tclass

Sentence start DRUGA sentence middle DRUGB sentence end .
sentence_id\tdoc_id\tdruga_id\tdrugb_id\tclass
```

Example data is available in the data folder.

**Note**: Depending on the classes in your dataset, lines 249 and 250 in load_data.py must be changes to include them.

### Training

```
Usage:
  train.py [options]

Options:
  -h --help                    show this help message and exit
  --word2vec=<file>            word vectors in gensim format
  --dataset=<file>             dataset (see data folder for example)
  --train_ids=<file>           ids of training examples (see data folder for example)
  --dev_ids=<file>             ids of dev exapmles (see data folder for example)
  --test_ids=<file>            ids of test examples (see data folder for example)
  --model=<file>               filename to use to save model
  --num_epochs=<arg>            Max number of epochs [default: 25]
  --mini_batch_size=<arg>       Minibatch size [default: 32]
  --num_classes=<arg>           Total number of classes for training [default: 5]
  --lstm_hidden_state=<arg>     lstm hidden state size [default: 256]
  --random_seed=<arg>           random seed [default: 42]
```

### Testing

*Note*: The current test code is mainly for evaluation purposes

```
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
```

## Acknowledgements

> Ramakanth Kavuluru, Anthony Rios, and Tung Tran. "[Extracting Drug-Drug Interactions with Word and Character-Level Recurrent Neural Networks.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5639883/)" In Healthcare Informatics (ICHI), 2017 IEEE International Conference on, pp. 5-12. IEEE, 2017.

```
@inproceedings{kavuluru2017extracting,
  title={Extracting Drug-Drug Interactions with Word and Character-Level Recurrent Neural Networks},
  author={Kavuluru, Ramakanth and Rios, Anthony and Tran, Tung},
  booktitle={Healthcare Informatics (ICHI), 2017 IEEE International Conference on},
  pages={5--12},
  year={2017},
  organization={IEEE}
}
```

Written by Anthony Rios (anthonymrios at gmail dot com)
