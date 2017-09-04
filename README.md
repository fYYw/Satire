# Satirical News Detection and Analysis

Code for the paper

**Satirical News Detection and Analysis using Attention Mechanism and Linguisitc Features** 

## Dataset

Currently, the dataset is hosted at [Mendeley](http://www.mendeley.com). 

Please download the data from [Here](https://data.mendeley.com/datasets/hx3rzw5dwt/draft?a=377d5571-af17-4e61-bf77-1b77b88316de).

### Pre-processing the dataset

The code requires linguitsitc features to be pre-processed.

We will provide the scripts for pre-processing and update dependencies soon.

### Dependencies

Theano 0.9
Lasagne 0.2
python 3.4

### Content

* `char_hierarchical_linguistic_rnn.py` - 4 level char-word-paragraph-document network with paragraph-level and document-level                                                     linguistic features.
* `char_hierarchical_para.py` - 4 level char-word-paragraph-document network with paragraph-level linguistic features
* `char_hierarchical_doc.py` - 4 level char-word-paragraph-document network with document-level linguistic features
* `char_hierarchical.py` - 4 level char-word-paragraph-document network
* `hierarchical_linguistic_rnn.py` - 3 level word-paragraph-document network with linguistic features
* `hierarchical_rnn.py` - Implementation of [Hierarchical Attention Network](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
* `arg_parser.py`
* `data_utilities.py`
* `evaluations.py`
* `minibatch.py`
* `networks.py`
* `theano_function.py`
* `utilities.py`

### Hyper-parameters

Please see `arg_parser.py` for detail.

## Question

For more question about the dataset, please contact Fan Yang, fyang11@uh.edu


