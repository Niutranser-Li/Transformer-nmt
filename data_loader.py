import os, sys
from urllib.request import urlretrieve

import tensorflow as tf
import sentencepiece
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataLoader:
    DIR = None
    PATHS = {}
    BPE_VOCAB_SIZE = 0
    MODES = ['source', 'target']
    dictionary = {
        'source':{
            'token2idx': None,
            'idx2token': None
        },
        'target':{
            'token2idx': None,
            'idx2token': None
        }
    }
    DATASET_SOURCE = {
        'wmt14/en-de':{
            'source_lang': 'en',
            'target_lang': 'de',
            'base_url': 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/',
            'train_files': ['train.en', 'train.de'],
            'vocab_files': ['vocab.50K.en', 'vocab.50K.de'],
            'dictionary_files': ['dict.en-de'],
            'test_files': [
                'newstest2012.en', 'newstest2012.de',
                'newstest2013.en', 'newstest2013.de',
                'newstest2014.en', 'newstest2014.de',
                'newstest2015.en', 'newstest2015.de'
            ]
        }
    }
    BPE_MODEL_SUFFIX = '.model'
    BPE_VOCAB_SUFFIX = '.vocab'
    BPE_RESULT_SUFFIX = '.sequences'
    SEQ_MAX_LEN = {
        'source': 100,
        'target': 100
    }
    DATA_LIMIT = None
    TRAIN_RATIO = 0.9
    BATCH_SIZE = 16

    source_sp = None
    target_sp = None

    def __init__(self, dataset_name=None, data_dir=None, batch_size=16, bpe_vocab_size=32000, seq_max_len_source=100, 
                seq_max_len_target=100, data_limit=None, train_ratio=0.9):
        if dataset_name is None or data_dir is None:
            raise ValueError('dataset_name and data_dir parameters must be defined.')
        if dataset_name not in self.DATASET_SOURCE.keys():
            raise ValueError('dataset_name parameter must be set in {}'.format(list(self.DATASET_SOURCE.keys())))
        self.DIR = data_dir
        self.DATASET = dataset_name
        self.BPE_VOCAB_SIZE = bpe_vocab_size
        self.BATCH_SIZE = batch_size
        self.SEQ_MAX_LEN['source'] = seq_max_len_source
        self.SEQ_MAX_LEN['target'] = seq_max_len_target
        self.DATA_LIMIT = data_limit
        self.TRAIN_RATIO = train_ratio

        self.PATHS['source_data'] = os.path.join(self.DIR, self.DATASET_SOURCE[self.DATASET]['train_files'][0])
        self.PATHS['source_bpe_prefix'] = self.PATHS['source_data'] + '.segmented'

        self.PATHS['target_data'] = os.path.join(self.DIR, self.DATASET_SOURCE[self.DATASET]['train_files'][1])
        self.PATHS['target_bpe_prefix'] = self.PATHS['target_data'] + '.seqmented'

        self.parameters_checking()

    def parameters_checking(self):
        if not isinstance(self.BATCH_SIZE, int) or self.BATCH_SIZE<=0:
            raise ValueError('batch_size parameter must be set a greater than zero interger.')
        if not isinstance(self.BPE_VOCAB_SIZE, int) or self.BPE_VOCAB_SIZE<=0:
            raise ValueError('bpe_vocab_size parameter must be set a greater than zero interger.')
        if not isinstance(self.SEQ_MAX_LEN['source'], int) or self.SEQ_MAX_LEN['source']<=0:
            raise ValueError('seq_max_len_source parameter must be set a greater than zero interger.')
        if not isinstance(self.SEQ_MAX_LEN['target'], int) or self.SEQ_MAX_LEN['target']<=0:
            raise ValueError('seq_max_len_target parameter must be set a greater than zero interger.')
        if self.TRAIN_RATIO < 0 or self.TRAIN_RATIO > 1:\
            raise ValueError('train_ratio parameter must be in [0, 1]')

    def _download(self, download_url):
        print('_download_url:', download_url)
        save_route = os.path.join(self.DIR, download_url.split('/')[-1])
        if not os.path.exists(save_route):
            with TqdmCustom(unit='B', unit_scale=True, unit_divisor=8192, miniters=1, desc=download_url) as t:
                urlretrieve(download_url, save_route, t.update_to)
    
    def download_dataset(self):
        for file_name in (self.DATASET_SOURCE[self.DATASET]['train_files'] + self.DATASET_SOURCE[self.DATASET]['vocab_files']
                        + self.DATASET_SOURCE[self.DATASET]['dictionary_files'] + self.DATASET_SOURCE[self.DATASET]['test_files']):
            self._download(os.path.join(self.DATASET_SOURCE[self.DATASET]['base_url'], file_name))

    def load(self, custom_dataset=False):
        if custom_dataset:
            print('#step.1 - use custom dataset. please implement custom download_dataset function.')
        else:
            print('#step.1 - download dataset.')
            self.download_dataset()
        
        print('#step.2 - load dataset.')
        source_data = self.parse_data_and_save(self.PATHS['source_data'])
        target_data = self.parse_data_and_save(self.PATHS['target_data'])

        print('#step.3 - train bpe.')
        self.train_bpe(self.PATHS['source_data'], self.PATHS['source_bpe_prefix'])
        self.train_bpe(self.PATHS['target_data'], self.PATHS['target_bpe_prefix'])

        print('#step.4 - load bpe vocab.')
        self.dictionary['source']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )

        print('step.5 - encode line data with bpe.')
        source_sequences = self.texts_to_sequences(
            self.sentence_piece(
                source_data,
                self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                self.PATHS['target_bpe_prefix'] + self.BPE_RESULT_SUFFIX
            ),
            mode = 'source'
        )
        target_sequences = self.texts_to_sequences(
            self.sentence_piece(
                target_data,
                self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX,
                self.PATHS['target_bpe_prefix'] + self.BPE_RESULT_SUFFIX
            ),
            mode = 'target'
        )

        print('source sequence example:', source_sequences[0])
        print('target sequence example:', target_sequences[0])

        if self.TRAIN_RATIO==1.0:
            source_sequences_train, source_sequences_valid = source_sequences, []
            target_sequences_train, target_sequences_valid = target_sequences, []
        else:
            (source_sequences_train, source_sequences_valid, target_sequences_train, target_sequences_valid) = train_test_split(
                source_sequences, target_sequences, train_size=self.TRAIN_RATIO
            )
        
        if self.DATA_LIMIT is not None:
            print('dataset size limit No. limit size:', self.DATA_LIMIT)
            source_sequences_train = source_sequences_train[:self.DATA_LIMIT]
            target_sequences_train = target_sequences_train[:self.DATA_LIMIT]
        
        print('source sequence train size:', len(source_sequences_train))
        print('source sequence valid size:', len(source_sequences_valid))
        print('target sequence train size:', len(target_sequences_train))
        print('target sequence valid size:', len(target_sequences_valid))

        train_dataset = self.create_dataset(
            source_sequences_train, target_sequences_train
        )
        if self.TRAIN_RATIO==1.0: valid_dataset = None
        else:
            valid_dataset = self.create_dataset(
                source_sequences_valid, target_sequences_valid
            )
        
        return train_dataset, valid_dataset

    def load_test(self, index=0, custom_dataset=False):
        if index < 0 or index >= len(self.DATASET_SOURCE[self.DATASET]['test_files'])//2:
            raise ValueError('test file index out of range. value min:0 max:{}'.format(
                len(self.DATASET_SOURCE[self.DATASET]['test_files'])//2 - 1)
            )
        if custom_dataset:
            print('load test_dataset step.1 - use custom dataset, please implement custom download_dataset function.')
        else:
            print('load test_dataset step.1 - download test dataset.')
            self.download_dataset()
        
        print('load test_dataset step.2 - parse dataset.')
        source_test_data_path, target_test_data_path = self.get_test_data_path(index)
        source_test_data = self.parse_data_and_save(source_test_data_path)
        target_test_data = self.parse_data_and_save(target_test_data_path)

        print('load test_dataset step.3 - load bpe vocab.')
        self.dictionary['source']['token2idx'], self.dictionary['source']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['source_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )
        self.dictionary['target']['token2idx'], self.dictionary['target']['idx2token'] = self.load_bpe_vocab(
            self.PATHS['target_bpe_prefix'] + self.BPE_VOCAB_SUFFIX
        )

        return source_test_data, target_test_data
    
    def get_test_data_path(self, index=0):
        source_test_data_path = os.path.join(self.DIR, self.DATASET_SOURCE[self.DATASET]['test_files'][index*2])
        target_test_data_path = os.path.join(self.DIR, self.DATASET_SOURCE[self.DATASET]['test_files'][index*2+1])
        return source_test_data_path, target_test_data_path

    def create_dataset(self, source_sequences, target_sequences):
        new_source_sequences, new_target_sequences = [], []
        for source, target in zip(source_sequences, target_sequences):
            if len(source) > self.SEQ_MAX_LEN['source']: source = source[:self.SEQ_MAX_LEN['source']]
            if len(target) > self.SEQ_MAX_LEN['target']: target = target[:self.SEQ_MAX_LEN['target']]
            new_source_sequences.append(source)
            new_target_sequences.append(target)
        source_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_source_sequences, maxlen=self.SEQ_MAX_LEN['source'], padding='post'
        )
        target_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=new_target_sequences, maxlen=self.SEQ_MAX_LEN['target'], padding='post'
        )
        buffer_size = int(source_sequences.shape[0]*0.3)
        dataset = tf.data.Dataset.from_tensor_slices(
            (source_sequences, target_sequences)
        ).shuffle(buffer_size)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def train_bpe(self, data_route, model_prefix):
        model_path = model_prefix + self.BPE_MODEL_SUFFIX
        vocab_path = model_prefix + self.BPE_VOCAB_SUFFIX

        if not os.path.exists(model_path) and not os.path.exists(vocab_path):
            print('bpe model does not exists. start train bpe, model_path:{} vocab_path:{}'.format(model_path, vocab_path))
            train_params = "--input={} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_prefix={} --vocab_size={} --model_type=bpe".format(
                data_route, model_prefix, self.BPE_VOCAB_SIZE)
            print('train_params:', train_params)
            sentencepiece.SentencePieceTrainer.Train(train_params)
        else:
            print('bpe model exists, load bpe, model path:{} vocab path:{}'.format(model_path, vocab_path))
    
    def sentence_piece(self, source_data, source_bpe_model_path, result_data_path):
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(source_bpe_model_path)

        print('sentence_piece result_data_path:', result_data_path)

        if os.path.exists(result_data_path):
            print('encoded data exists. load data. path:', result_data_path)
            with open(result_data_path, 'r', encoding='utf-8') as filein:
                sequences = filein.read().strip().split('\n')
                return sequences
        
        print('encoded data does not exists. encode data. path:', result_data_path)
        sequences = []
        with open(result_data_path, 'w', encoding='utf-8') as fileout:
            for sentence in tqdm(source_data):
                pieces = sp.EncodeAsPieces(sentence)
                sequence = " ".join(pieces)
                sequences.append(sequence)
                fileout.write(sequence + '\n')
        return sequences
    
    def load_bpe_vocab(self, bpe_vocab_path):
        print('load_bpe_vocab load bpe route:', bpe_vocab_path)
        with open(bpe_vocab_path, 'r', encoding='utf-8') as filein:
            vocab = [line.split()[0] for line in filein.read().splitlines()]
        
        token2idx = {}
        idx2token = {}

        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token
    
    def texts_to_sequences(self, texts, mode='source'):
        if mode not in self.MODES: ValueError('not allowed mode.')
        sequences = []
        for text in texts:
            text_list = ['</s>'] + text.split() + ['</s>']
            sequence = [
                self.dictionary[mode]['token2idx'].get(
                    token, self.dictionary[mode]['token2idx']['<unk>']
                )
                for token in text_list
            ]
            sequences.append(sequence)
        return sequences
    
    def sequences_to_texts(self, sequences, mode='source'):
        if mode not in self.MODES: ValueError('not allowed mode.')
        texts = []
        for sequence in sequences:
            if mode == 'source':
                if self.source_sp is None:
                    self.source_sp = sentencepiece.SentencePieceProcessor()
                    self.source_sp.Load(self.PATHS['source_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.source_sp.DecodeIds(sequences)
            else:
                if self.target_sp is None:
                    self.target_sp = sentencepiece.SentencePieceProcessor()
                    self.target_sp.load(self.PATHS['target_bpe_prefix'] + self.BPE_MODEL_SUFFIX)
                text = self.target_sp.DecodeIds(sequence)
            texts.append(text)
        return texts

    def parse_data_and_save(self, path):
        print('load dataset from {}'.format(path))
        with open(path, 'r', encoding='utf-8') as filein:
            lines = filein.read().strip().split('\n')
        print('correct file: {} lines number: {}'.format(os.path.basename(path), len(lines)))
        if lines is None:
            raise ValueError('vocab file is invalid.')
        with open(path, 'w', encoding='utf-8') as fileout:
            fileout.write('\n'.join(lines))
        return lines

class TqdmCustom(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b*bsize - self.n)

data_loader = DataLoader('wmt14/en-de', './datasets')
data_loader.load()

