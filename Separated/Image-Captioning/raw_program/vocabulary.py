import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):

    def __init__(self, vocab_threshold, vocab_file='./vocab.pkl', start_word="<start>", end_word="<end>",
                 unk_word="<unk>", annotations_file='../coco2017/annotations_trainval2017/captions_train2017.json',
                 vocab_from_file=False):
        """Initializes Vocabulary class
        Args:
            vocab_threshold: Minimum word frequency threshold
            vocab_file: File containing vocabulary
            start_word: Special word denoting sentence start
            end_word: Special word denoting sentence end
            unk_word: Unknown word denoting rare words with frequency below the threshold
            annotations_file: Path to train annotation.json file
            vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load vocabulary from an existing file or create vocabulary file"""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Loaded vocabulary from vocab.pkl file')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
            print('Initialized vocabulary and saved it to vocab.pkl file')
        
    def build_vocab(self):
        """Populate dictionary to convert tokens to integers (and vice-versa)"""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize a dictionary to build a vocabulary for converting captions to tensors"""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add word to dictionary"""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Read captions and add all tokens that reach or exceed the threshold to the vocabulary."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("\r[%d/%d] Tokenizing captions and building vocabulary..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        """Convert word to corresponding integer in the dictionary"""
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
