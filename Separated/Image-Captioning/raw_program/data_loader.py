import nltk
import os
import torch
import torch.utils.data as data
from raw_program.vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """
    Returns the data loader.
    
    Args:
    - transform: Image transform.
    - mode: One of 'train' or 'test'.
    - batch_size: Batch size (if in testing mode, must have batch_size=1).
    - vocab_threshold: Minimum word frequency threshold.
    - vocab_file: File containing the vocabulary.
    - start_word: Special word denoting sentence start.
    - end_word: Special word denoting sentence end.
    - unk_word: Special word denoting unknown words.
    - vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
    - num_workers: Number of subprocesses to use for data loading.
    - cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if vocab_from_file == False:
        assert mode == 'train', "To generate vocab from captions file, mode must be set to 'train' mode."
        
    # Set img_folder and annotations_file based on mode (train, val, test)
    if mode == 'train':
        if vocab_from_file:
            assert os.path.exists(vocab_file), "vocab_file does not exist. To create vocab from scratch, set vocab_from_file to False."
        img_folder = os.path.join(cocoapi_loc, 'coco2017/train2017/')
        annotations_file = os.path.join(cocoapi_loc, 'coco2017/annotations_trainval2017/captions_train2017.json')
    if mode == 'test':
        assert batch_size == 1, "In test mode, batch_size should be set to 1."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "To use existing vocab file, set vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'coco2017/test2017/')
        annotations_file = os.path.join(cocoapi_loc, 'coco2017/image_info_test2017/annotations/image_info_test2017.json')

    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        # Randomly sample caption lengths, and use corresponding indices to retrieve elements (i.e., captions) from the dataset
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve elements using the indices returned above
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader


 
class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        # Initialize a Vocabulary class, see vocabulary.py for more details
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Tokenizing captions...')
            # Convert all input captions to lowercase and tokenize, one element per annotation
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            # Get a list of the token length for each annotation
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            # One element per filename in test set
            self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
        # Get the image and its caption
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and preprocess it
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())  # Convert to lowercase and tokenize
            caption = []
            caption.append(self.vocab(self.vocab.start_word))           # Add start token
            caption.extend([self.vocab(token) for token in tokens])     # Add middle tokens
            caption.append(self.vocab(self.vocab.end_word))             # Add end token
            caption = torch.Tensor(caption).long()                      # Convert from list of tokens to long tensor

            # Return preprocessed image and caption tensor
            return image, caption

        # Get image in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and preprocess it
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and preprocessed image
            return orig_image, image

    def get_train_indices(self):
        # Choose a caption length randomly from the list of caption lengths
        sel_length = np.random.choice(self.caption_lengths)
        # Find the indices of the captions in the list that have the same length as the randomly chosen length
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # Choose batch_size number of indices randomly from the indices with the same caption length
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

def __len__(self):
    if self.mode == 'train':
        return len(self.ids)
    else:
        return len(self.paths)
