import torch
import torch.nn as nn
from torchvision import transforms
import sys
COCOAPIROOT = r".."
from pycocotools.coco import COCO
from raw_program.data_loader import get_loader
from raw_program.model import EncoderCNN, DecoderRNN
import math
import os
import torch.utils.data as data
import numpy as np

import requests
import time


# Select appropriate parameters
batch_size = 128          # batch size
vocab_threshold = 5        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 512          # number of features in hidden state of the RNN decoder
num_epochs = 3             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# Create transforms
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create data loader
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file,
                         cocoapi_loc=COCOAPIROOT)

# Define the vocabulary size
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Create a list of learnable parameters
params = list(encoder.embed.parameters()) + list(decoder.parameters())

# Choose the optimizer
optimizer = torch.optim.Adam(params, lr=0.001)

# Set how many steps to train per epoch
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

# Save
torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-0.pkl'))
torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-0.pkl'))
# Load
decoder_file = 'decoder-0.pkl'
encoder_file = 'encoder-0.pkl'
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file),  map_location='cpu'))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file),  map_location='cpu'))

# Open the training log file.
f = open(log_file, 'w')

# Select True if training on local desktop. False to train on GPU workspace
local = False
# if not local:
start_time = time.time()
#     response = requests.request("GET", 
#                                 "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 
#                                 headers={"Metadata-Flavor":"Google"})
for epoch in range(1, num_epochs+1):
    for i_step in range(1, total_step+1):
        # If not running locally, send a keep-alive signal every minute
        # to prevent workspace from disconnecting.
        # This is only necessary when running on a remote workspace.
        # To use this feature, uncomment the code below.
#         if not local:
#             if time.time() - old_time > 60:
#                 old_time = time.time()
#                 requests.request("POST", 
#                                  "https://nebula.udacity.com/api/v1/remote/keep-alive", 
#                                  headers={'Authorization': "STAR " + response.text})

        # Randomly sample indices from caption_length and return corresponding indices
        indices = data_loader.dataset.get_train_indices()
        # Create a new sample
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        # Get the batch data
        images, captions = next(iter(data_loader))
        # Move the data to GPU
        images = images.to(device)
        captions = captions.to(device)
        # Zero the gradients
        decoder.zero_grad()
        encoder.zero_grad()
        # Pass inputs to encoder and decoder
        features = encoder(images)
        outputs = decoder(features, captions)
        # Compute loss function
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        # Backpropagation
        loss.backward()
        # Update optimizer parameters
        optimizer.step()
        # Get training statistics
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        # Print training statistics (same line)
        print('\r' + stats, end="")
        sys.stdout.flush()
        # Write data to file
        f.write(stats + '\n')
        f.flush()
        # Print training statistics (new line)
        if i_step % print_every == 0:
            print('\r' + stats)

    # Save weights
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

# Close log file
f.close()
end_time = time.time()
print("Training duration: {}".format(end_time-start_time))