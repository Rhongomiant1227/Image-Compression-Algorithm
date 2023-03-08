import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)   # Load pretrained resnet50
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]      
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)   

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) 
        features = self.embed(features)  # Transform the feature vector to the same size as the word embedding size by linear layer
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_p=0.1):
        super(DecoderRNN, self).__init__()

        # model param
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Word Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # LSTM part
        # batch_first=True:
        # Input tensor shape: (batch_size, caption_length, in_features/embedding_dim)
        # Output tensor shape: (batch_size, caption_length, out/hidden)
        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            self.num_layers, 
                            dropout=drop_p,
                            batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(drop_p)

        # Convert hidden layer values to index vectors as output using fully connected layers
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

        # self.softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):

        # Referring to the original document, take the first n-1 tokens and drop the <end> part
        captions = captions[:, :-1]

        # get Embedded Vector after Word Embedding
        embed = self.embedding(captions)

        # features.size(0) == batch_size
        features = features.view(features.size(0), 1, -1)

        # Splice the vector of Features and Word embedding
        inputs = torch.cat((features, embed), dim=1)

        # hidden -> (h, c)
        # h size: (1, batch_size, hidden_size)
        # c size: (1, batch_size, hidden_size)
        lstm_out, hidden = self.lstm(inputs)
        out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))

        # Dropout Layer
        out = self.dropout(out)

        # Map the hidden layer from the hidden dimension space to the vocabulary dimension space using a fully connected layer
        out = self.fc(out)

        # Output tensor shape: (batch_size*sequence_length, vocabulary_size)
        # Reshape to: (batch_size, sequence_length, vocabulary_size)
        out = out.view(lstm_out.size(0), lstm_out.size(1), -1)

        # Log-Softmax / SoftMax: dim=2
        #out = self.softmax(out) # a probability for each token in the vocabulary

        # Return the final output, not the hidden layer state because it will not be used later

        return out
    
def sample(self, inputs, states=None, max_len=20):
    """
    Receives pre-processed image tensor (inputs) and returns predicted sentence (word indices)

    Args:
        inputs (tensor): pre-processed image tensor
        states (tensor): hidden state initialization
        max_len (int): maximum length of index array to be returned

    Returns:
        outputs (list): index list; length: max_len
    """

    # Initialize output list with values all equal to <end>
    outputs = [1] * max_len
    
    # Initialize hidden state to 0
    hidden = states
    
    # Pass image and get tokens sequence. This is similar to forward propagation.
    with torch.no_grad():
        for i in range(max_len):
            # lstm_out size: (batch_size=1, sequence_length=1, hidden_size)
            lstm_out, hidden = self.lstm(inputs, hidden)
            # out size: (1, hidden_size)
            out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
            # fc_out size (1, vocabulary_size)
            fc_out = self.fc(out)

            # Calculate probability of each index
            p = F.softmax(fc_out, dim=1).data

            # Move to cpu
            p = p.cpu()
            
            # Use top_k sampling to get index of next word
            top_k = 5
            p, top_indices = p.topk(top_k)
            top_indices = top_indices.numpy().squeeze()
            
            # Select the most probable next index in some element-random cases
            p = p.numpy().squeeze()
            token_id = int(np.random.choice(top_indices, p=p/p.sum()))

            # Store this index in the output list
            outputs[i] = token_id
            
            # Build next input from output token
            # inputs size: (1, 1, embedding_size=512)
            input_token = torch.Tensor([[token_id]]).long()
            inputs = self.embedding(input_token)

            if token_id == 1:
                # <end>
                break

    return outputs

