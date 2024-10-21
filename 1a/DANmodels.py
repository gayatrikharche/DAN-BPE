import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples, WordEmbeddings
from utils import Indexer

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer, max_len=50):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.word_indexer = word_indexer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        # Convert words to their corresponding indices, using 1 for UNK and 0 for PAD
        word_indices = [self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1 else 1 for word in words]
    
        # Padding or truncating the word indices to the maximum length
        if len(word_indices) < self.max_len:
            word_indices += [0] * (self.max_len - len(word_indices))  # Pad with zeros
        else:
            word_indices = word_indices[:self.max_len]  # Truncate to max_len
    
        # Convert to torch LongTensor (which is required by the embedding layer)
        word_indices = torch.tensor(word_indices, dtype=torch.long)  # Ensure LongTensor
    
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure LongTensor for labels too
        return word_indices, label
        
class DAN(nn.Module):
    def __init__(self, word_embeddings, hidden_size, num_layers=1, dropout=0.0):
        super(DAN, self).__init__()

        # Ensure embedding layer is initialized with the correct vocab size
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=False)
        embed_dim = word_embeddings.get_embedding_length()

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(embed_dim, hidden_size))

        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dtype!=torch.long:
            x=x.long()
            
        embedded = self.embeddings(x)  # Check if indices exceed vocab size here
        averaged_embeddings = embedded.mean(dim=1)

        for fc in self.fc_layers:
            averaged_embeddings = F.relu(fc(averaged_embeddings))
            averaged_embeddings = self.dropout(averaged_embeddings)

        x = self.output_layer(averaged_embeddings)
        return F.log_softmax(x, dim=1)

