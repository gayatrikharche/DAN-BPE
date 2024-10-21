import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentiment_data import read_sentiment_examples
import argparse
import time
import matplotlib.pyplot as plt
import collections

class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size  
        self.token_to_id = {}
        self.id_to_token = {}

    def learn_bpe(self, corpus):
        token_freq = collections.defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                word = ' '.join(list(word)) + ' </w>'
                token_freq[word] += 1
        
        for word in token_freq:
            for token in word.split():
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)
        
        for _ in range(20):
            pairs = self.get_stats(token_freq)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_pair(best_pair, token_freq)
        
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
    def get_stats(self, token_freq):
        pairs = collections.defaultdict(int)
        for word, freq in token_freq.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq
        return pairs

    def merge_pair(self, pair, token_freq):
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        token_freq = {word.replace(bigram, replacement): freq for word, freq in token_freq.items()}

    def tokenize_(self, sentence):
        tokens = []
        for word in sentence:
            word = ' '.join(list(word)) + ' </w>'
            tokens.extend(word.split())
        return [self.token_to_id.get(token, 1) for token in tokens]

    def train(self, corpus_path):
        spm.SentencePieceTrainer.train(
            input=corpus_path, model_prefix="spm", vocab_size=self.vocab_size
        )
        self.sp = spm.SentencePieceProcessor(model_file="spm.model")

    def load(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def tokenize(self, sentence):
        return self.sp.encode(' '.join(sentence), out_type=int)

    def detokenize(self, ids):
        return self.sp.decode(ids)

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, tokenizer, max_len=50):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        word_indices = self.tokenizer.tokenize(self.sentences[idx])
        if len(word_indices) < self.max_len:
            word_indices += [0] * (self.max_len - len(word_indices))
        else:
            word_indices = word_indices[:self.max_len]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

    
class DAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=1, dropout=0.0):
        super(DAN, self).__init__()

        # Initialize embedding layer with random vectors
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(embed_dim, hidden_size))

        for _ in range(1, num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embeddings(x)
        averaged_embeddings = embedded.mean(dim=1)

        for fc in self.fc_layers:
            averaged_embeddings = F.relu(fc(averaged_embeddings))
            averaged_embeddings = self.dropout(averaged_embeddings)

        x = self.output_layer(averaged_embeddings)
        return F.log_softmax(x, dim=1)
