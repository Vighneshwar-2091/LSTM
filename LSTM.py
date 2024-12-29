import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# Define the Dictionary class
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

# Define the TextProcess class
class TextProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1

        num_batches = rep_tensor.shape[0] // batch_size
        rep_tensor = rep_tensor[:num_batches * batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor

# Define the LSTM-based TextGenerator model
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(TextGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

# Main script
if __name__ == "__main__":
    # Hyperparameters
    embed_size = 128
    hidden_size = 1024
    num_layers = 1
    num_epochs = 20
    batch_size = 20
    timesteps = 30
    learning_rate = 0.002

    # Input file path (in the same directory as the script)
    input_file = os.path.join(os.getcwd(), 'alice.txt')

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file 'alice.txt' not found in the current directory: {os.getcwd()}")

    # Prepare data
    corpus = TextProcess()
    rep_tensor = corpus.get_data(input_file, batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = rep_tensor.shape[1] // timesteps

    # Initialize model, loss function, and optimizer
    model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))

        for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
            inputs = rep_tensor[:, i:i + timesteps]
            targets = rep_tensor[:, i + 1:(i + 1) + timesteps]

            output, _ = model(inputs, states)
            loss = loss_fn(output, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // timesteps
            if step % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{step}/{num_batches}], Loss: {loss.item():.4f}')

    # Generate text
    with torch.no_grad():
        with open('Results.txt', 'w') as f:
            state = (torch.zeros(num_layers, 1, hidden_size),
                     torch.zeros(num_layers, 1, hidden_size))
            input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)

            for i in range(500):
                output, _ = model(input, state)
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()
                input.fill_(word_id)

                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i + 1) % 100 == 0:
                    print(f'Sampled [{i + 1}/500] words saved to Results.txt')

    print("Training and text generation complete. Results saved to 'Results.txt'")
