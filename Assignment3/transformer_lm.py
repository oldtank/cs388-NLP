# models.py

import numpy as np
import torch
from transformer import PositionalEncoding, Transformer
import random

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size


    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel, torch.nn.Module):
    def __init__(self, voc_size, d_model, num_positions, vocab_index):
        super().__init__()
        self.num_positions = num_positions
        self.voc_size = voc_size
        self.vocab_index = vocab_index

        self.embedding = torch.nn.Embedding(embedding_dim=d_model, num_embeddings=voc_size)
        self.positional_embedding = PositionalEncoding(d_model=d_model, num_positions=num_positions)

        # self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, 5, dim_feedforward=100, batch_first=False)
        # self.transformer = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.transformer = Transformer(vocab_size=voc_size, num_positions=num_positions, d_model=d_model, d_internal=200, num_layers=2,
                                       num_classes=voc_size)
        self.softmax_fn = torch.nn.LogSoftmax(dim=1)

        self.linear_layer = torch.nn.Linear(in_features=d_model, out_features=voc_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
    #     embedded_seq = self.positional_embedding(self.embedding(x))
    #
    #     mask = torch.triu(torch.ones(self.num_positions, self.num_positions) * float('-inf'), diagonal=1)
    #     # transformer_output = self.transformer(embedded_seq, mask=self.mask)
    #     transformer_output = self.transformer(embedded_seq)

        (log_probs, _) = self.transformer(x)

        # log_probs = self.softmax_fn(self.relu(self.linear_layer(transformer_output)))
        return log_probs

    def get_next_char_log_probs(self, context):
        self.eval()
        context = self.pad_input(context)
        context = self.get_input_index(context)

        log_probs = self.forward(context)
        return log_probs[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        prob = 0.0
        for curr_char in next_chars:
            # curr_char = next_chars[i]
            probs = self.get_next_char_log_probs(context)

            # print(curr_char)
            index_to_get = self.vocab_index.index_of(curr_char)
            # print(probs)
            # print(index_to_get)
            # print(probs[index_to_get])
            prob += probs[index_to_get]
            context = context + curr_char
        return prob

    def pad_input(self, context):
        if len(context) < self.num_positions:
            num_to_pad = self.num_positions - len(context)
            padding = ""
            for i in range(num_to_pad):
                padding += " "
            return padding + context
        else:
            return context[-self.num_positions:]

    def get_input_index(self, input):
        if len(input) != self.num_positions:
            raise Exception('Expecting input of length ' + repr(self.num_positions) + ': ' + input + ' actual size: ' + repr(len(input)))
        input_indexed = np.array([self.vocab_index.index_of(ci) for ci in input])
        return  torch.LongTensor(input_indexed)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # train_text = dev_text
    loss_fcn = torch.nn.NLLLoss()

    # convert string into chunks of 20 characters
    chunk_size = 30
    chunks = []
    for i in range(0, len(train_text), chunk_size):
        chunks.append(train_text[i:i+chunk_size])

    model = NeuralLanguageModel(voc_size=27, d_model=30, vocab_index=vocab_index, num_positions=chunk_size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    for t in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(0, len(chunks))]
        random.shuffle(ex_idxs)

        for ex_id in ex_idxs:
            curr_chunk = chunks[ex_id]
            if len(curr_chunk) < chunk_size:
                continue

            input = " " + curr_chunk[0:chunk_size-1]
            input_indexed = np.array([vocab_index.index_of(ci) for ci in input])

            # print(curr_chunk)
            # print(input)
            log_probs = model(torch.LongTensor(input_indexed))
            labels = torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in curr_chunk]))
            loss = loss_fcn(log_probs, labels)
            loss_this_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("total loss for this epoch %i: %f" % (t, loss_this_epoch))
    model.eval()
    # first = chunks[0]
    # first = " " + first[0:19]
    #
    # label = torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in first]))
    # output = model(first)
    # loss_val = loss_fcn(output, label).item()
    # log_probs = model.get_next_char_log_probs("i am a student")
    # print(log_probs)
    # print(np.sum(np.exp(log_probs)))
    return model