# transformer.py
import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_dim=d_model, num_embeddings=vocab_size)

        self.transformers = []
        for i in range(num_layers):
            self.transformers.append(
                TransformerLayer(d_model, d_internal)
            )
        self.transformers_list = torch.nn.ModuleList(self.transformers)

        self.linear = torch.nn.Linear(in_features=d_model, out_features=num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.relu = torch.nn.ReLU()


    def forward(self, indices):
        embedding = self.embedding(indices)
        transformer_output = None
        self.attention_maps = []
        # positional encoding

        # transformer layers
        for layer in self.transformers_list:
            (transformer_output, attention) = layer(embedding)
            self.attention_maps.append(attention)
            embedding = transformer_output

        # embedding has the output now
        log_probs = self.softmax(self.relu(self.linear(embedding)))
        # print(log_probs)
        return (log_probs, self.attention_maps)


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    class FFN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.identity = torch.nn.Identity()
            self.relu = torch.nn.ReLU()

            layers = [
                torch.nn.Linear(in_features=dim, out_features=100),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=100, out_features=dim)
            ]
            self.network = torch.nn.Sequential(*layers)
            self.norm = torch.nn.LayerNorm([dim])

        def forward(self, x):
            output = self.network(x)
            return self.norm(output + x)

    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()

        # attention
        self.w_q = torch.nn.Linear(in_features=d_model, out_features=d_internal)
        self.w_k = torch.nn.Linear(in_features=d_model, out_features=d_internal)
        self.w_v = torch.nn.Linear(in_features=d_model, out_features=d_internal)

        # output weight
        self.w_o = torch.nn.Linear(in_features=d_internal, out_features=d_model)

        self.ffn = self.FFN(dim=d_model)
        self.softmax_fn = torch.nn.Softmax(dim=1)
        self.scale_factor = torch.sqrt(torch.FloatTensor([d_internal]))
        self.norm = torch.nn.LayerNorm([d_model])

    def forward(self, input_vecs):
        Q = self.w_q(input_vecs)
        K = self.w_k(input_vecs)
        V = self.w_v(input_vecs)

        # print('Q: ' + repr(Q.shape) + '; K: ' + repr(K.shape) + '; V: ' + repr(V.shape))
        softmax_input = torch.matmul(Q, torch.transpose(K, 0, 1)) / self.scale_factor
        softmax = self.softmax_fn(softmax_input)
        # print('softmax shape: ' + repr(softmax))
        attention = torch.matmul(softmax, V)

        # print('attention shape: ' + repr(attention.shape))
        attention_output = self.w_o(attention)

        # residual
        attention_output += input_vecs

        attention_output = self.norm(attention_output)

        # print('attention output shape: ' + repr(attention_output.shape))
        ffn_output = self.ffn(attention_output)
        # print('ffn output shape: ' + repr(ffn_output.shape))
        return (ffn_output, attention)


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # train  = dev

    # model_test = Transformer(vocab_size=27, num_positions=20, d_model=27, d_internal=20, num_classes=3, num_layers=1)
    # (output, attentions) = model_test.forward(dev[0].input_tensor)
    # return 0

    model = Transformer(vocab_size=27, num_positions=20, d_model=64, d_internal=50, num_classes=3, num_layers=1)
    if device is not None:
        model = model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fcn = nn.NLLLoss()

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        for ex_idx in ex_idxs:
            example = train[ex_idx]
            input_tensor = example.input_tensor
            if device is not None:
                input_tensor = input_tensor.to(device)
            (output, attention_map) = model(input_tensor)

            # predictions = np.argmax(output.detach().numpy(), axis=1)
            # print(predictions)
            loss = loss_fcn(output, example.output_tensor.to(device))
            loss_this_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss at this epoch: ' + repr(loss_this_epoch))
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        # print(log_probs)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
