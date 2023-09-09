# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    def __init__(self, word_embeddings:WordEmbeddings, embedding_dim, hidden_size, num_classes):
        super(NeuralSentimentClassifier, self).__init__()
        self.word_embeddings = word_embeddings
        self.log_softmax = nn.LogSoftmax(dim=0)

        self.embedding = self.word_embeddings.get_initialized_embedding_layer(frozen=True)
        # self.avg = torch.nn.AvgPool1d(kernel_size=embedding_dim)

        layers =[
            nn.Linear(in_features=embedding_dim, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x.shape: <num_words>
        vectors = self.embedding(x)
        # vectors.shape: num_words * embedding_dimension
        avg = torch.mean(vectors, dim=0)
        # avg.shape: embedding_dimension
        return self.network(avg)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        word_indices = []
        for word in ex_words:
            word_index = self.word_embeddings.word_indexer.index_of(word)
            if word_index != -1:
                word_indices.append(word_index)
            else:
                word_indices.append(self.word_embeddings.word_indexer.index_of("UNK"))
        output = self.forward(torch.tensor(word_indices))
        log_probs = self.log_softmax(output)
        return torch.argmax(log_probs).item()


class PrefixEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_initialized_embedding_layer(self, frozen=True):
        """
        :param frozen: True if you want the embedding layer to stay frozen, false to fine-tune embeddings
        :return: torch.nn.Embedding layer you can use in your network
        """
        return torch.nn.Embedding.from_pretrained(torch.FloatTensor(self.vectors), freeze=frozen)

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_prefix_embeddings(embeddings_file: str) -> PrefixEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()

    prefix_counts = {}
    vectors = []
    # Make position 0 a PAD token, which can be useful if you
    word_indexer.add_and_get_index("PAD")
    # Make position 1 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the PAD and UNK vectors to start. Have to do this weirdly because we need to read the first line
            # of the file to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)

            # handle the prefix vector
            prefix = 'prefix:' + word[0:3]
            if word_indexer.index_of(prefix) > 0:
                index = word_indexer.index_of(prefix)
                existing_vector = vectors[index]
                existing_count = prefix_counts[prefix]
                vectors[index] = (existing_vector * existing_count + vector) / (existing_count + 1)
                prefix_counts[prefix] = existing_count + 1
            else:
                word_indexer.add_and_get_index(prefix)
                vectors.append(vector)
                prefix_counts[prefix] = 1


    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return PrefixEmbeddings(word_indexer, np.array(vectors))


class TypoClassifier(SentimentClassifier, nn.Module):
    def __init__(self, prefix_embeddings:PrefixEmbeddings, embedding_dim, hidden_size, num_classes):
        super(TypoClassifier, self).__init__()
        self.prefix_embeddings = prefix_embeddings
        self.log_softmax = nn.LogSoftmax(dim=0)

        self.embedding = self.prefix_embeddings.get_initialized_embedding_layer(frozen=False)

        layers =[
            nn.Linear(in_features=embedding_dim, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_classes)
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x.shape: <num_words>
        vectors = self.embedding(x)
        # vectors.shape: num_words * embedding_dimension
        avg = torch.mean(vectors, dim=0)
        # avg.shape: embedding_dimension
        return self.network(avg)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        word_indices = []
        for word in ex_words:
            word_index = self.prefix_embeddings.word_indexer.index_of(word)
            prefix_index= self.prefix_embeddings.word_indexer.index_of('prefix:' + word[0:3])
            if word_index != -1:
                word_indices.append(word_index)
            else:
                if prefix_index != -1:
                    word_indices.append(prefix_index)

                else :
                    word_indices.append(self.prefix_embeddings.word_indexer.index_of("UNK"))

        output = self.forward(torch.tensor(word_indices))
        log_probs = self.log_softmax(output)
        return torch.argmax(log_probs).item()


def get_word_indices(sentence:List[str], indexer:Indexer):
    word_indices = []
    for word in sentence:
        word_index = indexer.index_of(word)
        if word_index != -1:
            word_indices.append(word_index)
        else:
            word_indices.append(indexer.index_of("UNK"))
    return torch.tensor(word_indices)


def get_word_indices_with_prefix(sentence:List[str], indexer:Indexer):
    word_indices = []
    for word in sentence:
        word_index = indexer.index_of(word)
        prefix_index = indexer.index_of('prefix:' + word[0:3])
        if word_index != -1:
            word_indices.append(word_index)
        else:
            if prefix_index != -1:
                word_indices.append(prefix_index)
            else:
                word_indices.append(indexer.index_of("UNK"))
    return torch.tensor(word_indices)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    training_data = train_exs

    if train_model_for_typo_setting:
        prefix_embeddings = read_prefix_embeddings(args.word_vecs_path)
        model = TypoClassifier(prefix_embeddings, prefix_embeddings.get_embedding_length(), args.hidden_size, num_classes=2)
    else:
        model = NeuralSentimentClassifier(word_embeddings, word_embeddings.get_embedding_length(), args.hidden_size, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_epoch = 3 if train_model_for_typo_setting else args.num_epochs

    for epoch in range(total_epoch):
        ex_indices = [i for i in range(0, len(training_data))]
        random.shuffle(ex_indices)
        total_loss = 0.0

        for idx in ex_indices:
            sentence = training_data[idx].words
            label = training_data[idx].label

            if train_model_for_typo_setting:
                word_indices = get_word_indices_with_prefix(sentence, prefix_embeddings.word_indexer)
            else:
                word_indices = get_word_indices(sentence, word_embeddings.word_indexer)
            output = model(word_indices)
            loss_val = loss_fn(output.view(1,-1), torch.tensor([label]))
            total_loss += loss_val
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

        print("total loss for this epoch %i: %f" % (epoch, total_loss))
    return model
