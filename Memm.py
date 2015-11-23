import numpy as np
from datetime import datetime
from Features import Feature100
from Features import Feature103
from Features import Feature104


class PosTrainer:
    def __init__(self):
        self.history_tag_tuple_set = set()
        self.tags = set()
        self.history_tag_tuple_array = np.array

        self.features_apply_on_tuples = []

        self.seen_trigrams = set()
        self.seen_bigrams = set()
        self.seen_word_tag = set()

    # Go over the sentences in the training data, and learn the parameters
    def train(self):
        with open('train.wtag', 'r') as f:
            for line in f:
                self.get_history_tag_tuple_line(line)
        f.close()

    def get_history_tag_tuple_line(self, sentence):
        sentence_words = []
        sentence_tags = []

        for word in sentence.split():
            split_word = word.split('_')
            sentence_words.append(split_word[0])
            sentence_tags.append(split_word[1])
            #self.tags.add(split_word[1])
            self.seen_word_tag.add((split_word[0], split_word[1]))

        self.get_tags_trigrams(sentence_tags)

        clean_sentence = ' '.join(sentence_words)

        for index, word in enumerate(sentence_words):
            if index == 0:
                first_tag = '*'
                second_tag = '*'
            elif index == 1:
                first_tag = '*'
                second_tag = sentence_tags[index-1]
            else:
                first_tag = sentence_tags[index-2]
                second_tag = sentence_tags[index-1]

            self.history_tag_tuple_set.add(((first_tag, second_tag, clean_sentence, index), sentence_tags[index]))
            # self.history_tag_tuple_array = np.array(self.history_tag_tuple_set)

    # Get all the seen sentence tags trigams (and add it to self.seen_trigrams)
    def get_tags_trigrams(self, sentence_tags):
        for index, val in enumerate(sentence_tags):
            if index == 0:
                self.seen_trigrams.add(('*', '*', sentence_tags[index]))
                self.seen_bigrams.add(('*', sentence_tags[index]))
            elif index == 1:
                self.seen_trigrams.add(('*', sentence_tags[index-1], sentence_tags[index]))
                self.seen_bigrams.add((sentence_tags[index-1], sentence_tags[index]))
            else:
                self.seen_trigrams.add((sentence_tags[index-2], sentence_tags[index-1], sentence_tags[index]))
                self.seen_bigrams.add((sentence_tags[index-1], sentence_tags[index]))

    # def apply_features_103(self):
    #     for tags_i in self.tags:
    #         for tags_j in self.tags:
    #             for tags_k in self.tags:
    #                 feature = Feature103(tags_i, tags_j, tags_k)
    #                 # for i in range(0, len(self.history_tag_tuple_set)):
    #                 #     x = self.history_tag_tuple_array[i]
    #                 #     np.append(self.features_apply_on_tuples, feature.calc(self.history_tag_tuple_array[i]))
    #                 for train_tuple in self.history_tag_tuple_set:
    #                     np.append(self.features_apply_on_tuples, feature.calc(train_tuple))

    def apply_features_103(self):
        for trigram in self.seen_trigrams:
            feature = Feature103(trigram[0], trigram[1], trigram[2])
            for train_tuple in self.history_tag_tuple_set:
                self.features_apply_on_tuples.append(feature.calc(train_tuple))

    def apply_features_104(self):
        for bigram in self.seen_bigrams:
            feature = Feature104(bigram[0], bigram[1])
            for train_tuple in self.history_tag_tuple_set:
                self.features_apply_on_tuples.append(feature.calc(train_tuple))


startTime = datetime.now()
##########################
trainer = PosTrainer()
trainer.train()
# trainer.apply_features_104()
# x = np.array(trainer.features_apply_on_tuples)
# y = [1] * len(trainer.features_apply_on_tuples)
# yy = np.array(y)
# z = x * yy
print('Seen bigram tag is ' + str(len(trainer.seen_bigrams)))
print('Seen trigram tag is ' + str(len(trainer.seen_trigrams)))
print('Seen word tag is '+ str(len(trainer.seen_word_tag)))
##########################
print(datetime.now() - startTime)
