from Features import Feature100
from Features import Feature103
from Features import Feature104
from datetime import datetime
import numpy as np

class MemmTrain:
    features = []
    possibleTags = set()
    possibleWords = set()
    trainingSetTuples = set()

    word_tag_tuple = set()

    def train(self):
        with open('train.wtag','r') as f:
            for line in f:
                self.get_history_words_tags(line)
        f.close()

    def get_history_words_tags(self, sentence):
        tags = []
        words = []

        for word in sentence.split():
            split_word = word.split('_')

            words.append(split_word[0])
            tags.append(split_word[1])

            self.word_tag_tuple.add((split_word[0], split_word[1]))

        clean_sentence = ' '.join(words)

        for tag in tags:
            self.possibleTags.add(tag)
        for word in words:
            self.possibleWords.add(word)

        for index, word in enumerate(words):
            if index == 0:
                first_tag = '*'
                second_tag = '*'
            elif index == 1:
                first_tag = '*'
                second_tag = tags[index-1]
            else:
                first_tag = tags[index-2]
                second_tag = tags[index-1]

            self.trainingSetTuples.add(((first_tag, second_tag, clean_sentence, index), tags[index]))

    def apply_features(self, v_vector, f_vector, training_set_tuple):
        value = 0
        for index, func in enumerate(f_vector):
            value = value + func.calc(training_set_tuple)*v_vector[index]
        return value

    def calc_features(self):
        for tup in self.trainingSetTuples:
            val = x.apply_features([1]*len(x.features), self.features, tup)
            print(val)

    def calc_features2(self):
        for tup in self.trainingSetTuples:
            val = x.apply_features(np.array([1]*len(x.features)), np.array(self.features), np.array(tup))
            print(val)

    # def make_tags_features100(self):
    #     for word_i in self.possibleWords:
    #         for tag_j in self.possibleTags:
    #             self.features.append(Feature100(word_i,tag_j))

    def make_tags_features100(self):
        for tup in self.word_tag_tuple:
            t = Feature100(tup[0], tup[1])
            for his in self.trainingSetTuples:
                print(t.calc(his))


            self.features.append(Feature100(tup[0], tup[1]))

    def make_tags_features104(self):
        for tags_i in self.possibleTags:
            for tags_j in self.possibleTags:
                self.features.append(Feature104(tags_i, tags_j))

    def make_tags_features103(self):
        for tags_i in self.possibleTags:
            for tags_j in self.possibleTags:
                for tags_k in self.possibleTags:
                    self.features.append(Feature103(tags_i, tags_j, tags_k))

    def test(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        print(x*y)

x = MemmTrain()
startTime = datetime.now()
x.train()
x.make_tags_features100()
x.make_tags_features103()
x.make_tags_features104()

# result = x.apply_features([1]*len(x.features), x.features, (('IN', 'DT', "By contrast , Value Line said Georgia-Pacific `` is in a comparatively good position to deal with weakening paper markets , '' because its production is concentrated not in the Northwest but in the South , where it should be able to avoid some of the cost pressures from rising wood-chip prices .", 46), 'NN'))
# print(result)
# print(datetime.now() - startTime)
# print(len(x.features))
# print(len(x.possibleWords))

# x.calc_features()
# print(len(x.trainingSetTuples))
# print(datetime.now() - startTime)

#print(x.trainingSetTuples.pop())

# print(len(x.possibleWords))

# x.test()

