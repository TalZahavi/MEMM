from datetime import datetime
import numpy as np


class MemmTrainer:
    def __init__(self):
        self.history_tag_tuples = set()
        self.word_to_tags_dict = dict()
        self.bigram_tag_dict = dict()
        self.trigram_tag_dict = dict()
        # self.features_matrix = np.zeros(shape=(0, 0), dtype=int)
        self.features_num = dict()
        self.tags = set()

    def train_history_tag_tuples(self):
        with open('train.wtag', 'r') as f:
            for line in f:
                self.train_history_tag_tuples_for_sentence(line)
        f.close()

    def train_history_tag_tuples_for_sentence(self, sentence):
        sentence_words = []
        sentence_tags = []

        for word in sentence.split():
            split_word = word.split('_')
            sentence_words.append(split_word[0])
            sentence_tags.append(split_word[1])

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

            self.history_tag_tuples.add(((first_tag, second_tag, clean_sentence, index), sentence_tags[index]))
            self.tags.add(sentence_tags[index])

    def train_dicts(self):
        for i, train_tuple in enumerate(self.history_tag_tuples):
            word_tag = train_tuple[1]
            history = train_tuple[0]
            word_index = history[3]
            split_sentence = (history[2]).split()
            word = split_sentence[word_index]

            self.add_word_tag_to_dict(word, word_tag, i)
            self.add_bigram_tag_to_dict(history[1], word_tag, i)
            self.add_trigram_tag_to_dict(history[0], history[1], word_tag, i)

    def add_word_tag_to_dict(self, word, tag, history_num):
        if word in self.word_to_tags_dict:
            tags_per_word_dict = self.word_to_tags_dict[word]
            if tag in tags_per_word_dict:
                history_num_list_pet_tag = tags_per_word_dict[tag]
                history_num_list_pet_tag.append(history_num)
            else:
                tags_per_word_dict[tag] = [history_num]
        else:
            self.word_to_tags_dict[word] = {tag: [history_num]}

    def add_bigram_tag_to_dict(self, tag_minus, tag, history_num):
        if tag in self.bigram_tag_dict:
            last_tags_per_tag_dict = self.bigram_tag_dict[tag]
            if tag_minus in last_tags_per_tag_dict:
                history_num_list_per_last_tag = last_tags_per_tag_dict[tag_minus]
                history_num_list_per_last_tag.append(history_num)
            else:
                last_tags_per_tag_dict[tag_minus] = [history_num]
        else:
            self.bigram_tag_dict[tag] = {tag_minus: [history_num]}

    def add_trigram_tag_to_dict(self, tag_minus_two, tag_minus, tag, history_num):
        if tag in self.trigram_tag_dict:
            last_two_tags_per_tag_dict = self.trigram_tag_dict[tag]
            if (tag_minus_two, tag_minus) in last_two_tags_per_tag_dict:
                history_num_list_per_last_two_tags = last_two_tags_per_tag_dict[(tag_minus_two, tag_minus)]
                history_num_list_per_last_two_tags.append(history_num)
            else:
                last_two_tags_per_tag_dict[(tag_minus_two, tag_minus)] = [history_num]
        else:
            self.trigram_tag_dict[tag] = {(tag_minus_two, tag_minus): [history_num]}

    def count_number_of_features(self):
        features_count = 0
        for key in self.word_to_tags_dict:
            features_count += len(self.word_to_tags_dict[key])
        for key in self.bigram_tag_dict:
            features_count += len(self.bigram_tag_dict[key])
        for key in self.trigram_tag_dict:
            features_count += len(self.trigram_tag_dict[key])
        return features_count

    # def fill_in_features_matrix(self):
    #     self.features_matrix = np.zeros(shape=(len(self.history_tag_tuples), self.count_number_of_features()), dtype=int)
    #
    #     feature_num = 0
    #     for key in self.word_to_tags_dict:
    #         tags_for_word_dict = self.word_to_tags_dict[key]
    #         for key_tags in tags_for_word_dict:
    #             one_list = tags_for_word_dict[key_tags]
    #             for one_index in one_list:
    #                 self.features_matrix[one_index, feature_num] = 1
    #             feature_num += 1
    #
    #     for key in self.bigram_tag_dict:
    #         last_tag_dict = self.bigram_tag_dict[key]
    #         for key_tags in last_tag_dict:
    #             one_list = last_tag_dict[key_tags]
    #             for one_index in one_list:
    #                 self.features_matrix[one_index, feature_num] = 1
    #             feature_num += 1
    #
    #     for key in self.trigram_tag_dict:
    #         last_two_tags_dict = self.trigram_tag_dict[key]
    #         for key_tags in last_two_tags_dict:
    #             one_list = last_two_tags_dict[key_tags]
    #             for one_index in one_list:
    #                 self.features_matrix[one_index, feature_num] = 1
    #             feature_num += 1

    def fill_in_features_matrix2(self):
        self.features_m = dict()

        feature_num = 0
        for key in self.word_to_tags_dict:
            tags_for_word_dict = self.word_to_tags_dict[key]
            for key_tags in tags_for_word_dict:
                one_list = tags_for_word_dict[key_tags]
                for one_index in one_list:
                    # self.features_matrix[one_index, feature_num] = 1
                    if feature_num in self.features_m:
                        index_list = self.features_m[feature_num]
                        index_list.append(one_index)
                    else:
                        self.features_m[feature_num] = [one_index]
                self.features_num[(key, key_tags)] = feature_num
                feature_num += 1

        for key in self.bigram_tag_dict:
            last_tag_dict = self.bigram_tag_dict[key]
            for key_tags in last_tag_dict:
                one_list = last_tag_dict[key_tags]
                for one_index in one_list:
                    # self.features_matrix[one_index, feature_num] = 1
                    if feature_num in self.features_m:
                        index_list = self.features_m[feature_num]
                        index_list.append(one_index)
                    else:
                        self.features_m[feature_num] = [one_index]
                self.features_num[(key, key_tags)] = feature_num
                feature_num += 1

        for key in self.trigram_tag_dict:
            last_two_tags_dict = self.trigram_tag_dict[key]
            for key_tags in last_two_tags_dict:
                one_list = last_two_tags_dict[key_tags]
                for one_index in one_list:
                    # self.features_matrix[one_index, feature_num] = 1
                    if feature_num in self.features_m:
                        index_list = self.features_m[feature_num]
                        index_list.append(one_index)
                    else:
                        self.features_m[feature_num] = [one_index]
                self.features_num[key, key_tags] = feature_num
                feature_num += 1

    # def func_l_part_one(self, v):
    #     result = 0
    #     for i in range(0, len(self.history_tag_tuples)):
    #         result += np.dot(v, self.features_matrix[i, ])
    #     print('Result 1 is ' + str(result))

    def func_l_part_one2(self, v):
        result = 0
        for i in range(0, len(v)):
            result += len(self.features_m[i])*v[i]
        return result

########################################################################

    def get_features_numbers_for_tuple(self, train_tuple):
        features_num = []

        word_tag = train_tuple[1]
        history = train_tuple[0]
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]

        # 100 feature
        features_num.append(self.features_num[(word, word_tag)])
        # 103 feature
        features_num.append(self.features_num[(word_tag, (history[0], history[1]))])
        # 104 feature
        features_num.append(self.features_num[(word_tag, history[1])])

        return features_num

    def calculate_feature_vector(self, v_vector, train_tuple):
        result = 0

        word_tag = train_tuple[1]
        history = train_tuple[0]
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]

        if (word, word_tag) in self.features_num:
            result += v_vector[self.features_num[(word, word_tag)]]
        if (word_tag, history[1]) in self.features_num:
            result += v_vector[self.features_num[(word_tag, history[1])]]
        if (word_tag, (history[0], history[1])) in self.features_num:
            result += v_vector[self.features_num[(word_tag, (history[0], history[1]))]]
        return result

    def func_l_part_two(self, v_vector):
        total_result = 0
        for train_tuple in self.history_tag_tuples:
            exp_array = []
            history = train_tuple[0]
            for tag in self.tags:
                exp_array.append(self.calculate_feature_vector(v_vector, (history, tag)))
            exp_array_result = np.exp(exp_array)

            total_result += np.log2(sum(exp_array_result))
        return total_result

################################################################

    def func_l(self, v_vector):
        a = self.func_l_part_one2(v_vector)
        b = self.func_l_part_two(v_vector)
        return a-b


startTime = datetime.now()
###############################

trainer = MemmTrainer()
trainer.train_history_tag_tuples()
trainer.train_dicts()

# trainer.fill_in_features_matrix()
# trainer.func_l_part_one(np.ones(shape=trainer.count_number_of_features(), dtype=int))

trainer.fill_in_features_matrix2()
v = np.ones(shape=trainer.count_number_of_features(), dtype=int)
trainer.func_l(v)

###############################
print(datetime.now() - startTime)

