from datetime import datetime
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class BasicTrainer:
    def __init__(self):
        self.tags = set()
        self.history_tag_tuples = set()

        self.num_features = 0
        self.word_to_tags_dict = dict()
        self.bigram_tag_dict = dict()
        self.trigram_tag_dict = dict()

        self.features = dict()
        self.features_by_index = dict()

        self.calculated_features = dict()

    # Go over the training data and find all the (history,tag) tuples
    def get_history_tag_tuples(self):
        with open('train.wtag', 'r') as f:
            for line in f:
                self.get_history_tag_tuples_for_sentence(line)
        f.close()

    # find the (history,tag) tuple for a given sentence (also all the possible tags)
    def get_history_tag_tuples_for_sentence(self, sentence):
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

    # Fill the features dicts according to the seen data
    def fill_features_dicts(self):
        for (history, word_tag) in self.history_tag_tuples:
            word_index = history[3]
            split_sentence = (history[2]).split()
            word = split_sentence[word_index]

            self.add_word_tag_to_dict(word, word_tag)
            self.add_bigram_tag_to_dict(history[1], word_tag)
            self.add_trigram_tag_to_dict(history[0], history[1], word_tag)

    # Fill the 100 feature with a new seen feature
    # If already saw the feature, update the counter
    def add_word_tag_to_dict(self, word, word_tag):
        if word in self.word_to_tags_dict:
            tags_per_word_dict = self.word_to_tags_dict[word]

            if word_tag in tags_per_word_dict:
                tags_per_word_dict[word_tag] += 1
            else:
                tags_per_word_dict[word_tag] = 1
                self.num_features += 1

        else:
            self.word_to_tags_dict[word] = {word_tag: 1}
            self.num_features += 1

    # Fill the 104 feature with a new seen feature
    # If already saw the feature, update the counter
    def add_bigram_tag_to_dict(self, tag_minus, word_tag):
        if word_tag in self.bigram_tag_dict:
            tag_per_tag_dict = self.bigram_tag_dict[word_tag]

            if tag_minus in tag_per_tag_dict:
                tag_per_tag_dict[tag_minus] += 1
            else:
                tag_per_tag_dict[tag_minus] = 1
                self.num_features += 1

        else:
            self.bigram_tag_dict[word_tag] = {tag_minus: 1}
            self.num_features += 1

    # Fill the 103 feature with a new seen feature
    # If already saw the feature, update the counter
    def add_trigram_tag_to_dict(self, tag_minus2, tag_minus, word_tag):
        if word_tag in self.trigram_tag_dict:
            two_tags_per_tag_dict = self.trigram_tag_dict[word_tag]

            if (tag_minus2, tag_minus) in two_tags_per_tag_dict:
                two_tags_per_tag_dict[(tag_minus2, tag_minus)] += 1
            else:
                two_tags_per_tag_dict[(tag_minus2, tag_minus)] = 1
                self.num_features += 1

        else:
            self.trigram_tag_dict[word_tag] = {(tag_minus2, tag_minus): 1}
            self.num_features += 1

    # Remove features that seen only one time
    # Recount the features number
    # Add a little safety check for "duplicate features"
    def get_frequented_features(self):
        counter = 0
        self.num_features = 0
        for word in self.word_to_tags_dict:
            tag_to_word_dict = self.word_to_tags_dict[word]
            for tag in tag_to_word_dict:
                if tag_to_word_dict[tag] > 1:
                    if (word, tag) not in self.features:
                        self.features[(word, tag)] = counter
                        self.features_by_index[counter] = ((word, tag), 100)
                        counter += 1
                        self.num_features += 1

        for tag in self.bigram_tag_dict:
            tag_to_tag_dict = self.bigram_tag_dict[tag]
            for tag_minus in tag_to_tag_dict:
                if tag_to_tag_dict[tag_minus] > 1:
                    if (tag_minus, tag) not in self.features:
                        self.features[(tag_minus, tag)] = counter
                        self.features_by_index[counter] = ((tag_minus, tag), 104)
                        counter += 1
                        self.num_features += 1

        for tag in self.trigram_tag_dict:
            two_tag_to_tag_dict = self.trigram_tag_dict[tag]
            for (tag_minus2, tag_minus) in two_tag_to_tag_dict:
                if two_tag_to_tag_dict[(tag_minus2, tag_minus)] > 1:
                    if ((tag_minus2, tag_minus), tag) not in self.features:
                        self.features[((tag_minus2, tag_minus), tag)] = counter
                        self.features_by_index[counter] = (((tag_minus2, tag_minus), tag), 103)
                        counter += 1
                        self.num_features += 1

    # TRY 1
    ################################################################

    # Return the result for v-dot-f on a given (history,tag) tuple
    def calculate_v_dot_f_for_tuple(self, data_tuple, v_vector):
        result = 0

        history = data_tuple[0]
        word_tag = data_tuple[1]
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]
        tag_minus = history[1]
        tag_minus2 = history[0]

        if (word, word_tag) in self.features:
            result += v_vector[self.features[(word, word_tag)]]
        if (tag_minus, word_tag) in self.features:
            result += v_vector[self.features[(tag_minus, word_tag)]]
        if ((tag_minus2, tag_minus), word_tag) in self.features:
            result += v_vector[self.features[((tag_minus2, tag_minus), word_tag)]]

        return result

    # Calculate the first sum of L(v)
    def func_l_part_one(self, v_vector):
        result = 0
        for data_tuple in self.history_tag_tuples:
            result += self.calculate_v_dot_f_for_tuple(data_tuple, v_vector)
        return result

    # Calculate the second sum of L(v)
    def func_l_part_two(self, v_vector):
        total_result = 0
        for data_tuple in self.history_tag_tuples:
            history = data_tuple[0]
            features_on_tuples_array = []

            for tag in self.tags:
                features_on_tuples_array.append(self.calculate_v_dot_f_for_tuple((history, tag), v_vector))

            exp_arr = np.exp(features_on_tuples_array)
            total_result += np.log2(sum(exp_arr))

        return total_result

    def func_l(self, v_vector):
        a = self.func_l_part_one(v_vector)
        b = self.func_l_part_two(v_vector)
        return a-b

    # END TRY 1
    #################################################################

    # TRY 2
    ################################################################

    def calculate_all_v_dot_f_for_tuple(self):
        for data_tuple in self.history_tag_tuples:
            for word_tag in self.tags:

                temp_arr = []

                history = data_tuple[0]
                word_index = history[3]
                split_sentence = (history[2]).split()
                word = split_sentence[word_index]
                tag_minus = history[1]
                tag_minus2 = history[0]

                if (word, word_tag) in self.features:
                    temp_arr.append(self.features[(word, word_tag)])
                if (tag_minus, word_tag) in self.features:
                    temp_arr.append(self.features[(tag_minus, word_tag)])
                if ((tag_minus2, tag_minus), word_tag) in self.features:
                    temp_arr.append(self.features[((tag_minus2, tag_minus), word_tag)])

                self.calculated_features[(history, word_tag)] = temp_arr

    def calculate_v_dot_f_for_tuple2(self, data_tuple, v_vector):
        result = 0
        features_num_arr = self.calculated_features[data_tuple]
        for index in features_num_arr:
            result += v_vector[index]
        return result

    def func_part1(self, v_vector):
        result = 0
        for data_tuple in self.history_tag_tuples:
            result += self.calculate_v_dot_f_for_tuple2(data_tuple, v_vector)
        return result

    def func_part2(self, v_vector):
        total_result = 0
        for data_tuple in self.history_tag_tuples:
            history = data_tuple[0]
            features_on_tuples_array = []

            for tag in self.tags:
                features_on_tuples_array.append(self.calculate_v_dot_f_for_tuple2((history, tag), v_vector))

            exp_arr = np.exp(features_on_tuples_array)
            total_result += np.log2(sum(exp_arr))

        return total_result

    def func_l_new(self, v_vector):
        a = self.func_part1(v_vector)
        b = self.func_part2(v_vector)
        return a-b

    # END TRY 2
    ##################################################################

    # Gradient
    ##################################################################

    def calculate_specific_gradient_first_sum(self, v_vector, index):
        result = 0
        feature_tuple = self.features_by_index[index]

        for data_tuple in self.history_tag_tuples:
            history = data_tuple[0]
            word_tag = data_tuple[1]
            word_index = history[3]
            split_sentence = (history[2]).split()
            word = split_sentence[word_index]
            tag_minus = history[1]
            tag_minus2 = history[0]

            if feature_tuple[1] == 100:
                feature_data = feature_tuple[0]
                feature_word = feature_data[0]
                feature_word_tag = feature_data[1]

                if feature_word == word and feature_word_tag == word_tag:
                    result += 1

            if feature_tuple[1] == 103:
                feature_data = feature_tuple[0]
                feature_tag = feature_data[1]
                tags_tuple = feature_data[0]
                feature_tag_minus2 = tags_tuple[0]
                feature_tag_minus = tags_tuple[1]

                if feature_tag_minus2 == tag_minus2 and feature_tag_minus == tag_minus and feature_tag == word_tag:
                    result += 1

            if feature_tuple[1] == 104:
                feature_data = feature_tuple[0]
                feature_tag_minus = feature_data[0]
                feature_tag = feature_data[1]

                if feature_tag_minus == tag_minus and feature_tag == word_tag:
                    result += 1

        return result

    def calculate_specific_gradient_second_sum(self, v_vector, index):
        result = 0
        return result

    def calculate_specific_gradient(self, v_vector, index):
        a = self.calculate_specific_gradient_first_sum(v_vector, index)
        b = self.calculate_specific_gradient_second_sum(v_vector, index)
        return a-b

startTime = datetime.now()
x = BasicTrainer()
x.get_history_tag_tuples()
print('\nFound ' + str(len(x.history_tag_tuples)) + ' different history_tag tuples')
print('Found ' + str(len(x.tags)) + ' different tags\n')
print('Searching for all seen features in data...')
x.fill_features_dicts()
print('Done features searching. Found ' + str(x.num_features) + ' different features\n')
print('Removing unfrequented features...')
x.get_frequented_features()
print('After optimization, only ' + str(x.num_features) + ' features left\n')

# print('Init v vector,')
# v = np.ones(shape=x.num_features, dtype=int)
# print('Calculating function L(v)...')
# res = minimize(x.func_l, x0=v, method='L-BFGS-B')
# print('The result is ' + str(res))

print('Calculate features on all (history,tag) options...')
x.calculate_all_v_dot_f_for_tuple()
print('Done calculating all possible features!\n')


v = np.ones(shape=x.num_features)
# print('Calculating function L(v)...')
# res = fmin_l_bfgs_b(x.func_l_new, x0=v_vector1, approx_grad=1)
# print('The result is ' + str(res))

print('Lets check first sum of gradient')
print(x.calculate_specific_gradient_first_sum(v, 20))


print('\nFROM BEGINNING TO NOW ONLY IN ' + str(datetime.now() - startTime) + ' SECONDS!')
