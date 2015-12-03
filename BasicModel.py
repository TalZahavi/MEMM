from datetime import datetime
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pickle


class BasicTrainer:
    def __init__(self):
        self.tags = set()
        self.history_tag_tuples = set()

        self.num_features = 0
        self.word_to_tags_dict = dict()
        self.bigram_tag_dict = dict()
        self.trigram_tag_dict = dict()

        self.features = dict()
        self.num_his_per_feature = dict()

        self.calculated_features = dict()
        self.tuples_per_feature = dict()  # For a given feature, return which possible (history,tag) return 1

        self.temp_e_fv_specific = dict()  # For a given (history,tag), return the calc for e^(vf(h,y))
        self.temp_e_fv = dict()  # For a given (history), return the calc for e^(vf(h,y_tag)) [y_tag = possible tags]

        self.iteration_start_time = (datetime.now(), 0)

        self.lambda_param = 50
        self.features_freq_param = 2
        self.v_vec = np.zeros(shape=1)

        self.last_l = 0

        self.freq_word_tag_dict = dict()
        self.word_to_freq_tags_dict = dict()
        self.num_tags = dict()

    # Fill the tags count tag
    def fill_tags_count_dict(self, tag):
        if tag in self.num_tags:
            self.num_tags[tag] += 1
        else:
            self.num_tags[tag] = 1

    # Count the number of tags for each word
    def fill_freq_tags_dict(self, word, tag):
        if word in self.freq_word_tag_dict:
            tags_dict = self.freq_word_tag_dict[word]

            if tag in tags_dict:
                tags_dict[tag] += 1
            else:
                tags_dict[tag] = 1
        else:
            self.freq_word_tag_dict[word] = {tag: 1}

    # Save the (word)->(tag) for tags that appear more then "x" times for a specific word
    def find_freq_tags(self):
        # for word in self.freq_word_tag_dict:
        #     for tag in self.freq_word_tag_dict[word]:
        #         if self.freq_word_tag_dict[word][tag] > 7:
        #             self.word_to_freq_tags_dict[word] = [tag]
        for word in self.freq_word_tag_dict:
            self.word_to_freq_tags_dict[word] = []
            for tag in self.freq_word_tag_dict[word]:
                self.word_to_freq_tags_dict[word].append(tag)



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

            self.fill_freq_tags_dict(split_word[0], split_word[1])
            self.fill_tags_count_dict(split_word[1])

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
                if tag_to_word_dict[tag] > self.features_freq_param:
                    if (word, tag) not in self.features:
                        self.features[(word, tag)] = counter
                        self.num_his_per_feature[counter] = tag_to_word_dict[tag]
                        counter += 1
                        self.num_features += 1

        for tag in self.bigram_tag_dict:
            tag_to_tag_dict = self.bigram_tag_dict[tag]
            for tag_minus in tag_to_tag_dict:
                if tag_to_tag_dict[tag_minus] > self.features_freq_param:
                    if (tag_minus, tag) not in self.features:
                        self.features[(tag_minus, tag)] = counter
                        self.num_his_per_feature[counter] = tag_to_tag_dict[tag_minus]
                        counter += 1
                        self.num_features += 1

        for tag in self.trigram_tag_dict:
            two_tag_to_tag_dict = self.trigram_tag_dict[tag]
            for (tag_minus2, tag_minus) in two_tag_to_tag_dict:
                if two_tag_to_tag_dict[(tag_minus2, tag_minus)] > self.features_freq_param:
                    if ((tag_minus2, tag_minus), tag) not in self.features:
                        self.features[((tag_minus2, tag_minus), tag)] = counter
                        self.num_his_per_feature[counter] = two_tag_to_tag_dict[(tag_minus2, tag_minus)]
                        counter += 1
                        self.num_features += 1

    # FUNCTION L(V)
    ################################################################

    # Calculation before the v_dot_f function
    # Fill a dict, that : (history_tag) -> (num features that apply for him)
    # Need to preform only once
    def calculate_all_dot_f_for_tuple(self):
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

                for num_feature in temp_arr:
                    if num_feature in self.tuples_per_feature:
                        self.tuples_per_feature[num_feature].append((history, word_tag))
                    else:
                        self.tuples_per_feature[num_feature] = [(history, word_tag)]

    # Calculate v_dot_f for a given tuple and a vector v
    def calculate_v_dot_f_for_tuple2(self, data_tuple, v_vector):
        result = 0
        features_num_arr = self.calculated_features[data_tuple]
        for index in features_num_arr:
            result += v_vector[index]
        return result

    # Calculate the first sum of the function L(v) (for a given vector)
    def func_part1(self, v_vector):
        result = 0
        for data_tuple in self.history_tag_tuples:
            result += self.calculate_v_dot_f_for_tuple2(data_tuple, v_vector)
        return result

    # Calculate the second sum of the function L(v) (for a given vector)
    def func_part2(self, v_vector):
        total_result = 0
        self.temp_e_fv = dict()
        self.temp_e_fv_specific = dict()
        for data_tuple in self.history_tag_tuples:
            history = data_tuple[0]
            e_sum = 0

            for tag in self.tags:
                temp_value = self.calculate_v_dot_f_for_tuple2((history, tag), v_vector)
                temp_value_e = np.exp(temp_value)
                e_sum += temp_value_e
                self.temp_e_fv_specific[(history, tag)] = temp_value_e

            self.temp_e_fv[history] = e_sum
            total_result += np.log(e_sum)

        return total_result

    # Calculate the function L(v) for a given vector
    def func_l_new(self, v_vector):
        self.iteration_start_time = (datetime.now(), self.iteration_start_time[1]+1)
        a = self.func_part1(v_vector)
        b = self.func_part2(v_vector)
        res = (-(a-b-self.lambda_calc(v_vector)))
        self.last_l = res
        return res

    def lambda_calc(self, v_vector):
        result = 0
        for i in range(0, len(v_vector)):
            result += v_vector[i] * v_vector[i]
        return (self.lambda_param/2)*result

    # END FUNCTION L(V)
    ##################################################################

    # Gradient
    ##################################################################

    # Calculate p(y|x;v) given tuple (x,y) and vector v
    # REMARK - The history have to be seen in the training data! do not use for viterbi!
    # REMARK 2 - This will be used ONLY AFTER L(V)!! so we can use self.temp_e_fv
    def calculate_p_given_tuple(self, data_tuple, v_vector):
        up = self.temp_e_fv_specific[data_tuple]
        down = self.temp_e_fv[data_tuple[0]]
        return up/down

    # Calculate the second sum of the gradient for a specific location
    def calculate_specific_gradient_second_sum(self, v_vector, index):
        result = 0

        tuples_for_feature = self.tuples_per_feature[index]
        for data_tuple in tuples_for_feature:
            result += self.calculate_p_given_tuple(data_tuple, v_vector)

        return result

    # Calculate a specific location in the gradient
    def calculate_specific_gradient(self, v_vector, index):
        a = self.num_his_per_feature[index]
        b = self.calculate_specific_gradient_second_sum(v_vector, index)
        return -(a-b-(self.lambda_param*v_vector[index]))

    # Get a gradient vector for a given v
    def get_gradient_vector(self, v_vector):
        grad_vector = np.zeros(shape=self.num_features)
        for index in range(0, self.num_features):
            # print('Start gradient at position ' + str(index))
            grad_vector[index] = self.calculate_specific_gradient(v_vector, index)
        print('The ' + str(self.iteration_start_time[1]) + ' iteration took ' +
              str(datetime.now() - self.iteration_start_time[0]))
        return grad_vector

    # END Gradient
    ########################################################################

    def train(self):
        self.get_history_tag_tuples()
        print('\nFound ' + str(len(self.history_tag_tuples)) + ' different history_tag tuples')
        print('Found ' + str(len(self.tags)) + ' different tags\n')

        self.find_freq_tags()

        print('Searching for all seen features in data...')
        self.fill_features_dicts()
        print('Done features searching. Found ' + str(self.num_features) + ' different features\n')
        print('Removing unfrequented features...')
        self.get_frequented_features()
        print('After optimization, only ' + str(self.num_features) + ' features left\n')

        print('Let me save some data for later...')
        pickle.dump(self.features, open("features_dict.p", "wb"), protocol=2)
        pickle.dump(self.tags, open("tags.p", "wb"), protocol=2)
        pickle.dump(self.word_to_freq_tags_dict, open("freq_tags.p", "wb"), protocol=2)
        print('Done saving data\n')

        print('Calculate features on all (history,tag) options - wait about 20 seconds...')
        self.calculate_all_dot_f_for_tuple()
        print('Done calculating all possible features!\n')

        v_vector_temp = np.zeros(shape=self.num_features)
        print('Lets try to find the best v... may take some time...(approximately 15 minutes)')

        start = datetime.now()
        res = fmin_l_bfgs_b(self.func_l_new, x0=v_vector_temp, fprime=self.get_gradient_vector)
        self.v_vec = res[0]
        print('Found the best v only in ' + str(datetime.now()-start) + '!! ITS A NEW RECORD!!!')

x = BasicTrainer()
x.train()
print('The value of the function target is ' + str(x.last_l))
print('The best v vector is saved to opt_v.npy')
np.save('opt_v', x.v_vec)







