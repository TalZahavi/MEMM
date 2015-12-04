from datetime import datetime
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from BASIC import BasicModelFeatures
from Improved import ImprovedModelFeature


class Trainer:
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
        self.v_vec = np.zeros(shape=1)

        self.last_l = 0

        self.freq_word_tag_dict = dict()
        self.word_to_freq_tags_dict = dict()

        self.improved_mode = False

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

    # Save the (word)->(tags) for every word
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
        if self.improved_mode:
            ImprovedModelFeature.fill_features_dicts(self)
        else:
            BasicModelFeatures.fill_features_dicts(self)

    # Remove features that seen only one time
    # Recount the features number
    # Add a little safety check for "duplicate features"
    def get_frequented_features(self):
        if self.improved_mode:
            ImprovedModelFeature.get_frequented_features(self)
        else:
            BasicModelFeatures.get_frequented_features(self)

    # FUNCTION L(V)
    ################################################################

    # Calculation before the v_dot_f function
    # Fill a dict, that : (history_tag) -> (num features that apply for him)
    # Need to preform only once
    def calculate_all_dot_f_for_tuple(self):
        if self.improved_mode:
            ImprovedModelFeature.calculate_all_dot_f_for_tuple(self)
        else:
            BasicModelFeatures.calculate_all_dot_f_for_tuple(self)

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

    def train(self, improved_mode):

        if improved_mode == 'Improved':
            self.improved_mode = True
            print('\nStarted IMPROVED mode training')
        elif improved_mode == 'Basic':
            self.improved_mode = False
            print('\nStarted BASIC mode training')
        else:
            print('You need to choose improved or basic mode!')
            return

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

        print('Calculate features on all (history,tag) options - wait about 20 seconds...')
        self.calculate_all_dot_f_for_tuple()
        print('Done calculating all possible features!\n')

        v_vector_temp = np.zeros(shape=self.num_features)
        print('Lets try to find the best v... may take some time...(approximately 18 minutes)')

        start = datetime.now()
        res = fmin_l_bfgs_b(self.func_l_new, x0=v_vector_temp, fprime=self.get_gradient_vector)
        self.v_vec = res[0]
        print(self.last_l)
        print('Found the best v only in ' + str(datetime.now()-start) + '!! ITS A NEW RECORD!!!')
        print(res[2])
        print('\nSaving the data...')
        if self.improved_mode:
            ImprovedModelFeature.save_data_to_files(self)
        else:
            BasicModelFeatures.save_data_to_files(self)
        print('ALL DONE!')

x = Trainer()
x.train('Improved')







