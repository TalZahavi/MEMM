import numpy as np
import pickle
import Utilities
from datetime import datetime


class MemmInference:
    def __init__(self):
        self.v_vector = np.zeros(shape=1)
        self.features = dict()  # For a given feature number, return the feature data
        self.tags = set()  # All possible tags

        self.e_v_dot_f = dict()
        self.sum_e_v_dot_f = dict()

        self.freq_tags = dict()

        self.mode = 'Basic'

    # Load the data that was learned before
    def load_data(self, mode):
        if mode == 'Improved':
            self.mode = 'Improved'
            self.v_vector = np.load('improved_opt_v.npy')
            self.tags = pickle.load(open("improved_tags.p", "rb"))
            self.features = pickle.load(open("improved_features_dict.p", "rb"))
            self.freq_tags = pickle.load(open("improved_freq_tags.p", "rb"))
        elif mode == 'Basic':
            self.mode = 'Basic'
            self.v_vector = np.load('basic_opt_v.npy')
            self.tags = pickle.load(open("basic_tags.p", "rb"))
            self.features = pickle.load(open("basic_features_dict.p", "rb"))
            self.freq_tags = pickle.load(open("basic_freq_tags.p", "rb"))
        else:
            self.mode = 'Comp'
            self.v_vector = np.load('comp_opt_v.npy')
            self.tags = pickle.load(open("comp_tags.p", "rb"))
            self.features = pickle.load(open("comp_features_dict.p", "rb"))
            self.freq_tags = pickle.load(open("comp_freq_tags.p", "rb"))

    # Get all the possible tags at a position in the sentence
    def get_possible_tags_at_location(self, index, word):
        if index < 0:
            return ['*']
        else:
            if word in self.freq_tags:
                return self.freq_tags[word]
            return ['NNP', 'NNS', 'NN', 'VBD', 'VB', 'VBN', 'RB', 'JJ', 'VBD', 'NNPS', 'VBZ', 'VBP', 'CD', 'VBG']

    def get_num_features_for_given_tuple(self, data_tuple):
        num_f = []

        word_tag = data_tuple[1]
        history = data_tuple[0]
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]
        tag_minus = history[1]
        tag_minus2 = history[0]

        if (word, word_tag) in self.features:
            num_f.append(self.features[(word, word_tag)])
        if (tag_minus, word_tag) in self.features:
            num_f.append(self.features[(tag_minus, word_tag)])
        if ((tag_minus2, tag_minus), word_tag) in self.features:
            num_f.append(self.features[((tag_minus2, tag_minus), word_tag)])

        if self.mode == 'Improved' or self.mode == 'Comp':
            if (Utilities.get_suffix(word, 2), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_suffix(word, 2), word_tag)])
            if (Utilities.get_suffix(word, 3), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_suffix(word, 3), word_tag)])
            if (Utilities.get_prefix(word, 2), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_prefix(word, 2), word_tag)])

            if Utilities.check_number(word) and word_tag == 'CD':
                num_f.append(self.features['number', 'number'])
            if Utilities.check_capital(word, word_index) and word_tag == 'NNP':
                num_f.append(self.features['capital', 'capital'])
            if Utilities.check_bar(word) and word_tag == 'JJ':
                num_f.append(self.features['bar', 'bar'])

        # Features only for improved
        if self.mode == 'Improved':
            if (Utilities.get_suffix(word, 1), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_suffix(word, 1), word_tag)])
            if (Utilities.get_suffix(word, 4), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_suffix(word, 4), word_tag)])
            if (Utilities.get_prefix(word, 3), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_prefix(word, 3), word_tag)])
            if (Utilities.get_prefix(word, 4), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_prefix(word, 4), word_tag)])
            if (Utilities.get_prefix(word, 1), word_tag) in self.features:
                num_f.append(self.features[(Utilities.get_prefix(word, 1), word_tag)])

            if (word_tag, '') in self.features:
                num_f.append(self.features[(word_tag, '')])

        return num_f

    def calculate_e_v_dot_f(self, data_tuple, v_vec):
        if data_tuple in self.e_v_dot_f:
            return self.e_v_dot_f[data_tuple]

        num_features = self.get_num_features_for_given_tuple(data_tuple)
        sum_temp = 0
        for i in num_features:
            sum_temp += v_vec[i]

        result = np.exp(sum_temp)
        self.e_v_dot_f[data_tuple] = result
        return result

    def calculate_p(self, v_vec, u, v, t, k, sentence):
        new_tuple = ((t, u, sentence, k), v)

        up = self.calculate_e_v_dot_f(new_tuple, v_vec)

        history = new_tuple[0]
        if history in self.sum_e_v_dot_f:
            sum_down = self.sum_e_v_dot_f[history]
        else:
            sum_down = 0
            for tag in self.tags:
                sum_down += self.calculate_e_v_dot_f((history, tag), v_vec)
            self.sum_e_v_dot_f[history] = sum_down

        return up/sum_down

    # The main part of the viterbi algorithm
    def get_max_pi_and_arg_max_tag(self, k, viterbi_dic, u, v, sentence):
        max_arg_t = ''
        max_pi = 0

        sen_split = sentence.split()
        if k == 0 or k == 1:
            w_i = ''
        else:
            w_i = sen_split[k-2]

        for t in self.get_possible_tags_at_location(k-2, w_i):
            pi_val = (viterbi_dic[k-1])[(t, u)][0]
            q_val = self.calculate_p(self.v_vector, u, v, t, k, sentence)

            val = pi_val*q_val

            if val > max_pi:
                max_pi = val
                max_arg_t = t

        return max_pi, max_arg_t

    # Used for finding the last two tags in the sentence
    @staticmethod
    def get_max_tags_tuple(vit_dict):
        max_pi = 0
        result_tuple = ('', '')
        for (u, v) in vit_dict:
            if vit_dict[(u, v)][0] > max_pi:
                max_pi = vit_dict[(u, v)][0]
                result_tuple = (u, v)
        return result_tuple

    # Using VITERBI ALGORITHM
    def sentence_inference(self, sentence):
        viterbi_dict = dict()
        self.e_v_dot_f = dict()
        self.sum_e_v_dot_f = dict()
        viterbi_dict[-1] = {('*', '*'): (1, '*')}

        split_s = sentence.split()
        len_sentence = len(sentence.split())

        for k in range(0, len_sentence):
            temp_dict = dict()  # (u,v) -> (pi,bp)

            if k == 0:
                w_i = ''
            else:
                w_i = split_s[k-1]

            for u in self.get_possible_tags_at_location(k-1, w_i):
                for v in self.get_possible_tags_at_location(k, split_s[k]):
                    viterbi_tuple = self.get_max_pi_and_arg_max_tag(k, viterbi_dict, u, v, sentence)
                    temp_dict[(u, v)] = viterbi_tuple

            viterbi_dict[k] = temp_dict

        sentence_tags = dict()

        # Finding t_n-1 t_n tags
        tags_tuple = self.get_max_tags_tuple(viterbi_dict[len_sentence-1])
        sentence_tags[len_sentence-1] = tags_tuple[1]
        sentence_tags[len_sentence-2] = tags_tuple[0]

        # Finding the rest of the tags
        for k in range(len_sentence-3, -1, -1):
            (pi, bp) = viterbi_dict[k+2][(sentence_tags[k+1], sentence_tags[k+2])]
            sentence_tags[k] = bp

        return sentence_tags

    # Return a sentence with given tags
    @staticmethod
    def get_sentence_with_tags(sentence, tags_list):
        full_sen = []
        split_sentence = sentence.split()
        for i in range(0, len(split_sentence)):
            full_sen.append(split_sentence[i] + '_' + tags_list[i])
        return ' '.join(full_sen)

    # Return the (correct, num of words) of a given sentence with tags
    def get_acq_for_sentence_with_tags(self, sentence_with_tags):
        num_of_words = 0
        correct = 0

        temp_arr = []
        for word_tag in sentence_with_tags.split():
            temp_arr.append(word_tag.split('_')[0])
        clean_sen = ' '.join(temp_arr)
        my_guess = self.get_sentence_with_tags(clean_sen, self.sentence_inference(clean_sen))

        my_guess_arr = my_guess.split()
        for i, word in enumerate(sentence_with_tags.split()):
            if word == my_guess_arr[i]:
                correct += 1
            num_of_words += 1

        return correct, num_of_words

    # Go over the "test" (sentences with tags) and write the accuracy)
    def check_acq_for_file_with_tags(self, mode):
        if mode == 'Improved':
            self.load_data('Improved')
        elif mode == 'Basic':
            self.load_data('Basic')
        elif mode == 'Comp':
            self.load_data('Comp')
        else:
            print('PLEASE CHOOSE IMPROVED\BASIC\COMP!')
            return
        num_sentence = 0
        sum_correct = 0
        total = 0
        with open('test.wtag', 'r') as f:
            for line in f:
                (correct, num) = self.get_acq_for_sentence_with_tags(line)
                sum_correct += correct
                total += num

                num_sentence += 1
                print('Done sentence number ' + str(num_sentence))
                print('The accuracy by now is ' + str(round((sum_correct/total)*100, 2)) + '%')
        f.close()
        print('\nThe final accuracy is ' + str(round((sum_correct/total)*100, 2)) + '%')

    # THIS METHOD CREATE THE FILES FOR THE COMPETITION
    def inference_comp(self, mode):
        if mode == 'Improved':
            self.load_data('Improved')
        elif mode == 'Basic':
            self.load_data('Basic')
        elif mode == 'Comp':
            self.load_data('Comp')
        else:
            print('PLEASE CHOOSE IMPROVED\BASIC\COMP!')
            return

        if mode == 'Basic':
            f_w = open('comp_m1_200279040.wtag', 'w+')
        else:
            f_w = open('comp_m2_200279040.wtag', 'w+')
        counter_sen = 0
        print('Start working... wait a minute...')
        with open('comp.words', 'r') as f:
            for line in f:
                counter_sen += 1
                my_guess = self.get_sentence_with_tags(line, self.sentence_inference(line))
                f_w.write(my_guess)
                if counter_sen != 1000:
                    f_w.write('\n')
        f.close()
        f_w.close()

y = MemmInference()
start = datetime.now()

# y.check_acq_for_file_with_tags('Comp')
y.inference_comp('Comp')
y.inference_comp('Basic')


print('\nDone in ' + str(datetime.now()-start))
