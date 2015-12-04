import pickle
import numpy as np


feature_105_dict = dict()
feature_101_dict = dict()  # Ends with 'ing', and tag is 'VBG'
feature_102_dict = dict()  # Starts with 'pre', and the is 'NN'
feature_number_dict = dict()  # Check if the word represent a number, and the tag is CD


# Auxiliary function - check if a given word ends with the given suffix (num_char is the length of the suffix)
def check_suffix(word, num_char, suffix):
    if word == '':
        return False
    word_chars = list(word)
    suffix_index = num_char
    for i in range(len(word)-1, len(word)-num_char-1, -1):
        if word_chars[i] != suffix[suffix_index-1]:
            return False
        suffix_index -= 1
    return True


# Auxiliary function - check if a given word starts with the given prefix (num_char is the length of the prefix)
def check_prefix(word, num_char, prefix):
    if word == '':
        return False
    word_chars = list(word)
    prefix_index = 0
    for i in range(0, num_char):
        if word_chars[i] != prefix[prefix_index]:
            return False
        prefix_index += 1
    return True


# Auxiliary function - check if a given word represent a number
def check_number(word):
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    found_one_digit = False
    for c in list(word):
        if c in nums:
            found_one_digit = True
    if not found_one_digit:
        return False
    for c in list(word):
        if c != '.' and c != ',' and c not in nums:
            return False
    return True


# Fill the features dicts according to the seen data
def fill_features_dicts(self):
    for (history, word_tag) in self.history_tag_tuples:
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]
        tag_minus2 = history[0]
        tag_minus = history[1]

        add_word_tag_to_dict(self, word, word_tag)
        add_bigram_tag_to_dict(self, tag_minus, word_tag)
        add_trigram_tag_to_dict(self, tag_minus2, tag_minus, word_tag)

        if check_suffix(word, 3, 'ing') and word_tag == 'VBG':
            add_suffix_ing_to_dict()
        if check_prefix(word, 3, 'pre') and word_tag == 'NN':
            add_prefix_pre_to_dict()
        if check_number(word) and word_tag == 'CD':
            add_number_to_dict()


def add_number_to_dict():
    if 'number' in feature_number_dict:
        feature_number_dict['number'] += 1
    else:
        feature_number_dict['number'] = 1


# Fill the 101 feature with the number of times the feature seen
def add_suffix_ing_to_dict():
    if 'ing' in feature_101_dict:
        feature_101_dict['ing'] += 1
    else:
        feature_101_dict['ing'] = 1


# Fill the 102 feature with the number of times the feature seen
def add_prefix_pre_to_dict():
    if 'pre' in feature_102_dict:
        feature_102_dict['pre'] += 1
    else:
        feature_102_dict['pre'] = 1


# Fill the 105 feature with a new seen feature
# If already saw the feature, update the counter
def add_tag_to_dict(tag):
    if tag in feature_105_dict:
        feature_105_dict[tag] += 1
    else:
        feature_105_dict[tag] = 1


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


# Remove features that seen only few times
# Recount the features number
# Add a little safety check for "duplicate features"
# Gives the features a specific number
def get_frequented_features(self):
    counter = 0
    self.num_features = 0
    for word in self.word_to_tags_dict:
        tag_to_word_dict = self.word_to_tags_dict[word]
        for tag in tag_to_word_dict:
            if tag_to_word_dict[tag] > 1:
                if (word, tag) not in self.features:
                    self.features[(word, tag)] = counter
                    self.num_his_per_feature[counter] = tag_to_word_dict[tag]
                    counter += 1
                    self.num_features += 1

    for tag in self.bigram_tag_dict:
        tag_to_tag_dict = self.bigram_tag_dict[tag]
        for tag_minus in tag_to_tag_dict:
            if tag_to_tag_dict[tag_minus] > 3:
                if (tag_minus, tag) not in self.features:
                    self.features[(tag_minus, tag)] = counter
                    self.num_his_per_feature[counter] = tag_to_tag_dict[tag_minus]
                    counter += 1
                    self.num_features += 1

    for tag in self.trigram_tag_dict:
        two_tag_to_tag_dict = self.trigram_tag_dict[tag]
        for (tag_minus2, tag_minus) in two_tag_to_tag_dict:
            if two_tag_to_tag_dict[(tag_minus2, tag_minus)] > 2:
                if ((tag_minus2, tag_minus), tag) not in self.features:
                    self.features[((tag_minus2, tag_minus), tag)] = counter
                    self.num_his_per_feature[counter] = two_tag_to_tag_dict[(tag_minus2, tag_minus)]
                    counter += 1
                    self.num_features += 1

    # for tag in feature_105_dict:
    #     if feature_105_dict[tag] > 20:
    #         if (tag, tag) not in self.features:
    #             self.features[(tag, tag)] = counter
    #             self.num_his_per_feature[counter] = feature_105_dict[tag]
    #             counter += 1
    #             self.num_features += 1

    self.features[('ing', 'ing')] = counter
    self.num_his_per_feature[counter] = feature_101_dict['ing']
    counter += 1
    self.num_features += 1
    self.features[('pre', 'pre')] = counter
    self.num_his_per_feature[counter] = feature_102_dict['pre']
    counter += 1
    self.num_features += 1
    self.features[('number', 'number')] = counter
    self.num_his_per_feature[counter] = feature_number_dict['number']
    counter += 1
    self.num_features += 1


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

            # if (word_tag, word_tag) in self.features:
            #     temp_arr.append(self.features[(word_tag, word_tag)])
            if check_suffix(word, 3, 'ing') and word_tag == 'VBG':
                temp_arr.append(self.features[('ing', 'ing')])
            if check_prefix(word, 3, 'pre') and word_tag == 'NN':
                temp_arr.append(self.features[('pre', 'pre')])
            if check_number(word) and word_tag == 'CD':
                temp_arr.append(self.features[('number', 'number')])

            self.calculated_features[(history, word_tag)] = temp_arr

            for num_feature in temp_arr:
                if num_feature in self.tuples_per_feature:
                    self.tuples_per_feature[num_feature].append((history, word_tag))
                else:
                    self.tuples_per_feature[num_feature] = [(history, word_tag)]


def save_data_to_files(self):
    pickle.dump(self.features, open("improved_features_dict.p", "wb"), protocol=2)
    pickle.dump(self.tags, open("improved_tags.p", "wb"), protocol=2)
    pickle.dump(self.word_to_freq_tags_dict, open("improved_freq_tags.p", "wb"), protocol=2)
    np.save('improved_opt_v', self.v_vec)


