import pickle
import numpy as np


feature_number_dict = dict()  # Check if the word represent a number, and the tag is CD
feature_capital_dict = dict()  # Check if the word starts with a capital letter, not the first word, and the tag is NNP
feature_bar_dict = dict()  # Check if the word has "-" in the middle, and the tag is JJ

feature_suffix_dict = dict()  # Holds the tags for suffix (currently only length 3 suffix)
feature_prefix_dict = dict()  # Holds the tags for prefix (currently only length 2 prefix)


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


# Auxiliary function - check if a given word start with a capital, and not the first word of the sentence
def check_capital(word, index):
    if index == 0:
        return False
    return word[0].isupper()


# Auxiliary function - check if the word start\end with a letter, and there's "-" in the middle
def check_bar(word):
    if not word[0].isalpha() or not word[len(word)-1].isalpha():
        return False
    return '-' in word


# Auxiliary function - getting the last n chars
def get_suffix(word, num):
    return word[-num:]


# Auxiliary function - getting the first n chars
def get_prefix(word, num):
    return word[0:num]


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

        if check_number(word) and word_tag == 'CD':
            add_number_to_dict(self)
        if check_capital(word, word_index) and word_tag == 'NNP':
            add_capital_to_dict(self)
        if check_bar(word) and word_tag == 'JJ':
            add_bar_to_dict(self)

        if (len(word)) > 3:
            add_general_suffix_to_dict(self, word, word_tag)
            add_general_prefix_to_dict(self, word, word_tag)


# Fill the general suffix feature (Feature 101)
def add_general_suffix_to_dict(self, word, word_tag):
    suffix_3 = get_suffix(word, 3)

    if suffix_3 in feature_suffix_dict:
        suffix_tags_dict = feature_suffix_dict[suffix_3]

        if word_tag in suffix_tags_dict:
            suffix_tags_dict[word_tag] += 1
        else:
            suffix_tags_dict[word_tag] = 1
            self.num_features += 1
    else:
        feature_suffix_dict[suffix_3] = {word_tag: 1}
        self.num_features += 1

    suffix_2 = get_suffix(word, 2)

    if suffix_2 in feature_suffix_dict:
        suffix_tags_dict = feature_suffix_dict[suffix_2]

        if word_tag in suffix_tags_dict:
                suffix_tags_dict[word_tag] += 1
        else:
            suffix_tags_dict[word_tag] = 1
            self.num_features += 1
    else:
        feature_suffix_dict[suffix_2] = {word_tag: 1}
        self.num_features += 1


# Fill the general prefix feature (Feature 102)
def add_general_prefix_to_dict(self, word, word_tag):
    prefix_2 = get_prefix(word, 2)

    if prefix_2 in feature_prefix_dict:
        prefix_tags_dict = feature_prefix_dict[prefix_2]

        if word_tag in prefix_tags_dict:
            prefix_tags_dict[word_tag] += 1
        else:
            prefix_tags_dict[word_tag] = 1
            self.num_features += 1

    else:
        feature_prefix_dict[prefix_2] = {word_tag: 1}
        self.num_features += 1


# Fill the number feature
def add_number_to_dict(self):
    if 'number' in feature_number_dict:
        self.num_features += 1
        feature_number_dict['number'] += 1
    else:
        self.num_features += 1
        feature_number_dict['number'] = 1


# Fill the capital feature
def add_capital_to_dict(self):
    if 'capital' in feature_capital_dict:
        self.num_features += 1
        feature_capital_dict['capital'] += 1
    else:
        self.num_features += 1
        feature_capital_dict['capital'] = 1


def add_bar_to_dict(self):
    if 'bar' in feature_bar_dict:
        self.num_features += 1
        feature_bar_dict['bar'] += 1
    else:
        self.num_features += 1
        feature_bar_dict['bar'] = 1


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

    for suffix in feature_suffix_dict:
        tags_for_suffix_dict = feature_suffix_dict[suffix]
        for tag_for_sux in tags_for_suffix_dict:

            if len(suffix) == 2:
                if tags_for_suffix_dict[tag_for_sux] > 1000:
                    if (suffix, tag_for_sux) not in self.features:
                        self.features[(suffix, tag_for_sux)] = counter
                        self.num_his_per_feature[counter] = tags_for_suffix_dict[tag_for_sux]
                        counter += 1
                        self.num_features += 1
            else:
                if tags_for_suffix_dict[tag_for_sux] > 830:
                    if (suffix, tag_for_sux) not in self.features:
                        self.features[(suffix, tag_for_sux)] = counter
                        self.num_his_per_feature[counter] = tags_for_suffix_dict[tag_for_sux]
                        counter += 1
                        self.num_features += 1

    for prefix in feature_prefix_dict:
        tags_for_prefix_dict = feature_prefix_dict[prefix]
        for tag_for_pre in tags_for_prefix_dict:
            if tags_for_prefix_dict[tag_for_pre] > 700:
                if (prefix, tag_for_pre) not in self.features:
                    self.features[(prefix, tag_for_pre)] = counter
                    self.num_his_per_feature[counter] = tags_for_prefix_dict[tag_for_pre]
                    counter += 1
                    self.num_features += 1

    self.features[('number', 'number')] = counter
    self.num_his_per_feature[counter] = feature_number_dict['number']
    counter += 1
    self.num_features += 1

    self.features[('capital', 'capital')] = counter
    self.num_his_per_feature[counter] = feature_capital_dict['capital']
    counter += 1
    self.num_features += 1

    self.features[('bar', 'bar')] = counter
    self.num_his_per_feature[counter] = feature_bar_dict['bar']
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
            suffix_3 = get_suffix(word, 3)
            suffix_2 = get_suffix(word, 2)
            prefix_2 = get_prefix(word, 2)

            if (word, word_tag) in self.features:
                temp_arr.append(self.features[(word, word_tag)])
            if (tag_minus, word_tag) in self.features:
                temp_arr.append(self.features[(tag_minus, word_tag)])
            if ((tag_minus2, tag_minus), word_tag) in self.features:
                temp_arr.append(self.features[((tag_minus2, tag_minus), word_tag)])

            if (suffix_3, word_tag) in self.features:
                temp_arr.append(self.features[(suffix_3, word_tag)])
            if (suffix_2, word_tag) in self.features:
                temp_arr.append(self.features[(suffix_2, word_tag)])
            if (prefix_2, word_tag) in self.features:
                temp_arr.append(self.features[(prefix_2, word_tag)])
            if check_number(word) and word_tag == 'CD':
                temp_arr.append(self.features[('number', 'number')])
            if check_capital(word, word_index) and word_tag == 'NNP':
                temp_arr.append(self.features[('capital', 'capital')])
            if check_bar(word) and word_tag == 'JJ':
                temp_arr.append(self.features[('bar', 'bar')])

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
