import numpy as np
import pickle


# Fill the features dicts according to the seen data
def fill_features_dicts(self):
    for (history, word_tag) in self.history_tag_tuples:
        word_index = history[3]
        split_sentence = (history[2]).split()
        word = split_sentence[word_index]

        add_word_tag_to_dict(self, word, word_tag)
        add_bigram_tag_to_dict(self, history[1], word_tag)
        add_trigram_tag_to_dict(self, history[0], history[1], word_tag)


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


def save_data_to_files(self):
    pickle.dump(self.features, open("basic_features_dict.p", "wb"), protocol=2)
    pickle.dump(self.tags, open("basic_tags.p", "wb"), protocol=2)
    pickle.dump(self.word_to_freq_tags_dict, open("basic_freq_tags.p", "wb"), protocol=2)
    np.save('basic_opt_v', self.v_vec)
