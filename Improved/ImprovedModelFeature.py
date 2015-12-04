feature_105_dict = dict()
feature_101_dict = dict()
feature_102_dict = dict()


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
        print('checking ' + word_chars[i])
        print('with ' + prefix[prefix_index])
        if word_chars[i] != prefix[prefix_index]:
            return False
        prefix_index += 1
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
            add_suffix_ing_to_dict(word)
        if check_prefix(word, 3, 'pre') and word_tag == 'NN':
            add_prefix_pre_to_dict(word)


# Fill the 101 feature with a new seen feature
# If already saw the feature, update the counter
def add_suffix_ing_to_dict(word):
    if word in feature_101_dict:
        feature_101_dict[word] += 1
    else:
        feature_101_dict[word] = 1


# Fill the 102 feature with a new seen feature
# If already saw the feature, update the counter
def add_prefix_pre_to_dict(word):
    if word in feature_102_dict:
        feature_102_dict[word] += 1
    else:
        feature_102_dict[word] = 1


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

    for tag in feature_105_dict:
        if feature_105_dict[tag] > 7:
            if (tag, tag) not in self.features:
                self.features[(tag, tag)] = counter
                self.num_his_per_feature[counter] = feature_105_dict[tag]
                counter += 1
                self.num_features += 1

    for word in feature_101_dict:
        if feature_101_dict[word] > 3:
            if (word, 'ing') not in self.features:
                self.features[(word, 'ing')] = counter
                self.num_his_per_feature[counter] = feature_101_dict[word]
                counter += 1
                self.num_features += 1

    for word in feature_102_dict:
        if feature_102_dict[word] > 3:
            if (word, 'pre') not in self.features:
                self.features[(word, 'pre')] = counter
                self.num_his_per_feature[counter] = feature_102_dict[word]
                counter += 1
                self.num_features += 1


