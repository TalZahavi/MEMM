from datetime import datetime


class BasicTrainer:
    def __init__(self):
        self.tags = set()
        self.history_tag_tuples = set()

        self.num_features = 0
        self.word_to_tags_dict = dict()
        self.bigram_tag_dict = dict()
        self.trigram_tag_dict = dict()

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
                num_times_features_seen = tags_per_word_dict[word_tag]
                num_times_features_seen += 1
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
                num_times_features_seen = tag_per_tag_dict[tag_minus]
                num_times_features_seen += 1
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
                num_times_features_seen = two_tags_per_tag_dict[(tag_minus2, tag_minus)]
                num_times_features_seen += 1
            else:
                two_tags_per_tag_dict[(tag_minus2, tag_minus)] = 1
                self.num_features += 1

        else:
            self.trigram_tag_dict[word_tag] = {(tag_minus2, tag_minus): 1}
            self.num_features += 1

startTime = datetime.now()
x = BasicTrainer()
x.get_history_tag_tuples()
print('')
print('Found ' + str(len(x.history_tag_tuples)) + ' different history_tag tuples')
print('Found ' + str(len(x.tags)) + ' different tags')
print('')
print('Searching for all seen features in data...')
x.fill_features_dicts()
print('Done features searching. Found ' + str(x.num_features) + ' different features')
print('')
print('FROM BEGINNING TO NOW ONLY ' + str(datetime.now() - startTime) + ' SECONDS!')