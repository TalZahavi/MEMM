from datetime import datetime


class MemmTrainer:
    def __init__(self):
        self.history_tag_tuples = set()
        self.word_to_tags_dict = dict()
        self.bigram_tag_dict = dict()
        self.trigram_tag_dict = dict()

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
                history_num_list_per_last_two_tags =  last_two_tags_per_tag_dict[(tag_minus_two, tag_minus)]
                history_num_list_per_last_two_tags.append(history_num)
            else:
                last_two_tags_per_tag_dict[(tag_minus_two, tag_minus)] = [history_num]
        else:
            self.trigram_tag_dict[tag] = {(tag_minus_two, tag_minus): [history_num]}



startTime = datetime.now()
###############################
trainer = MemmTrainer()
trainer.train_history_tag_tuples()
trainer.train_dicts()

###############################
print(datetime.now() - startTime)

