class Feature100:
    word = ''
    tag = ''

    def __init__(self, m_word, m_tag):
        self.word = m_word
        self.tag = m_tag

    def calc(self, training_set_tuple):
        if self.tag != training_set_tuple[1]:
            return 0
        history = training_set_tuple[0]
        current_sentence = history[2]
        current_index = history[3]
        temp_sentence = current_sentence.split(' ')
        if self.word != temp_sentence[current_index]:
            return 0
        return 1


class Feature103:
    tag1 = ''
    tag2 = ''
    tag3 = ''

    def __init__(self, m_tag1, m_tag2, m_tag3):
        self.tag1 = m_tag1
        self.tag2 = m_tag2
        self.tag3 = m_tag3

    def calc(self, training_set_tuple):
        if self.tag3 != training_set_tuple[1]:
            return 0
        history = training_set_tuple[0]
        if history[0] != self.tag1:
            return 0
        if history[1] != self.tag2:
            return 0
        return 1


class Feature104:
    tag1 = ''
    tag2 = ''

    def __init__(self, m_tag1, m_tag2):
        self.tag1 = m_tag1
        self.tag2 = m_tag2

    def calc(self, training_set_tuple):
        if self.tag2 != training_set_tuple[1]:
            return 0
        history = training_set_tuple[0]
        if history[1] != self.tag1:
            return 0
        return 1
