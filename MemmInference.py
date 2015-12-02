import numpy as np


class MemmInference:
    def __init__(self):
        self.v_vector = np.ones(shape=1)
        self.features = dict()  # For a given feature number, return the feature data
        self.tags = dict()  # All possible tags

    # Load the data that was learned before
    def load_data(self):
        self.v_vector = np.load('opt_v.npy')
        # TODO: Load Features Dict From Basic Model
        # TODO: Load Tags Set From Basic Model

    # Get all the possible tags at a position in the sentence
    def get_possible_tags_at_location(self, index):
        if index < 0:
            return ['*']
        else:
            return self.tags

    def calculate_p(self):
        return 0
        # TODO: FILL IN

    # The main part of the viterbi algorithm
    def get_max_pi_and_arg_max_tag(self, k, viterbi_dic, u, v):
        max_arg_t = ''
        max_pi = 0

        for t in self.get_possible_tags_at_location(k-2):
            pi_val = viterbi_dic[k-1][(t, u)]
            q_val = self.calculate_p()
            val = pi_val * q_val

            if val >= max_pi:
                max_pi = val
                max_arg_t = t

        return max_pi, max_arg_t

    # Used for finding the last two tags in the sentence
    @staticmethod
    def get_max_tags_tuple(vit_dict):
        max_pi = 0
        result_tuple = ('', '')
        for (u, v) in vit_dict:
            if vit_dict[(u, v)][0] >= max_pi:
                max_pi = vit_dict[(u, v)][0]
                result_tuple = (u, v)
        return result_tuple

    # Using VITERBI ALGORITHM
    def sentence_inference(self, sentence, v_vec):
        viterbi_dict = dict()
        viterbi_dict[-1] = {('*', '*'): (1, '*')}

        for k in range(0, len(sentence)):
            temp_dict = dict()  # (u,v) -> (pi,bp)

            for u in self.get_possible_tags_at_location(k-1):
                for v in self.get_possible_tags_at_location(k):
                    viterbi_tuple = self.get_max_pi_and_arg_max_tag(k, viterbi_dict, u, v)
                    temp_dict[(u, v)] = viterbi_tuple

            viterbi_dict[k] = temp_dict

        sentence_tags = []

        # Finding t_n-1 t_n tags
        tags_tuple = self.get_max_tags_tuple(viterbi_dict[len(sentence)-1])
        sentence_tags[len(sentence)-1] = tags_tuple[1]
        sentence_tags[len(sentence)-2] = tags_tuple[0]

        # Finding the rest of the tags
        for k in range(len(sentence)-3, 0, -1):
            (pi, bp) = viterbi_dict[k+2][(sentence_tags[k+1], sentence_tags[k+2])]
            sentence_tags[k] = bp

        return sentence_tags

