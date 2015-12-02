import numpy as np


class MemmInference:
    def __init__(self):
        self.v_vector = np.ones(shape=1)
        self.features = dict()  # For a given feature number, return the feature data
        self.tags = dict()  # All possible tags


    def load_data(self):
        self.v_vector = np.load('opt_v.npy')
        # TODO: Load Features Dict From Basic Model
        # TODO: Load Tags Set From Basic Model

    def get_possible_tags_at_location(self, index):
        if index < 1:
            return ['*']
        else:
            return self.tags

    def calculate_p(self):
        pass
        # TODO: FILL IN

    def get_max_pi_and_arg_max_tag(self):
        return (0,'')
        # TODO: FILL IN!

        # for t in tag_group(k-2):
        #             temp_dict_k = viterbi_dict[k-1]
        #             temp_tuple = temp_dict_k[(t, u)]
        #             data = ((t, u, sentence_str, k-1), v)
        #             p_yx = x.calculate_p_y_x(data, v_vector)
        #             #print ("prop for history= ",(t, u, sentence_str, k-1), "and tag: ", v , "is: ", p_yx)
        #             temp_prob = temp_tuple[0] * p_yx
        #             if max_prob <= temp_prob:
        #                 max_prob = temp_prob
        #                 max_tag = t


    # Using VITERBI ALGORITHM
    def sentence_inference(self, sentence, v_vec):
        viterbi_dict = dict()
        #  Init

        for k in range(1, len(sentence)+1):
            temp_dict = dict()  # (u,v) -> (pi,bp)

            for u in self.get_possible_tags_at_location(k-1):
                for v in self.get_possible_tags_at_location(k):
                    #TODO : HERE NEED TO FILL PAI AND BP
                    viterbi_tuple = self.get_max_pi_and_arg_max_tag()
                    temp_dict[(u, v)] = viterbi_tuple

            viterbi_dict[k] = temp_dict

        sentence_tags = []
        # Finding t_n-1 t_n tags

        # Finding the rest of the tags
        for k in range(len(sentence)-2, 1, -1):
            (pi, bp) = viterbi_dict[k+2][(sentence_tags[k+1],sentence_tags[k+2])]
            sentence_tags[k] = bp



    # def temp(self):
    #
    # viterbi_dict[0] = {('*', '*'): (1, '*')}
    #
    # # finding tn-1 tn
    # n_dic = viterbi_dict[n]
    # max_n_prob = 0
    # max_n_tags = ('', '')
    # for tuple_tags in n_dic:
    #     tuple_values = n_dic[tuple_tags]
    #     if max_n_prob <= tuple_values[0]:
    #         max_n_prob = tuple_values[0]
    #         max_n_tags = tuple_tags
    # sentence_tags[n-1] = max_n_tags[0]
    # sentence_tags[n] = max_n_tags[1]


