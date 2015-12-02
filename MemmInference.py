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

    # Using VITERBI ALGORITHM
    def sentence_inference(self, sentence, v_vec):
        viterbi_dict = dict()
        sentence_tags = []

        for k in range(1, len(sentence)+1):
            for u in self.get_possible_tags_at_location(k-1):
                for v in self.get_possible_tags_at_location(k):
                    pass
                    #TODO : HERE NEED TO FILL PAI AND BP

        # Finding t_n-1 t_n tags

        # Finding the rest of the tags
        for k in range(len(sentence)-2, 1, -1):
            (pi, bp) = viterbi_dict[k+2][(sentence_tags[k+1],sentence_tags[k+2])]
            sentence_tags[k] = bp



    # def temp(self):
    # #variables
    # viterbi_dict = dict()
    # sentence_str = ' '.join(sentence)
    # print ("sentence: ", sentence_str)
    # #input:
    #     #a sentence w1....wn
    #     #log-linear model that provides q(v|t,u,w[1:n],i) for any t,u,v, for any i from 1 to n #Todo check how I get q
    # #initialization: set pi(0,*,*) = 1
    # viterbi_dict[0] = {('*', '*'): (1, '*')}
    # #algo:
    # n = len(sentence)
    # for k in range(1, n+1):
    #     print ("sentence: ", sentence[k-1])
    #     viterbi_dict[k] = dict()
    #
    #     #optional optimization - don't calculate un possible sequence ('dt','v'), (',','.')....
    #     #print ("tag_group ", k-1, tag_group(k-1), "tag_group: ", k, tag_group(k))
    #     for u in tag_group(k-1):
    #         for v in tag_group(k):
    #             max_prob = 0
    #             max_tag = ''
    #             for t in tag_group(k-2):
    #                 temp_dict_k = viterbi_dict[k-1]
    #                 temp_tuple = temp_dict_k[(t, u)]
    #                 data = ((t, u, sentence_str, k-1), v)
    #                 p_yx = x.calculate_p_y_x(data, v_vector)
    #                 #print ("prop for history= ",(t, u, sentence_str, k-1), "and tag: ", v , "is: ", p_yx)
    #                 temp_prob = temp_tuple[0] * p_yx
    #                 if max_prob <= temp_prob:
    #                     max_prob = temp_prob
    #                     max_tag = t
    #             temp_dict = viterbi_dict[k]
    #             temp_dict[(u, v)] = (max_prob, max_tag)
    #
    # #print ("viterbi_dict", viterbi_dict)
    # sentence_tags = dict()
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
    #
    # #finding t1....tn-2
    # for k in range(n-2,0,-1):
    #     tuple_value = viterbi_dict[k+2][(sentence_tags[k+1],sentence_tags[k+2])]
    #     sentence_tags[k] = tuple_value[1]
    # #return tags for the input sentece
    # print ("sentence tag: ", sentence_tags)
    # return sentence_tags
