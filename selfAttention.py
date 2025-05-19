import torch
import math
sentence = torch.tensor(
    [0, # can
     7, # you
     1, # help
     2, # me
     5, # to
     6, # translate
     4, # this
     3] # sentence
    )

print(f'sentence: {sentence}')

# the following code will produce word embedding of our eight words
torch.manual_seed(123)
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()
embedded_sentence.shape
print(embedded_sentence.shape)
# print(embedded_sentence)
# Now we compute the dot product
omega = torch.empty(8,8)
for i, x_i in enumerate(embedded_sentence):
    for j, x_j in enumerate(embedded_sentence):
        omega[i, j] = torch.dot(x_i, x_j)

omega_mat = embedded_sentence.matmul(embedded_sentence.T)

# since the previous is a repetition, the following simply check if the result is the same
torch.allclose(omega_mat, omega)
print(torch.allclose(omega_mat, omega))

# computing the attention weight using pytorch's softmax function
import torch.nn.functional as F
attention_weights = F.softmax(omega, dim=1)
attention_weights.shape
print(attention_weights.shape)
# print(attention_weights)
print(attention_weights.sum(dim=1))

x_2 = embedded_sentence[1, :]
context_vec_2 = torch.zeros(x_2.shape)
for j in range(8):
    x_j = embedded_sentence[j, :]
    context_vec_2 += attention_weights[1, j] * x_j
print(f'original matrix of attention weights : {context_vec_2}')
# try with raw math to get this straight
# from the embedded sentence getting the scalars for the new matrix calculating the dot product between the two vectors
# for i in range(len(embedded_sentence)):
#     attention_weightss = []
#     for j in range(len(embedded_sentence)): 
#         a = 0
#         for index in range(len(embedded_sentence[j])):
#             a += embedded_sentence[j][index] * embedded_sentence[i][index]
#         attention_weightss.append(a)
#     sum_of_exponential = 0
#     s_m_values = []
#     for exp in range(len(attention_weightss)):
#         sum_of_exponential = sum_of_exponential + math.exp(attention_weightss[exp]) # math.exp is the equivalent to the concept e^exp
#     for index_att in range(len(attention_weightss)):
#         single_soft_valu = math.exp(attention_weightss[index_att]) / sum_of_exponential
#         s_m_values.append(single_soft_valu)
#     zi_vectors_prev = []

#     att_for_all_weight = [0] * len(embedded_sentence[i]) # initializing all weight at 0
#     for indAttW in range(len(s_m_values)):
#         att_for_spec_weight = []
#         for indInsWord in range(len(embedded_sentence[i])):
#             att_for_spec_weight.append(s_m_values[indAttW] * embedded_sentence[i][indInsWord])
#         for at_ind_sp in range(len(att_for_spec_weight)):
#             att_for_all_weight[at_ind_sp] += att_for_spec_weight[at_ind_sp]

#     print(f'Manual att for all weight: {att_for_all_weight}')

# We can achieve this more efficiently using matrix multiplication as follow 
context_vectors = torch.matmul(attention_weights, embedded_sentence)
print(torch.allclose(context_vec_2, context_vectors[1]))
