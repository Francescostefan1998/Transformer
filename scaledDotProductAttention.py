
import torch
import math
import torch.nn.functional as F # Add this line
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
torch.manual_seed(123)
d = embedded_sentence.shape[1]
# randomize the three matrices of weights
U_query = torch.rand(d, d)
U_key = torch.rand(d, d)
U_value = torch.rand(d, d)

# getting x2 which is at the index of 1
x_2 = embedded_sentence[1]
# finally computing q2
query_2 = U_query.matmul(x_2)
# computing k2 and v2
key_2 = U_key.matmul(x_2)
value_2 = U_value.matmul(x_2)
# but to compute this we actually need the key anv value sequences for all other input elements
keys = U_key.matmul(embedded_sentence.T).T
values = U_value.matmul(embedded_sentence.T).T
# we just want to confirm that the result is the same
torch.allclose(key_2, keys[1])
values = U_value.matmul(embedded_sentence.T).T
print(torch.allclose(value_2, values[1]))
# we now compute the the wij as the dot product wij = qiT*kj
# for example the following code calculated the unnormalized attention weight for w23
omega_23 = query_2.dot(keys[2])
# print(omega_23)
# the following code calculate the unormalized attention weight for w2
omega_2 = query_2.matmul(keys.T)
# print(omega_2)
# now we normalize the weights
attention_weight_2 = F.softmax(omega_2 / d**0.5, dim=0)
# print(attention_weight_2)
# finally we can calculate z(i)
context_vector_2 = attention_weight_2.matmul(values)
# print(context_vector_2)
# Multi-head attention
torch.manual_seed(123)
d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)
# 8 head
h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)
multihead_query_2 = multihead_U_query.matmul(x_2)
print(multihead_query_2.shape)
multihead_key_2 = multihead_U_key.matmul(x_2)
multihead_value_2 = multihead_U_value.matmul(x_2)
print(multihead_key_2[2])

stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
print(stacked_inputs.shape)
multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
print(multihead_keys.shape)
multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape)
print(multihead_keys[2, 1])
multihead_values = torch.matmul(multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)
multihead_z_2 = torch.rand(8, 16)

linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())
print(context_vector_2.shape)