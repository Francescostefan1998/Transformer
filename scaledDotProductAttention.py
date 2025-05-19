
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
print(omega_23)
# the following code calculate the unormalized attention weight for w2
omega_2 = query_2.matmul(keys.T)
print(omega_2)