import torch
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