# LEcture from https://youtu.be/NUBqwmTcoJI?si=dSUMR8009EA14F5D

import torch

#Assuming a sentence like "Dream big and work on"
words = ['Dream', 'big', 'and', 'work', 'on']

inputs = torch.tensor([  # Input embeddings (vectors) for the input tokens will be taken care of by a pretrained model
    [0.72, 0.45, 0.31], # Dream -> x(1)
    [0.75, 0.20, 0.55], # big  -> x(2)
    [0.30, 0.80, 0.40], # and -> x(3)
    [0.85, 0.35, 0.60], # work -> x(4)
    [0.55, 0.15, 0.75]  # on -> x(5)
])

#Calculate magnitude of each vector 
magnitudes = torch.norm(inputs, dim=1) # tensor([0.9039, 0.9513, 0.9434, 1.0977, 0.9421])

for word, magnitude in zip(words, magnitudes):
    print(f"For word {word} the magnitude = {magnitude.item():.3f}")

#Calculate the attention between each word with itself 

#Calculating the attention for the second input token "big"
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0]) # Tensor of 6 rows 
for i, x in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x, query) # Dot product between "big" embedding and other word's embeddings one by one

print(f"Attention scores for {words[1]} = {attn_scores_2}")

#Simple normalization
attn_weights_2_norm = attn_scores_2 / attn_scores_2.sum() 
print("Attention Weights :- ", attn_weights_2_norm)
print("For validity we are checking if the sum of Attention weights are coming to 1 and here its coming to : ", attn_weights_2_norm.sum())

# To calculate the dot product of every tensor in inputs with each and every tensor in inputs itself 
attn_scores = inputs @ inputs.T 
print("All attention scores are ", attn_scores)
# A @ B -> Computes matrix product of A and B 
# if inputs is of (N,D) then inputs.T will be (D,N) hence matmul is possible
# Matrix multiplication effectively computes the similarity between all possible pairs of word embeddings

attn_weights = torch.softmax(attn_scores, dim=-1) #dim=-1 refers to the last dimension which technically would be column here but 
# when [[q1], [q2], [q3]] is the matrix, where each q = [0.34,0.321,0.65,0.98] then 
# softmax goes through each column element of the matrix as a whole and does the softmax computation for each matrix containing values (row matrix)
print("Attn Weights = \n", attn_weights)

# context vector for query 2 = attn weight of 2 with 1 * embedding 1 + attn weight of 2 with 2 * embedding 2....
all_context_vectors = attn_weights @ inputs
print("All context vectors are \n", all_context_vectors)