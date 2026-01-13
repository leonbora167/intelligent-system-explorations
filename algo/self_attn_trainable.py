# Lecture - https://youtu.be/aL2Qr5FXxko?si=ZlqJvvBCVoyM3fvo

import torch 
import numpy as np

tensor = torch.tensor([0.1, 0.2, -0.3, 0.45, -0.6])

#Softmax without scaling 
softmax_result = torch.softmax(tensor, dim=-1)
print("Softmax without scaling : ", softmax_result)

#Multiply tensor by 8 and then apply softmax 
scaled_tensor = tensor * 8 
softmax_scaled_result = torch.softmax(scaled_tensor, dim=-1)
print("Scaled softmax is : ", softmax_scaled_result)

def compute_variance(dim, num_trials=1000):
    dot_products = [] 
    scaled_dot_products = [] 

    for i in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)

        #Compute . product 
        dot_product = np.dot(q,k)
        dot_products.append(dot_product)

        #Scale dot prodcut by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)

    #Caclulate variance 
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling 

# for dimension 5 
variance_before_5, variance_after_5 = compute_variance(5)
variance_before_20, variance_after_20 = compute_variance(20)

print(f"Variance before scaling for (dims=5) is {variance_before_5}")
print(f"Variance after scaling for (dims=5) is {variance_after_5}")
print()
print(f"Variance before scaling for (dims=20) is {variance_before_20}")
print(f"Variance after scaling for (dims=20) is ",variance_after_20)

#Assuming a sentence like "Dream big and work on"
words = ['Dream', 'big', 'and', 'work', 'on']

inputs = torch.tensor([  # Input embeddings (vectors) for the input tokens will be taken care of by a pretrained model
    [0.72, 0.45, 0.31], # Dream -> x(1)
    [0.75, 0.20, 0.55], # big  -> x(2)
    [0.30, 0.80, 0.40], # and -> x(3)
    [0.85, 0.35, 0.60], # work -> x(4)
    [0.55, 0.15, 0.75]  # on -> x(5)
])

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 #Context vectors after Q,Kt and all 

#Randomly initialize Wq, Wk, Wv matrices 
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

print("W_q : ", W_query)
print("W_k : ", W_key)
print("W_v : ", W_value)

query_2 = x_2 @ W_query
value_2 = x_2 @ W_value
key_2 = x_2 @ W_key

print("Transformed query : ",query_2)
print("Transformed value : ",value_2)
print("Transformed key : ",key_2)

keys = inputs @ W_key
values = inputs @ W_value 
queries = inputs @ W_query

#Calculate attentions scores for query 2 
attn_scores_2 = query_2 @ keys.T 
#Normalize them 
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention Score for query 2 are : ", attn_scores_2)
print("Normalized attention scores -> Weights for query 2 are ", attn_weights_2)

context_vec2 = attn_weights_2 @ values 
print("Context vector for token 2 is ", context_vec2)

import torch.nn as nn 

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in, d_out))
        self.W_value = nn.Parameter(torch.randn(d_in, d_out))
        self.W_key = nn.Parameter(torch.randn(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value 
        queries = x @ self.W_query

        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights * values # Values should be the transformed inputs rt
        return context_vec

 