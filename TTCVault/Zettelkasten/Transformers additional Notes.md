2023-11-26 17:11
Status: #idea
Tags: [[From Matrices to Transformers]]

![[Pasted image 20231124180238.png]]

# Transformers additional Notes

## The Attention Mechanism

### Retrosynthetic perspective
With this I'll arrive at the attention part towards the end. Top-down, this way we give enough time to the moving parts of the mechanism and not just the attention part.

**Basis of vectors**
Let's imagine a matrix, $V$, which is simply $n_{basis}$ stacked vectors of size $d_v$. And that these vectors form a useful basis to propagate our residual stream tokens from. A plausible identity for the head could therefore be:
$$h_m^{l} = \sum_{n=1}^{n_{basis}} a_{mn} v_n$$
Where $h_m^{l}$ is a propagated residual stream vector for the m-th token, and $a_{mn}$ is simply a coefficient for drawing together the token $m$ and the basis vector $n$. In other words, we can write this as a matrix product:
$$
h(x^l) = AV^T
$$
![[Pasted image 20231126182319.png]]
The key point here is that our $A$ matrix attends to the different basis vectors, we'll show how this can be chosen in a minute, but the key point here is that we have n basis vectors which we want a normalized vector of coefficients to have some understanding of.

**Projection back to residual stream**
Note, however, this product $(n_{context} \times n_{basis}) \cdot (n_{basis} \times d_{v}) = (n_{context} \times d_{v})$ leaves the residual stream vector to be of size $d_v$. So finally if we want to add these vectors back to their to obtain a fuller description of such a way of propagation we want a matrix $W_O$ to project the propagated information back out: $$h(x^l) = AV^TW_O^T$$
![[Pasted image 20231126183812.png]]

!DIAGRAM ABOUT LINEAR COMBINATIONS + PROJECTION TO X_L

Section on attention scores

!DIAGRAM ON ATTENTION

Section on Parameterisation of everything

!DIAGRAM ABOUT LINEAR COMBINATIONS + PROJECTION TO X_L

Significance of parameterising the value matrix - virtual weights!





**The attention scores matrix, $A$** 
The detail we're missing here lies in how we parameterise $A$. We want $A$ to have the following property:
$$
\sum_{n=1}^{n_{basis}} a_{mn} = 1
$$
This guarantees that when we weight our vectors, we are confined to getting a vector that is of the same order of magnitude as the basis $V$. This can be achieved by the following reparameterisation of $A$:
$$A=Softmax(W)$$
Where the Softmax is applied across the basis vector dimension.



This now leads us to a discussion on why we'd want the weight matrix to be a function of the residual stream too, bearing in mind that the values matrix is  already a function of the residual stream.

Say for example our specific head at the specific layer took abstraction X of the sentence and computed the corresponding propagation of the information through the residual stream. If the attention scores were parameterised a-priori to in the model, the head will not be able to adapt to different flows of information;
it could simply 'null' the information if the abstraction X isn't present. That is, if your head computes something like the inference of a verb, but no verbs are present then the only way for it to null the result is to 
We want the attention mechanism to be responsive to the identity of the words in the sentence itself, such as, if the value vectors

You could in theory have more heads and more layers which in the limit form the same attention scores. However, since both the values and the attention matrix are parameterised by the state of the residual stream.



----
The vectors in this matrix are vaguely like we've taken a residual stream embedding of a sentence $x_l$, and for this particular head we pick out a set of vectors which are of interest:
```python
The cat sat on the mat.
-> x_l = [n_context x d_model]
-> K,Q,V where;
	V = [n_context x d_v] -> embedding(
		[The, cat, sat, on, the, mat]
	)
	V contains values of which we wish to propagate via the attention mechanism.
```


---

They are each of dimension $d_v$  
We want to somehow pick a linear combination of these $n$ vectors to enrich our representation.

Let's imagine the keys and values form a basis of vectors; they take values within $V_l$ 



**Components of h: $h(x_l) = h(x_l; W_Q, W_K, W_V)$**
The full attention mechanism isn't immediately intuitive, so we'll build up the equation from the beginning where I'll give my best attempt to explain each of it's parts. We start by projecting the residual stream onto three bases, compressing it's dimensionality from $d_{model}=1024$ to $d_k$ and $d_v$.

1. Queries: $Q_l = x_l W_Q^T$
2. Keys: $K_l = x_l W_K^T$
3. Values: $V_l = x_l W_V^T$

All of which have shapes $(n_{context} \times d_{model}) \cdot (d \times d_{model})^T = (n_{context} \times d)$ where we now note that we've performed a linear projection of the values of the residual stream onto the space of $d_{model} \rightarrow d_k$ or $d_v$ for the head. The parameterised aspect here allows the model to learn useful meanings behind this projection, which are aspects that still aren't understood about transformers, usual deep learning stuff. 

Note that the queries and keys have the same dimensions $d_k$ and the values will have a dimension size $d_v$. They're usually the same, commonly either $d_k=d_v = 64$ or $32$, but it's useful to remember that the queries can be matched to keys, but not to values. The compression aspect here isn't actually necessary, however we will go on to see how viewing it in this way makes it both intuitive and efficient to stack the process many times, leading to multi-headded attention.

**Query-Key matching**
The central component of the attention mechanism is the following product:
$$QK^T = x_lW_Q^T\cdot W_Kx_l^T =x_l \cdot W_{QK} \cdot x_l$$
Whose result is of shape $(n_{context} \times d_k) \cdot (d_k \times n_{context}) = (n_{context} \times n_{context})$. The token-wise query vectors each are matched in similarity the key vectors. These are scores for how closely each token embedding (query) matches the basis of 

K-V:
The key vectors represent the required query vector for each of the value vectors 

they are a basis derived from each token which represent.

In other words, for each $n_{context}$ token the associated embedding / query is matched with each vector in the set of keys to produce their dot product

Here we break down the full attention mechanism:
$$
Attention(Q,K,V) = AV= Softmax(QK^T)V
$$

1. $Q K^T$ - These are simply combinations of two different embeddings for the token embeddings in the residual stream. The queries and keys are matched in such a way that forms amplitudes for probabilities. The shape of this matrix is $(n_{context} \times d_{head}) \cdot (n_{context} \times d_{head})^T = (n_{context} \times n_{context})$
2. $A = Softmax(QK^T)$ - This converts the amplitudes of each feature dimension into probabilities. It gives us a probability assignment between positions. We call these attention scores. 
3. $AV$ - We can think of the value vector as the 


**Cross-attention, self-attention and why we call them Queries, Keys and Values**

The name comes from databases. The queries are embeddings formed in each head which will find the biggest values that amplify their value in the dot product.
In short, the query and value weights form a sort of customisable dictionary; they transform any word embedding into a relevant one. Each weight for QK forms a space wherein specific relationships between the resulting word embedding (from the query matrix) and the corresponding keys can be associated with one another and if their dot product is larger then the position-wise association of those two words will be larger. Then, the associated values will index them.

The natural question here is of course how can we simultaneously learn these two different meanings of the embeddings from the same vector alone. This is when the relevance of the original model becomes important: we can make a seperate model learn the query-key values for the model, a sort of dictionary of the embeddings, and have that encoded repr index the values within another model

- Q-decoder, KV - encoder. decoder -- this is the encoder-decoder
- encoder has self-attention
- Decoder self-attention allows model to attend to all
This is the encoder-decoder model.

**Caveat for Cross-attention: More detail on the original Transformer**

We discussed above how the difference between the decoder-only (GPT / BERT) transformers and encoder-decoder transformers stem from the residual feeding of QK matrices at some point in the encoder model. But there's an additional point I wanted to highlight: 

The sequence we encode may be of a different size to the one our decoder acts on. In this instance, the result matrix from the Softmax is no longer square:

$Q$ comes from the decoder: $(n_{target-context} \times d_{model})$
$K$ and $V$ both come from the encoder: $(n_{source-context} \times d_{model})$

Thus the attention scores become: $A_{\times} = Softmax(QK^T)$ will take shape $(n_{target-context} \times n_{source-context})$. We take note that the Softmax is applied in the dimension of the $n_{source-context}$, that is, they represent amplitudes for attention towards each token in the source. And following similar lines as above, we can then calculate dot products with the encoded value vector $V_{encoder}$ weighting it's encoded values along the $n_{source-context}$ dimension to get a resulting set of embedded vectors of $d_{model}$ dimensions. 


$$
h(x)_{cross} = (A_{cross}\otimes W_O) \cdot V_{encoder} = A_{cross}V_{encoder}W_O^T 
$$



$$
x_{i+2} = (I\otimes I + \sum_{h\in H} A_{cross}^h \otimes W_{OV}^h) \cdot x_i
$$


Each matrix, transforms the token vector, $x_i$, of dimensions $d_{model}$ from the residual stream to another vector, usually a smaller size, $d_{head}$. 

The 'keys' are interpreted as 

**Another caveaut you won't see fully explained**

1. dModel
2. reformulation to original MHA


---


**The Attention Mechanism**
$$
Attention(Q,K,V) = Softmax(QK^T)V
$$
1. $Q K^T$ - These are simply combinations of two different embeddings for the token embeddings in the residual stream. The queries and keys are matched in such a way that forms amplitudes for probabilities. The shape of this matrix is $(n_{context} \times d_{head}) \cdot (n_{context} \times d_{head})^T = (n_{context} \times n_{context})$
2. $Softmax(QK^T)$ - This converts the amplitudes of each feature dimension into probabilities. It gives us a probability assignment between positions in the  
3. $Softmax(QK^T)V$ - This uses the probabilities to index the. det 1.

We can phrase this using our notation, with $A=Softmax(Q^TK)$:

$$h(x)= (A\otimes W_V)\cdot x$$

Here we see the two components of the transformation that

**Multi-Headded Attention**

The dimensions to which we project our residual stream, $x_i$, is actually arbitrary but is taken as an integer multiple of $d_{model}$ for convenience, this way we can perform a single set of operations to compute all the $Q$, $K$ and $V$ matrices.

Finally, we apply another matrix transformation to the product above.

The full layer is therefore

$$
x_{i+1} = x_i + \sum_{h\in H} h(x_i)
$$
$$
x_{i+1} = (I\otimes I + \sum_{h\in H} (A \otimes W_V))\cdot x
$$

**Full Transformer and Multi-Head Attention**


The full transformer


$$T = (I \otimes W_U) \cdot (I\otimes I + \sum_{h\in H} A^h \otimes W_{OV}^h) \cdot(I \otimes W_E) \cdot x$$




---
The original mechanism was written as:
$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d_k}})V
$$


Let's decompose the elements of this:
- 
- Attention pattern: $A = Softmax(q^T k) = Softmax(x^T W_Q^T W_K x)$

The result vector is the linear combination of value vectors according to their attention:
$$
r = A v
$$

The notation that we 

$$h(x) = (I\otimes W_O) \cdot (A \otimes I) \cdot (I \otimes W_v) \cdot x = (A \otimes W_O W_V) \cdot x = (A \otimes W_{OV})x
$$

This shows succinctly that:
- The features are transformed
- Attention is applied in a position-wise manner

Note that for a forward pass we'd compute a unique $A$ matrix, as it depends on the input as described above.

## Build

There are a few design decisions about the transformer that seem strange or superficial at first. What's the point of the query, key and value matrices? They're superficial! $W_V W_O$ and $W_Q^T W_K$ in the equations we've outlined are paired together. However, we don't replace them by a single matrix as we want to keep the matrix as a low-rank separable matrix in this context.

## Zero-layer transformer
The residual stream will always have the original component of the token embedding, and is likely related to non-contextual information such as "Barack" is followed by "Obama". $$T=W_UW_E$$
# One-layer transformer

$$T = (I \otimes W_U) \cdot (I\otimes I + \sum_{h\in H} A^h \otimes W_{OV}^h) \cdot(I \otimes W_E)$$
We can rearrange this to the expanded form:
$$T = (I \otimes W_U W_E) + \sum_{h \in H} A^h \otimes W_U W_{OV}^h W_E$$
Here, we've clearly separated the zero-layer transformer from the additional part of the transformer that introduces the multi-headed attention. The $I$ being stated in the position where the position-wise acts indicates that it doesn't mix between positions!

Further, the $A^h$ is split into:
$$A^h \otimes W_U W_{OV}^h W_E = Softmax(t^T \cdot W_E^T\cdot W_{QK}^h\cdot W_E \cdot t) W_U \cdot W_{OV}^h \cdot W_E$$
Two circuits where:
- Output-value circuit maps $E$ space to $U$ space (how tokens map to the output logits)
- Query-key circuit maps $E$ space to $E$ space (how tokens map to other tokens)

This is further achieved by the significance of the matrix resulting from the softmax having a unitary determinant -- it won't change amplitudes of the vectors.

The OV-circuit maps the tokens
in other words, we can analyse the eigenvectors of the matrix $W_U W_{OV}^h W_E$
For an eigenvector with positive eigenvalue, the linear combination of those embeddings will "copy" itself, increase it's probability in the final output.

We would expect this to be zero on average (the average eigenvalue is zero) but it's often seen to be positive in 1-layer transformers.


## Expanding to several
If we wish to expand this
$$T = (I \otimes W_U) \cdot (I\otimes I + \sum_{h\in H} A^h \otimes W_{OV}^h) \cdot (I\otimes I + \sum_{h\in H} A^h \otimes W_{OV}^h) \cdot(I \otimes W_E)$$

---

Output of an attention-head:
1. Value vectors: $v_i = W_V x_i$
2. Result vector: $r_i = \sum_j A_{i,j}v_j$ -- linear combination of the value vectors in an attention pattern
3. Output vector: $h(x)_i = W_O r_i$
Single-step formulation:
x as a 2d matrix - vector for each token, $N * c$
The full attention involves multiplication of weight matrices on the "vector per token" side wheras A is on the "position" side. Is overall a (2,2)-tensor mappings

These are the components of the attention mechanism. 

### Quick note on Tensor / Kronecker Product notation
The details for the origins of the tensor product notation are detailed in the original thread, we skip the specific details here and focus on gaining the intuition that we seek to get from the notation.

Our input tensors to these operations, $x$, are 2-D matrices and we can either act across the rows or columns in this case.

If $x$ is an $n_{context} \times d_{model}$ matrix, where $n_{context}$ is the number of vectors in the scope and $d_{model}$ is the number of channels or features for each. If we have any square matrix $W$, then it is considered a type 1,1 tensor mapping vectors to vectors.

- Position-wise, for each channel: $xW$ is a multiplication across the $d_{model}$ dimension of $x$ and for the transformation to be 1,1, $W$ must be a matrix of size $d \times d$. In tensor product notation, this is exemplified easily through $(I\otimes W) \cdot x = xW^T$, where the matrix is explicitly shown to act with the same form across positions. 
- Channel-wise, for each position: $Ax$ is a multiplication across the $n_{context}$ dimension, where $A$ is therefore a $n_{context} \times n_{context}$ matrix. Equivalently, $(A \otimes I) \cdot x = Ax$ is an explicit way of showing that the matrix acts with the same multiplication across each feature, where the weights themselves are different for each position in the sequence.
in other words:

$$
h(x)_i = W_O r_i
$$

This is the central point behind the transformer, the combination of position-wise and channel-wise interactions. You're essentially combining linear layers across both spaces of the data through which matrix multiplication can act.

Overall, we have a 2,2-tensor transformation leaving the shape of the data matrix or residual stream, $x$, unchanged.


## Skip-trigrams


## Notes on implementation
There are a few ways in this article whereby we've formulated transformers in different ways to their implementation:

We'll have a seperate article on the formulation of an efficient transformer, where we'll also detail the implementation details for language and train on the wikitext dataset.

---
## Footnote on Notation
Note that the order of applications for $A\otimes W$ doesn't matter due to the commutativity property of matrices $(Ax)W^T = A(xW^T)$. 

You can also interpret a matrix form of the tensor product:
$A\otimes W$ is equivalent to a 4-D tensor:

```python
A[:, :, None, None] * W[None, None, :, :]
```

Broadcasting / copying the matrices into their respective dimensions:
The components of A form the spaces of a 2D matrix, within which are the components of the W matrix multiplied by each of it's elements. In other words 

```python
A[i, j] # a 2D matrix of W * a the scalar value from A_ij
```










___
# References