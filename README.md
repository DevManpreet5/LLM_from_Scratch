# LLM_from_Scratch

This repository documents my journey of building a **Large Language Model (LLM) from scratch**

## Daily Progress

### Day 1: Understanding LLMs & Revisiting Fundamentals

- Studied the **basics of Large Language Models (LLMs)**
- Revised **Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM)** networks
- Started reading:
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [Language Models Are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Day 2: Tokenization & Preprocessing

- Implemented a **simple tokenizer** from scratch & Added **special character tokens** → [Tokenizer.ipynb](1_Preprocessing/1_Tokenizer.ipynb)
- Implemented **Byte Pair Encoding (BPE)** using `tiktoken` → [Bytepairencoding.ipynb](1_Preprocessing/2_Bytepairencoding.ipynb)

### Day 3: Input-Target Pairs & Token Embedding

- Implemented **Input-Target data pair generation** using `DataLoader` → [TargetPair.ipynb](1_Preprocessing/3_Input_Target_pair.ipynb)

- Explored **vector embedding** → [Word2Vec Google News (300D)](https://huggingface.co/fse/word2vec-google-news-300)

- Created a **token embedder** in Torch using `torch.nn.Embedding`→ [TokenEmbedding.ipynb](1_Preprocessing/4_tokenEmbedding.ipynb)

- Implemented **positional token embedding** in Torch → [PositionalTokenEmbedding.ipynb](1_Preprocessing/5_positionTokenEmbedding.ipynb)

### Day 4: Basics of Attention Mechanism

- Read [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Explored simplified , self ,causal , multi-head attention, why RNN fails
- History of RNN , LSTM , Transformer
- Learned about Bahdanau attention

### Day 5: Attention Mechanism from Scratch

- Implemented **simplified attention mechanism** with non-trainable weights from scratch → [SimplifiedAttention.ipynb](2_Attention_Mech/1_SimplifedATT_noTrainableweights.ipynb)

- Implemented **self-attention mechanism** using key, query, and value matrices with trainable weights from scratch → [SelfAttention.ipynb](2_Attention_Mech/2_selfattention_trainable.ipynb)
