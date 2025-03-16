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

### Day 3: Creating Input-Target Data Pairs

- Implemented **Input-Target data pair generation** using `DataLoader`→ [TargetPair.ipynb](1_Preprocessing/3_Input_Target_pair.ipynb)
