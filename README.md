# LLM_from_Scratch

This repository documents my journey of building a **Large Language Model (LLM) from scratch**

## Daily Progress

### Day 1: Understanding LLMs & Revisiting Fundamentals

- Studied the **basics of Large Language Models (LLMs)**
- Revised **Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM)** networks
- Started watching videos and reading basics of:
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

- Implemented **simplified attention mechanism** with non-trainable weights from scratch → [SimplifiedAttention.ipynb](2_Attention_Mechanism/1_SimplifedATT_noTrainableweights.ipynb)

- Implemented **self-attention mechanism** using key, query, and value matrices with trainable weights from scratch → [SelfAttention.ipynb](2_Attention_Mechanism/2_selfattention_trainable.ipynb)

- Implemented **casual-attention mechanism** with dropout from scratch → [CasualAttention.ipynb](2_Attention_Mechanism/3_casualattention.ipynb)

### Day 6: Multihead Attention Mechanism from Scratch

- Implemented **Multihead Attention Mechanism from Scratch** using simple Implementation → [Multihead.ipynb](2_Attention_Mechanism/4_mutiheadattention.ipynb)

- Implemented **Multihead Attention Mechanism from Scratch** with weight split and one class( no wrapper class ) → [Multihead.ipynb](2_Attention_Mechanism/4_mutiheadattention.ipynb)

### Day 7: GPT-2 Core Components

- Added **boilerplate Code** for gpt 2 architecture → [BoilerplateCode.ipynb](3_GPT/1_boilercode.ipynb)

- Implemented **Layer Normalization class** for LLM → [LayerNorm.ipynb](3_GPT/2_layernorm.ipynb)

- Implemented a **feed forward network with GELU activations** for LLM → [Gelu.ipynb](3_GPT/3_gelu.ipynb)

- **Shortcut /Skips connections** for LLM → [ShortCutconnection.ipynb](3_GPT/4_shortcutconnections.ipynb)

### Day 8: Transformer Block & Training

- Implemented **Entire LLM Transformer Block** → [Transformer.ipynb](3_GPT/5_transformer.ipynb)

- Coding the **124 million parameter GPT-2 model** → [GPT2.ipynb](3_GPT/6_gpt2_124M.ipynb)

- Coding the GPT-2 to **predict the next token** → [nextwordprediction.ipynb](3_GPT/7_gpt2_generatenextword.ipynb)

- Implemented **Cross entropy and perplexity loss** for LLM → [Loss.ipynb](3_GPT/7_gpt2_generatenextword.ipynb)

### Day 9: Evaluation and Sampling

- **Evaluating LLM performance on real dataset** → [GPT2_RealDataset.ipynb](3_GPT/8_gpt2ondataset.ipynb)

- Coding the **Entire LLM Pre-training Loop** → [GPT2_entirePretraining.ipynb](3_GPT/8_gpt2ondataset.ipynb)

- Implemented **Temperature Scaling** in LLM → [TemperatureScaling.ipynb](3_GPT/9_tempScaling.ipynb)

- Implemented **Top-k sampling** in LLM → [TOP-Ksampling.ipynb](3_GPT/10_topK.ipynb)

### Day 10: Weight Loading

- **Saving and loading LLM model weights** using PyTorch → [Save_Load_weights.ipynb](3_GPT/11_Save_load.ipynb)

- Loading **pre-trained weights from OpenAI GPT-2 124M** → [Loading_OPEN-AI_weights.ipynb](3_GPT/12_loadingOpenAI.ipynb)

- Training using **OpenAI GPT-2 774M weights** → [774M_weights.ipynb](3_GPT/13_OpenAI_774M.ipynb)

### Day 11: Classification Fine-Tuning

- **Classification Finetuned 124M Model on Spam classification dataset** → [SpamClassificationFinetuned.ipynb](4_FineTune/1_classificationFinetuning_spam/1_spam.ipynb)

- **Classification Finetuned 124M Model on PubMed 20k dataset** → [MedicalClassificationFinetuned.ipynb](4_FineTune/2_classificationFinetuning_medical/1_medical.ipynb)

### Day 12: Instruction Fine-Tuning

- **Instruction Finetuned 124M Model on Alpaca-style prompt formatting** → [InstructionFinetuned.ipynb](4_FineTune/3_instructionFinetuning_basics/basic.ipynb)
