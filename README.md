# Neural Machine Translation with Attention: English to Thai Translation

This project explores the use of neural machine translation (NMT) to translate between English and Thai using three different attention mechanisms: General Attention, Multiplicative Attention, and Additive Attention. The goal is to evaluate the effectiveness of these attention mechanisms in translating between the two languages.

## Project Overview
This repository implements neural machine translation models that incorporate three attention mechanisms to translate between English and Thai. The models are evaluated using various metrics, and performance is compared through training/validation loss plots and attention maps.

### Attention Mechanisms Implemented:

**1. General Attention:** A simple attention mechanism where the attention score is computed using a dot product between the query vector $𝑠$ and the key vector $ℎ𝑖$

$$
e_i = s^T h_i \in \mathbb{R} \quad \text{where} \quad d_1 = d_2
$$

**2. Multiplicative Attention:** An enhanced attention mechanism where the query is transformed using a learned weight matrix $𝑊$, and the attention score is computed as the dot product between the transformed query and key

$$
e_i = s^T W h_i \in \mathbb{R} \quad \text{where} \quad W \in \mathbb{R}^{d_2 \times d_1}
$$

**3. Additive Attention:** A more complex attention mechanism that applies a non-linearity (tanh) to a linear combination of the query and key before computing the attention score.

$$
e_i = v^T \tanh(W_1 h_i + W_2 s) \in \mathbb{R}
$$

## Task 1: Language Pair and Dataset Preparation

### 1. Dataset Selection

The dataset used for training the model is the **who-en-th dataset** from Hugging Face, titled *Tsunnami/who-en-th*. This dataset consists of 538 text samples, containing pairs of English and Thai sentences. It is designed for training models in the task of neural machine translation, enabling the translation of text between English and Thai. The dataset is particularly useful for developing language models focused on cross-lingual understanding and translation between these two languages.

**Dataset Source:** Hugging Face Datasets, *Tsunnami/who-en-th*

**Available at:** [Hugging Face - Tsunnami/who-en-th](https://huggingface.co/datasets/Tsunnami/who-en-th)

### 2. Dataset Preprocessing
The preprocessing steps for the English-Thai dataset are as follows:

**Tokenization:** Tokenization is performed on both English and Thai sentences using the following methods:

- **English:** The English text is tokenized using the `spaCy` tokenizer (`en_core_web_sm`).
- **Thai:** The Thai text is tokenized using **PyThaiNLP**, a popular Thai language processing library, for word segmentation.

The tokenization process is done using the `get_tokenizer` function from `torchtext` for English and `word_tokenize` from **PyThaiNLP** for Thai.`
- **For English:** `get_tokenizer('spacy', language='en_core_web_sm')`
- **For Thai:** `pythainlp.word_tokenize()`

**Numericalization:** After tokenization, we build vocabularies for both the source (English) and target (Thai) languages. The vocabularies are constructed using **torchtext**'s `build_vocab_from_iterator`. Special tokens like `<unk>`, `<pad>`, `<sos>`, and `<eos>` are included in the vocabularies, and their corresponding indices are defined as follows:
- `<unk>` (Unknown token): 0
- `<pad>` (Padding token): 1
- `<sos>` (Start of sentence): 2
- `<eos>` (End of sentence): 3

This numericalization process is crucial for converting tokens into their respective indices, which can then be used by the neural network for training.

**Libraries Used:**
- **PyThaiNLP** for Thai tokenization
- **spaCy** for English tokenization
- **torchtext** for tokenization and vocabulary building:

### 3. Data Splitting

After preprocessing, the dataset is split into the following sets:

- **Training Set:** 80% of the dataset is used for training the model.
- **Validation Set:** 10% of the dataset is used for validating the model during training.
- **Test Set:** 10% of the dataset is used for evaluating the model's performance after training.

## Task 2: Experiment with Attention Mechanisms

### Neural Machine Translation Model

The sequence-to-sequence model is implemented with an encoder-decoder architecture that includes attention mechanisms.

### Training and Evaluation

The model is trained on the English-Thai dataset, and the following metrics are tracked during training and evaluation:

- **Training Loss**
- **Training Perplexity**
- **Validation Loss**
- **Validation Perplexity**

## Task 3: Evaluation and Verification

### 1. Comparison of Attention Mechanisms
This section evaluates the performance of the three attention mechanisms, analyzing their behavior and effectiveness in translating between English and Thai. The models are compared based on overall performance, with a focus on how each attention mechanism influences translation quality.

### 2. Performance Plots
Performance plots are generated to visualize the training and validation loss curves for each of the three attention mechanisms. These plots help in comparing the learning progress of each model during training.

### 3. Attention Maps Visualization
Attention maps are visualized to show how the model focuses on different parts of the input sentence during translation. These maps provide insights into which parts of the source sentence are more influential in generating the target sentence.

### 4. Analysis of Attention Mechanisms
- **General Attention:** This mechanism is simple and efficient, often performing well on shorter or simpler sentences. However, it may struggle to capture long-range dependencies or more complex relationships between the source and target sequences.

- **Multiplicative Attention:** A more flexible mechanism, multiplicative attention performs better in capturing complex dependencies between the source and target. It is especially useful when translating sentences with intricate word relationships or longer contexts.

- **Additive Attention:** This mechanism can model the most complex relationships and dependencies due to its non-linear nature. While it is highly expressive, it may require more computational resources and time to train effectively compared to the simpler attention mechanisms.

## Installation

**1. Clone the Repository:** Clone the repository to your local machine.
```bash
git clone https://github.com/Prapatsorn-A/a3-neural-machine-translation.git
cd a3-neural-machine-translation
```

**2. Install Dependencies:** Install the dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**3. Run the Jupyter Notebook:** To run the notebook, use the following command.
```bash
jupyter notebook nmt_english_to_thai_attention.ipynb
```


