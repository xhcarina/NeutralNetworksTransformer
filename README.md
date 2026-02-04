# Neural Networks and Transformers 

A **single project** that implements core deep-learning architectures from the ground up—no high-level NN libraries for the first part, then full Transformer and language-model code in PyTorch. The goal is to build and train **neural networks** and **Transformers** from first principles, then apply them to real tasks (image classification, text generation, sequence classification).

---

## Project overview

| Part | Notebook | Stack | Focus |
|------|----------|--------|--------|
| **1** | `NeturalNetwork.ipynb` | NumPy only | Fully-connected NNs: forward/backward, backprop, SGD from scratch |
| **2** | `Transformer.ipynb` | PyTorch | CNNs, self-attention, encoder/decoder, GPT-style LM, encoder-only classifier |

Together, the two notebooks cover the path from **vanilla NNs** → **CNNs** → **attention** → **Transformers** → **language models** and **sequence classification**.

---

## Part 1: Neural network from scratch (`NeturalNetwork.ipynb`)

- **Layers:** `Module`, `Sequential`, `Linear`, `ReLU`, `Dropout`, loss modules with **hand-written backward passes** (no autograd).
- **Training:** SGD, batching, epoch loop; optional numerical gradient checks.
- **Task:** **Fashion MNIST** image classification (10 classes, 28×28).
- **Experiments:** Cross-validation over learning rate, batch size, dropout; hyperparameter search and final model selection.

**Result:** Best accuracy **0.94980**

---

## Part 2: Transformer and language model from scratch (`Transformer.ipynb`)

- **CNN module:** MLP and Conv + pooling on **CIFAR-10** (image classification).
- **Transformer core:** Scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder/decoder blocks—all implemented explicitly.
- **Decoder-only LM:** GPT-style model for **text generation** (e.g. “Story-GPT”); sampling and generation loop.
- **Encoder-only classifier:** Transformer encoder on **nucleotide sequences** (DNA); tokenizer (A/C/G/T), positional embedding, stacked encoder layers, linear head for classification; CSV export for leaderboard.

**Result:** Encoder-only accuracy **0.72444**.

---

## Running the project

- **Part 1:** NumPy, matplotlib, scikit-image, `utils`; data via gdown (links in the notebook). Run all cells in order.
- **Part 2:** PyTorch, torchvision, NumPy, matplotlib, seaborn, tqdm. Run cells in order; data/setup steps are in the notebook.

Each notebook is self-contained; run Part 1 then Part 2 for the full “Neural Networks → Transformers” sequence.
