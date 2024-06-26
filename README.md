# SparseNet: Sparse Tweet Network for Classification of Informative Posts using Graph Convolutional Network

## Overview
SparseNet is a project focused on developing a classification model to distinguish between informative and non-informative tweets. The project utilizes BERT (Bidirectional Encoder Representations from Transformers) embeddings to represent the tweets and employs Graph Convolutional Neural Networks (GCNs) for classification. By leveraging relational data through GCNs, SparseNet aims to enhance the performance of tweet classification, particularly in processing social media interactions.

## Keywords
- Tweet Classification
- Information
- Graph Convolutional Network
- BERT
- Transformer

## Introduction
This research aims to develop a classification model to distinguish between informative and non-informative tweets using BERT embeddings to represent the tweet. BERT is used to extract contextual word representations, capturing the fine semantic nuances of the text. These embeddings are utilized to create a graph for the classification of tweets.

We have employed Graph Neural Networks, specifically the Graph Convolutional Neural Network (GCN), which performs well in classifying tweets, especially when evaluated on limited sample datasets. GNNs have demonstrated their efficacy in leveraging relational data, which augments the performance of the classifier in processing social media interactions. The proposed model integrates these two state-of-the-art approaches to achieve superior performance in tweet classification related to the context and content of information.

Additionally, we have explored the performance of classification using GCN with varying degrees of sparseness of the input graph by controlling the graph density.

## Major Contributions
- **Curating a Publicly Available Tweet Dataset**: Based on a significant event, we have curated a publicly available dataset of tweets.
- **Creating Tweet Graphs with Varying Degrees of Density**: We experimented with different levels of graph density to observe its impact on classification performance.
- **Extensive Experiments on Graph Convolutional Networks**: Conducted thorough experiments using GCNs to classify tweets as informative vs. non-informative.

## Dataset Statistics
The following table shows the number of informative and non-informative tweets in the dataset. The imbalance in both classes is mitigated by applying augmentation.

|                      | **# Informative Tweets** | **# Non-Informative Tweets** |
|----------------------|--------------------------|------------------------------|
| **Without Augmentation** | 1012                     | 3175                         |
| **With Augmentation**    | 3036                     | 3175                         |

## Results
The following table summarizes the performance metrics from various experiments conducted:

| **Experiment** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|----------------|--------------|---------------|------------|--------------|
| **Experiment 1** | 0.8840       | 0.8747        | 0.8864     | 0.8805       |
| **Experiment 2** | 0.8346       | 0.7780        | 0.8698     | 0.8213       |
| **Experiment 3** | 0.7970       | 0.7209        | 0.8410     | 0.7763       |
| **Experiment 4** | 0.5048       | 0.0374        | 0.4250     | 0.0687       |


## Installation
To get started with SparseNet, clone this repository and install the required dependencies:
```bash
git clone https://github.com/parul-88/TAUS.git
cd SparseNet
```
## Usage
### Data Preparation
Upload the tweet dataset `CLASSIFICATION_binary.ipynb` .

### Training the Model
Run the `CLASSIFICATION_binary.ipynb` script to start training the model:
```bash
python src/CLASSIFICATION_binary.ipynb
```

## References

1.Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems.

2.Kipf, T. N., & Welling, M. (2016). Semi-Supervised Classification with Graph Convolutional Networks. arXiv preprint arXiv:1609.02907.

## Contributors

We would like to thank the following contributors for their valuable input and support in this project:

- [parul-88](https://github.com/parul-88)
- [Pr1910](https://github.com/Pr1910)
- [SMIT-2411](https://github.com/SMIT-2411)
- [coderphilosophy](https://github.com/coderphilosophy)


