# TAUS: Sparse Tweet Network for Classification of Informative Posts using Graph Convolutional Network

## Overview
TAUS is a project focused on developing a classification model to distinguish between informative and non-informative tweets. The project utilizes BERT (Bidirectional Encoder Representations from Transformers) embeddings to represent the tweets and employs Graph Convolutional Neural Networks (GCNs) for classification. By leveraging relational data through GCNs, TAUS aims to enhance the performance of tweet classification, particularly in processing social media interactions.

The dataset used in this project consists of two classes:
- **Informative tweets**: These are tweets that contain useful and relevant information, labeled as **0**.
- **Non-informative tweets**: These are tweets that do not contain significant information, labeled as **1**.


## Introduction
This research aims to develop a classification model to distinguish between informative and non-informative tweets using BERT embeddings to represent the tweet. BERT is used to extract contextual word representations, capturing the fine semantic nuances of the text. These embeddings are utilized to create a graph for the classification of tweets.

We have employed Graph Neural Networks, specifically the Graph Convolutional Neural Network (GCN), which performs well in classifying tweets, especially when evaluated on limited sample datasets. GNNs have demonstrated their efficacy in leveraging relational data, which augments the performance of the classifier in processing social media interactions. The proposed model integrates these two state-of-the-art approaches to achieve superior performance in tweet classification related to the context and content of information.


## Major Contributions
- **Curating a Publicly Available Tweet Dataset**: Based on a significant event, we have curated a publicly available dataset of tweets.
- **Creating Tweet Graphs with Varying Degrees of Density**: We experimented with different levels of graph density to observe its impact on classification performance.
- **Extensive Experiments on Graph Convolutional Networks**: Conducted thorough experiments using GCNs to classify tweets as informative vs. non-informative.

## Dataset Statistics
The following table shows the number of informative and non-informative tweets in the dataset. The imbalance in both classes is mitigated by applying augmentation.

|                      | **# Informative Tweets** | **# Non-Informative Tweets** |
|----------------------|--------------------------|------------------------------|
| **Number of Tweets** | 3036                     | 3175                         |


## Contributors

We would like to thank the following contributors for their valuable input and support in this project:

- [parul-88](https://github.com/parul-88)
- [Pr1910](https://github.com/Pr1910)
- [SMIT-2411](https://github.com/SMIT-2411)
- [coderphilosophy](https://github.com/coderphilosophy)


