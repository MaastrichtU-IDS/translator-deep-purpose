# GNNExplanations

### Setup

1. Download [DRKG](https://github.com/gnn4dr/DRKG) or use the following command:

```
wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
```

2. When you untar ```drkg.tar.gz```, you will find ```drkg.tsv```.

3. Add ```drkg.tsv``` to ```Input```.

### Quick Tour

* TrainGNN.ipynb : Takes as input ```drkg.tsv``` and outputs a trained Graph Neural Network (can be either GCN, GraphSAGE or GAT) to ```GNNModels```.

* AssessExplainability.ipynb : Assesses the explanability for use case of Alzheimer's disease. Takes as input: 
- the prediction(s) obtained from a trained Graph Neural Network: ```GNNModels/GAT```, ```GNNModels/GCN``` or ```GNNModels/GraphSAGE```.
- the ALzheimerGraph: ```Input/AlzheimerGraph``` or ```Input/AlzheimerGraphDrugDisease``` (only consists of compounds and diseases).

The notebook returns a small subgraph of the input graph/dictionary that contain the most important nodes and edges that were most influential for the prediction(s) using Integrated Gradients and Saliency and calculates Fidelity+, Fidelity- and Characterization score. A visualization for one instance is also provided. 

### Abstract

Graph Neural Networks (GNNs) consist of multiple permutation invariant functions, which is one of many reasons why they are so powerful architectures for graphs. The main idea behind Graph Neural Networks is to compute new node representations by aggregating vector representation from the neighbouring nodes. The difficulty in tracing the origin of those new node representations is one of the key drawbacks of this aggregation strategy, which makes it difficult to \textit{explain} the predictions made by GNNs. Thus, Graph Neural Networks lack explainability. Consequently, this paper presents a framework for explaining predictions made by Graph Neural Networks. The proposed method takes as input any prediction ${\hat{y}}_i$ of a trained Graph Neural Network and returns an explanation $G_S \subseteq G_C$ in the form of a small subgraph of the input graph. To return an explanation in the form of a small subgraph, we employed Integrated Gradients and Saliency Maps, which are two attribution methods capable of attributing an importance score to the nodes and edges from the input graph. We test our proposed framework on a case study for Alzheimer's disease. The best performance results were obtained using Graph Attention Network and Saliency Maps by achieving a Fidelity+ score of $1$, Fidelity- score of $0.20$, and Characterization score of $0.89$.
