# Combustion

## Description

This use-case is a subcase of the reactive flow aiming at providing a minimum viable coupling between an LES solver and a GNN to provide an end-to-end numerical simulation of combustions on irregular geometrics (*i.e.* with industrailly plausible meshes).

## Dataset

The dataset can be downloaded from the Jülich Supercomputing Centre (JSC)'s open-access [JUDAC](https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/dc4eef36-1929-41f6-9eb9-c11417be1dcf).

## Models

Graph Neural Networks (GNNs, also called *GraphNets*) provide a framework to extract convolution-like features from irregular meshes, considered as graphs. This is especially valuable for CFD as it allows for neural network inference directly on the mesh used by the solder, without resorting to often intractable interpolations to a cartesian grid. Several GNN architectures exist, such as Graph Attention Networks (GAT) from [Veličković et al. (2017)](https://arxiv.org/abs/1710.10903), GIN from [Xu et al. (2018)](https://arxiv.org/abs/1810.00826) etc.


