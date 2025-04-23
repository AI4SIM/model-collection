# Reactive flow

The following is an effort to improve subgrid scale Large Eddy Simulations (LES) for combustion using deep learning models. This work has been carried under the [RAISE](https://www.coe-raise.eu/) European project.

## Description

In the combustion community, the determination of the subgrid scale contribution to the filtered reaction rate in reacting flows Large Eddy Simulation (LES) is an example of closure problem that has been daunting for a long time. Various analitical approximations has been proposed to capture the subgrid scale phenomena associated with combustion, but none fully convincingly. A new approach is proposed for premixed turbulent combustion modeling based on neural networks by reformulating the problem of subgrid flame surface density estimation as a machine learning task. In order to train a neural network for this task, a Direct Numerical Simulation (DNS) and the equivalent LES obtained by a spatial filtering of this DNS is needed.

The whole process then boils down to:
- generate a fully-resolved dataset (DNS)
- filter down the DNS to a reduced-order space
- train a neural network to learn the mapping between features of interest in the reduced-order space
- at inference, run a LES solver in the reduced-order space (to solve the CFD) coupled with the pretrained neural network (to solve the other phenomena, *e.g.* thermodynamics, chemistry)

## Dataset

The dataset can be [downloaded here](https://www.coe-raise.eu/od-combustion), which contains 113 scalar fields for a progress variable (input) and the target value of the flame surface density (output) used in for a LES combustion model. Those fields are obtained from DNS simulations and filtered to match a typical LES simulation. More details on how the dataset is generated in the preprint [Lapeyre et al. (2018)](https://arxiv.org/abs/1810.03691)

## Models 

This problem is approached in two different ways:
- with convolutional neural networks (CNNs): a 3D U-net approach is used first to match [Lapeyre et al. (2018)](https://arxiv.org/abs/1810.03691)'s approch;
- with graph neural networks (GNNs): to directly work with structured and complexed mesh geometries.

These neural networks differ in the way they encode as inductive bias the problem's geometry. CNNs strength lies in its ability to share weights between translated convolutional kernels, allowing to natively learn translation invariant features. However, their main limitation lies in the Euclidian geometry they assume, requirng to interpolate complex meshes to cartesian grids before using them. On the other hand, GNNs directly work on irregular meshes, with various cell sizes and topologies, allowing to seamlessly process physicals fields sampled on various mesh geometries.
