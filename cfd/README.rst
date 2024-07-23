CNF Use-Case
===============
The following is an effort to improve sub-grid scale LES models for combustion using artificial intelligence. This work is made under the RAISE european project: https://www.coe-raise.eu/

Description
-----------------
In the combustion community, the determination of the sub-grid scale contribution to the filtered reaction rate in reacting flows Large Eddy Simulation (LES) is an example of closure problem that has been daunting for a long time. A new approach is proposed for premixed turbulent combustion modeling based on convolutional neural networks by reformulating the problem of subgrid flame surface density estimation as a machine learning task.  In order to train a neural network for this task, a Direct Numerical Simulation (DNS) and the equivalent LES obtained by a spatial filtering of this DNS is needed.
In a first step, two DNS of a methane-air slot burner are run and then filtered to create the training dataset. Models are trained on this data in a supervised manner. In a second step, a new, unseen and more difficult case was used to ensure network capabilities.
This third DNS is a short-term transient started from the last field of the second DNS, where inlet velocity is doubled, going from 10 to 20 m/s for 1 ms, and then set back to its original value for 2 more ms.

Dataset
-----------------
The dataset can be `downloaded here <https://www.coe-raise.eu/open-data>`_, which contains 113 scalar fields for a progress variable (input) and the target value of the flame surface density (output) used in for a LES combustion model. Those fields are obtained from DNS simulations and filtered to match a typical LES simulation. More details on how the dataset is created in the preprint `Lapeyre et al. (2018) <https://arxiv.org/abs/1810.03691>`_.

Models 
-----------------
This problem is approached two ways: via CNNs and with GNNs. A 3D U-net approach is used first to match the Lapeyre's paper. However, CNNs are hardly applicable in complex non-structured grids. GNNs, however, are more suitable for working with unstructured, complex geometries. 
