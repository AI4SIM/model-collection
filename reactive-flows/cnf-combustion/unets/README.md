# CNF Use-Case

## Description

In this folder, a 3D U-Net is implemented and plugged on the reactive flow CNF use-case to learn subgrid scale phenomena. Once the model is trained, it can be coupled with a standard LES solver to provide a complete end-to-end simulation the reduced order space while retaining physically consistent averages of thermodynamic and chemical phenomena.

## Dataset

Direct Numerical Simulations (DNS) from the CNF use-case are provided by the [RAISE](https://www.coe-raise.eu/od-combustion) project. Once downloaded, it must be provided to the *CnfCombustionDataset* with the root parameter.

In a first step, two DNS of a methane-air slot burner are run and then filtered to create the training dataset. Models are trained on this data in a supervised manner. In a second step, a new, unseen and more difficult case was used to ensure network capabilities.
This third DNS is a short-term transient started from the last field of the second DNS, where inlet velocity is doubled, going from 10 to 20 m/s for 1 ms, and then set back to its original value for 2 more ms.

## Models

The U-Net model [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) is a specific convolutional architecture (CNN) which allows for physical field regression with two data paths:
- a deep path, where increasingly global features are learnt by deep layers, in a autoencoder-like fashion;
- a straight through path, where data sidestep the deep path through the skip connections, allowing fine-grained details to directly path through.

