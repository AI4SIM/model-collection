# Use Case - Emulating Ocean Atmosphere Interactions


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#model">Model</a></li>
        <li><a href="#built-with">Built with</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#build-your-own-training-dataset">Build your own training dataset</a></li>
        <li><a href="#training">Jit Training</a></li>
        <li><a href="#inference">Jit Inference</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- DESCRIPTION-->
## Description
High resolution oceanic climate models requires to model **air-sea interactions** to accurately predict the long-term dynamics of the ocean. There are mainly two strategies used to introduce these interactions in modern simulations:
- Using **fully coupled ocean-atmosphere models**, where both the atmosphere and the ocean dynamics are integrated concurrently with a similar spatial resolution. This approach is **computationally expensive**, mostly due to the atmospheric model which tends to exhibit fast dynamics compared to the ocean.
- Feeding the oceanic model with **pre-computed atmospheric fields** (from observations or simulations) in order to estimate air-sea interactions. This strategy is more efficient, at the cost of using decoupled atmospheric data which may not be as well spatially resolved as the ocean model.

This objective of this work is to improve the second approach by correcting the air-sea interactions using a deep neural network trained with fully coupled simulations.

<p align="right">(<a href="#top">back to top</a>)</p>

### Dataset
The training dataset consists of a series of coupled ocean-atmosphere simulations. All the scripts used to generate out dataset can be found in the [dataset_builder](dataset_builder) section of this repository.

<p align="right">(<a href="#top">back to top</a>)</p>

### Model
Coming soon
<p align="right">(<a href="#top">back to top</a>)</p>

### Built with
* [MesoNH](http://mesonh.aero.obs-mip.fr/): Mesoscale Non-Hydrostatic atmospheric model
* [Croco](https://www.croco-ocean.org/): Coastal and Regional Ocean COmmunity model
* [Docker](https://www.docker.com/) & [Singularity](https://sylabs.io/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
### **Build your own training dataset**
1. Install OCI images and prepare your run directory

```sh
# Create a directory to store simulation outputs
mkdir -p <your_run_directory>

# Build the OCI images needed to run a couplde ocean-atmosphere simulation
./docker_build.sh <your_run_directory>

# Download external surface topology needed by the atmospheric model
cd dataset_builder/run_wimulation
./download_PGD_files.sh <your_run_directory>

# Go to the simulation config you want to run. Here: wmed
cd wmed
```
2. Setup prepare_run_<your_machine>.sh run_<your_machine>.sh and config_<your_machine> for your own compute cluster. We provide an example for our in-house slurm machine (pise).
3. Run the preprocessing
```sh
# Preprocessing (initial conditions, forcing, interpolations, coupling setup)
./prepare_run_<your_machine>.sh <your_run_directory>/wmed
```
4. Run the simulation
```sh
# Run the simulation
./run_<your_machine>.sh <your_run_directory>/wmed
```
5. Once the simulation is complete, run the [post processing scripts](dataset_builder/post_processing/) 


<p align="right">(<a href="#top">back to top</a>)</p>

### **Training**
Coming soon
<p align="right">(<a href="#top">back to top</a>)</p>

### **Inference**
Coming soon
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Run the model on a West-Mediterratean config (wmed)
    - [x] Setup OCI images (MesoNH, Croco) to run coupled ocean-atmosphere simulations
    - [x] Setup build and run scripts for our in-house Pise cluster
- [x] Write down post-processing scripts to clean-up simulation outputs
- [ ] Create a data-loader to build the dataset from processed files
- [ ] Training
- [ ] Inference

<p align="right">(<a href="#top">back to top</a>)</p>