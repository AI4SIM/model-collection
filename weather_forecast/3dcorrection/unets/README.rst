3DCorrection Use-Case
=================
This pipeline aims at surrogating maelstrom-radiation's 3D corrections with 1D U-Nets.

Dataset
-----------------
The data are first downloaded with the climetlab's API. They are stored in data/cached with a DB storing which parts were correctly downloaded or missing.

Inputs:
- hl_inputs: temperature and pressure at "half-level"
- col_inputs : variables given at full level (see document)
- inter_inputs : cloud overlap parameter (how much the cloud parts overlap - summarized as f..)
- sca_inputs: scalar features, broadcasted on the entire vertical/column axis
See https://www.maelstrom-eurohpc.eu/content/docs/uploads/doc6.pdf for more information.

Outputs: flux variables, upward and downward, on short and long wavelengths
- flux_dn_sw: downward short wavelength flux
- flux_up_sw: upward short wavelength flux
- flux_dn_lw: downward long wavelength flux
- flux_up_lw: upward long wavelength flux

They are also preprocessed and sharded on-the-fly by the dataproc to data/processed. The file stats.pt stores statistics (mean, standard deviation and cardinality) of the shards.

Models
-----------------
The 1D U-Net takes the sharded data/processed as inputs (as specified in the config file), thanks to tailored data loaders.
