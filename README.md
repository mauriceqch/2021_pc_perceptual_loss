# A deep perceptual metric for 3D point clouds

<p align="center">
  <img src="image.png?raw=true" alt="Paper illustration"/>
</p>


* **Authors**:
[Maurice Quach<sup>1</sup>](https://scholar.google.com/citations?user=atvnc2MAAAAJ),
[Aladine Chetouani<sup>1,2</sup>](https://scholar.google.fr/citations?user=goUbn88AAAAJ),
[Giuseppe Valenzise<sup>1</sup>](https://scholar.google.com/citations?user=7ftDv4gAAAAJ) and
[Frederic Dufaux<sup>1</sup>](https://scholar.google.com/citations?user=ziqjbTIAAAAJ)  
<sup>1</sup> Université Paris-Saclay, CNRS, CentraleSupélec, Laboratoire des signaux et systèmes, 91190 Gif-sur-Yvette, France  
<sup>2</sup> Laboratoire PRISME, Universite d’Orléans; Orléans, France
* **Funding**: ANR ReVeRy national fund (REVERY ANR-17-CE23-0020)
* **Paper**: https://arxiv.org/abs/2102.12839

## Prerequisites

* Python 3.6.9
* Tensorflow 1.15.0 with CUDA 10.0.130 and cuDNN 7.4.2
* packages in `requirements.txt`

*Note*: using a Linux distribution such as Ubuntu is highly recommended  

## Overview

The ICIP2020 subjective dataset is available as a repack at [this URL](https://drive.google.com/file/d/1MemLa255e0wrGXbWoVKDS5ghoqAmfWHw/view?usp=sharing).
The training dataset is available in the Git repository as `ModelNet40_200_pc512_oct3_4k.zip` and pretrained models are available in `src/data/model` and `src/data/model_tdf`.

The `src/analysis_icip2020_perry_quality.ipynb` notebook includes all the necessary code to reproduce the results of the paper.

Below is an overview of the repository:

	├── ModelNet40_200_pc512_oct3_4k.zip                     Training dataset
	├── requirements.txt                                     Package requirements
	└── src
	    ├── analysis_icip2020_perry_quality.ipynb            [Experiments] Main notebook
	    ├── data
	    │   ├── icip2020_deg_metrics.json                    [Data] Cache for computed metrics
	    │   ├── icip2020_degraded_pcs_features.csv           [Data] Experimental data
	    │   ├── icip2020_degraded_pcs_features_preds.csv     [Data] Full experimental data with MOS predictions
	    │   ├── model                                        [Data] Trained model for binary representation
	    │   └── model_tdf                                    [Data] Trained model for TDF representation
	    ├── ds_mesh_to_pc.py                                 [Training Dataset] Convert mesh to point cloud
	    ├── ds_pc_octree_blocks.py                           [Training Dataset] Divide a point cloud into octree blocks
	    ├── ds_select_largest.py                             [Training Dataset] Select the N largest files from a folder
	    ├── figs                                             [Data] Generated figures
	    ├── repack_icip20_perry_quality.py                   [Subjective Dataset] Script to repack ICIP2020 for reference (requires manual steps detailed in comments)
	    ├── repack_icip20_perry_quality.sh                   [Subjective Dataset] Script to repack ICIP2020 with preset parameters
	    ├── train_perceptual.ipynb                           [Training] Notebook to train the neural networks
	    └── utils
		├── cc_utils.py                                  CloudCompare utils
		├── color_space.py                               Color space conversions
		├── confidence_intervals.py                      Confidence intervals
		├── distance_grid.py                             Distance grids (TDF)
		├── features.py                                  Point Cloud features/metrics
		├── focal_loss.py                                Focal loss
		├── icip20_perry_quality.py                      ICIP2020 subjective dataset utils
		├── log_reg.py                                   Logistic regression
		├── matplotlib_utils.py                          Matplotlib utils
		├── na_bce.py                                    NaBCE implementation
		├── octree_coding.py                             Octree partitioning
		├── parallel_process.py                          Parallel processing utils
		├── pc_io.py                                     Point Cloud IO utils
		└── perceptual_model.py                          Perceptual loss neural network model

## Details

### ICIP2020 subjective dataset

The `repack_icip20_perry_quality.sh` file provides an example of how to repack this dataset.
Note that a number of manuel steps are necessary to repack the dataset; these steps are specified in the code as comments (see `src/utils/icip20_perry_quality.py`).
For ease of use, the repack can be downloaded following the instructions above.


### Training Dataset

To reproduce the dataset, download the ModelNet40 manually aligned dataset: [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu).  
Then, we generate the training dataset specified in our paper (block size 64) with the following commands:

    python ds_select_largest.py ~/data/datasets/ModelNet40 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 200
    python ds_mesh_to_pc.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 --vg_size 512
    python ds_pc_octree_blocks.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 --vg_size 512 --level 3 
    python ds_select_largest.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3_4k 4000

Note that this is only for reference as the training dataset is already included in the repository.

### Training

The notebook in `src/train_perceptual.ipynb` performs the training of the models described in the paper.
The parameters can be changed to perform training with binary or TDF representation.
Note that pretrained models are available in the repository.

## Citation

	@inproceedings{quach_deep_perceptual,
	  TITLE = {{A deep perceptual metric for 3D point clouds}},
	  AUTHOR = {Quach, Maurice and Chetouani, Aladine and Valenzise, Giuseppe and Dufaux, Fr{\'e}d{\'e}ric},
	  BOOKTITLE = {{Image Quality and System Performance, IS\&T International Symposium on Electronic Imaging (EI 2021)}},
	  YEAR = {2021},
	  MONTH = Jan,
	}

