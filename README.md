# Screening the stones of Venice: Mapping social perceptions of cultural significance through graph-based semi-supervised classification

This is the Code and Dataset for the Paper '*Screening the stones of Venice: Mapping social perceptions of cultural significance through graph-based semi-supervised classification*' published in *ISPRS Journal of Photogrammetry and Remote Sensing* showing the mapping of cultural significance in the city of Venice.

## Cite as

Bai, N., Nourian, P., Luo, R., Cheng, T., & Pereira Roders, A. (2023). Screening the stones of Venice: Mapping social perceptions of cultural significance through graph-based semi-supervised classification. ISPRS Journal of Photogrammetry and Remote Sensing, 203, 135-164. https://doi.org/10.1016/j.isprsjprs.2023.07.018

```
@article{Bai2023StoneVenice, 
    title = {Screening the stones of Venice: Mapping social perceptions of cultural significance through graph-based semi-supervised classification}, 
    author = {Nan Bai and Pirouz Nourian and Renqian Luo and Tao Cheng and {Pereira Roders}, Ana},
    year = {2023},
    doi = {10.1016/j.isprsjprs.2023.07.018},
    volume = {203},
    pages = {135--164},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    issn = {0924-2716},
    publisher = {Elsevier},
}

```

## Table of Content
#### [Requirement and Dependency](#requirement)
#### [Workflow and Dataset](#workflow)
#### [Case Studies](#case)
#### [Dataset Summary](#dataset)

The following sections about the workflow can be skipped for those who only intend to use the provided datasets.
#### [Raw Data Collection](#raw)
#### [Multi-modal Feature Generation](#feature)
#### [Label Generation](#label)
#### [Multi-graph Construction](#graph)
#### [Acknowledgements and License](#license)

## Requirement and Dependency<a name="requirement"></a>
deep_translator == 1.7.0

facenet_pytorch == 2.5.2

fastai == 2.5.3

flickrapi == 2.4.0

matplotlib == 3.5.1

networkx == 2.6.3

numpy == 1.22.2

opencv-python == 4.5.5.62

osmnx == 1.1.2

pandas == 1.4.0

pillow == 9.0.1

[places365](https://github.com/CSAILVision/places365) (please download the repository ```places365``` and put under the root as ```./places365```)

scipy == 1.8.0

scikit-learn == 1.0.2

torch == 1.10.2+cu113

torchvision == 0.11.3+cu113

transformers == 4.16.2

[WHOSe_Heritage](https://github.com/zzbn12345/WHOSe_Heritage) (please download the repository ```WHOSe_Heritage``` and put under the root as ```./WHOSe_Heritage```)

## Workflow and Dataset<a name="workflow"></a>
This project provides a workflow to to construct graph-based multi-modal datasets HeriGraph concerning heritage values and attributes using data from social media platform Flickr.
The workflow is illustrated as follows: