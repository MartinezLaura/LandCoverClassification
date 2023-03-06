# Land Cover Classification

Pipeline used for classification of Land Cover LUCASS classes on LUCAS point images. 
This code is realate to a paper on review process:

Title:

**Semantic segmentation and Random Forests for land cover classification on in-situ LUCAS landscape photos**

Abstract:

Spatially explicit information on land cover (LC) is commonly derived using remote sensing, but the lack of training data still remains a major challenge to produce accurate LC products. Here, we develop a computer vision methodology to extract LC information from photos from the Land Use-Land Cover Area Frame Survey (LUCAS).
We first selected a representative sample of 1120 photos covering eight major LC types over the European Union. We applied semantic segmentation on these photos using a neural network trained with the ADE20k dataset. For each photo, we extracted the original LC identified, the segmented classes, and the pixel count for each class.
Using the latter as input features, we trained a Random Forest to classify the LC. The results show a mean F1-score of 89\%, increasing to 93\% when the Wetland class is not considered. This approach holds promise for automated retrieval of LC information from geo-referenced photos

Submitted to: Enviromental Modelling and Software

Instalation: 
Please check the [requirements.txt]() to install the python packages needed

Pipeline:

![image](https://user-images.githubusercontent.com/24717718/223086522-c798faf1-a6f1-4f13-9554-ffe4262b5787.png)

Usage:
1. Inference of the Images with Deeplabv3, trained with ADE20k, see: [Notebook](https://github.com/MartinezLaura/LandCoverClassification/blob/5644f70523f2de2ea81129d75ec5851a25a36dc4/Extract_segmentation_ADE20k.ipynb)

2. Random Forest pipeline where the semantic segmentation of the previous step are used to classify the LC of the image, see: [Notebook](https://github.com/MartinezLaura/LandCoverClassification/blob/5644f70523f2de2ea81129d75ec5851a25a36dc4/RF-Training.ipynb)



Relevant links:

- [GLUON Toolkit](https://cv.gluon.ai/) 
- [GLUONCV git](https://github.com/dmlc/gluon-cv)

- [ADE20k Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
- [ADE20k Dataset git](https://github.com/CSAILVision/ADE20K)

LUCAS Harmonized database:
  - Article: [d’Andrimont, R., Yordanov, M., Martinez-Sanchez, L. et al. Harmonised LUCAS in-situ land cover and use database for field surveys from 2006 to 2018 in the European Union. Sci Data 7, 352 (2020). https://doi.org/10.1038/s41597-020-00675-z](https://www.nature.com/articles/s41597-020-00675-z)
    

  
