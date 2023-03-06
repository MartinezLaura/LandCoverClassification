# Land Cover Classification

Pipeline used for classification of Land Cover LUCASS classes on LUCAS point images. 
This code is realate to a paper on review process:

Title:
*Semantic segmentation and Random Forests for land cover classification on in-situ LUCAS landscape photos*

Abstract:
Spatially explicit information on land cover (LC) is commonly derived using remote sensing, but the lack of training data still remains a major challenge to produce accurate LC products. Here, we develop a computer vision methodology to extract LC information from photos from the Land Use-Land Cover Area Frame Survey (LUCAS).
We first selected a representative sample of 1120 photos covering eight major LC types over the European Union. We applied semantic segmentation on these photos using a neural network trained with the ADE20k dataset. For each photo, we extracted the original LC identified, the segmented classes, and the pixel count for each class.
Using the latter as input features, we trained a Random Forest to classify the LC. The results show a mean F1-score of 89\%, increasing to 93\% when the Wetland class is not considered. This approach holds promise for automated retrieval of LC information from geo-referenced photos

Submitted to:
Enviromental Modelling and Software

Pipeline:
![image](https://user-images.githubusercontent.com/24717718/223086522-c798faf1-a6f1-4f13-9554-ffe4262b5787.png)

