# Multiple Adversarial Learning based Angiography Reconstruction for Ultra-low-dose Contrast Medium CT (J-BHI 2022)
Weiwei Zhang, Zhen Zhou, Zhifan Gao, Guang Yang, Lei Xu, Weiwen Wu, and Heye Zhang

# Abstract
<em>
Iodinated contrast medium (ICM) dose reduction is beneﬁcial for decreasing potential health risk to renal-insufﬁciency patients in CT scanning. Due to the lowintensity vessel in ultra-low-dose-ICM CT angiography, it cannot provide clinical diagnosis of vascular diseases. Angiography reconstruction for ultra-low-dose-ICM CT can enhance vascular intensity for directly vascular diseases diagnosis. However, the angiography reconstruction is challenging since patient individual differences and vascular disease diversity. In this paper, we propose a Multiple Adversarial Learning based Angiography Reconstruction (i.e., MALAR) framework to enhance vascular intensity. Speciﬁcally, a bilateral learning mechanism is developed for mapping a relationship between source and target domains rather than the image-to-image mapping. Then, a dual correlation constraint is introduced to characterize both distribution uniformity from across-domain features and sample inconsistency with domain simultaneously. Finally, an adaptive fusion module by combining multiscale information and long-range interactive dependency is explored to alleviate the interference of high-noise metal. Experiments are performed on CT sequences with different ICM doses. Quantitative results based on multiple metrics demonstrate the effectiveness of our MALAR on angiography reconstruction. Qualitative assessments by radiographers conﬁrm the potential of our MALAR for the clinical diagnosis of vascular diseases.
</em>

# Network Architecture
![Flow_chart](https://user-images.githubusercontent.com/64700979/194312703-2a28018f-a050-4413-8917-98e181258954.png)

# Requirements
```
python=3.7.1 tensorflow=1.15.0
```

# Our Related Work
* Artificial intelligence–based full aortic CT angiography imaging with ultra-low-dose contrast medium: a preliminary study, European Radiology 2022. [Paper](https://link.springer.com/article/10.1007/s00330-022-08975-1)