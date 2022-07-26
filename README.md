### Official repository for TIPS: Text-Induced Pose Synthesis

*Accepted in the European Conference on Computer Vision (ECCV) 2022*

[![badge_torch](https://img.shields.io/badge/made_with-PyTorch-red?style=flat-square&logo=PyTorch)](https://pytorch.org/)
[![badge_colab](https://img.shields.io/badge/Demo-Open_in_Colab-blue?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/prasunroy/tips/blob/main/notebooks/TIPS_demo.ipynb)

![teaser](https://github.com/prasunroy/tips/blob/main/docs/static/teaser.svg)

### Abstract
<p align="justify">
  In computer vision, human pose synthesis and transfer deal with probabilistic image generation of a person in a previously unseen pose from an already available observation of that person. Though researchers have recently proposed several methods to achieve this task, most of these techniques derive the target pose directly from the desired target image on a specific dataset, making the underlying process challenging to apply in real-world scenarios as the generation of the target image is the actual aim. In this paper, we first present the shortcomings of current pose transfer algorithms and then propose a novel text-based pose transfer technique to address those issues. We divide the problem into three independent stages: (a) text to pose representation, (b) pose refinement, and (c) pose rendering. To the best of our knowledge, this is one of the first attempts to develop a text-based pose transfer framework where we also introduce a new dataset DF-PASS, by adding descriptive pose annotations for the images of the DeepFashion dataset. The proposed method generates promising results with significant qualitative and quantitative scores in our experiments.
</p>

<br>

### Network Architecture
![network_architecture](https://github.com/prasunroy/tips/blob/main/docs/static/network_architecture.svg)
<p align="justify">
  The pipeline is divided into three stages. In stage 1, we estimate the target pose keypoints from the corresponding text description embedding. In stage 2, we regressively refine the initial estimation of the facial keypoints and obtain the refined target pose keypoints. Finally, in stage 3, we render the target image by conditioning the pose transfer on the source image.
</p>

<br>

### Generation Results
![results](https://github.com/prasunroy/tips/blob/main/docs/static/results.svg)
<p align="justify">
  Keypoints-guided methods tend to produce structurally inaccurate results when the physical appearance of the target pose reference significantly differs from the condition image. This observation is more frequent for the <i>out of distribution</i> target poses than the <i>within distribution</i> target poses. On the other hand, the existing text-guided method occasionally misinterprets the target pose due to a limited set of basic poses used for pose representation. The proposed text-guided technique successfully addresses these issues while retaining the ability to generate visually decent results close to the keypoints-guided baseline.
</p>

<br>

### Try the TIPS inference pipeline demo in Colab
[![badge_colab](https://img.shields.io/badge/Demo-Open_in_Colab-blue?style=flat-square&logo=googlecolab)](https://colab.research.google.com/github/prasunroy/tips/blob/main/notebooks/TIPS_demo.ipynb)

[![tips_demo](https://github.com/prasunroy/tips/blob/main/docs/static/colab_enjoyer.svg)](https://colab.research.google.com/github/prasunroy/tips/blob/main/notebooks/TIPS_demo.ipynb)

<br>

### External Links
<h4>
  <a href="https://prasunroy.github.io/tips">Project</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="http://arxiv.org/abs/2207.11718">arXiv</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/17cvo22Eh_Z_S6fb-J-c6qw97WH6UeIHo">DF-PASS Dataset</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://drive.google.com/drive/folders/1DwEcAPeYkXUNQ_SBhSJpydaLBTjh3_ms">Pretrained Models</a>&nbsp;&nbsp;&bull;&nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/prasunroy/tips/blob/main/notebooks/TIPS_demo.ipynb">Colab Demo</a>
</h4>

<br>

### Citation
```
@inproceedings{roy2022tips,
  title     = {TIPS: Text-Induced Pose Synthesis},
  author    = {Roy, Prasun and Ghosh, Subhankar and Bhattacharya, Saumik and Pal, Umapada and Blumenstein, Michael},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month     = {October},
  year      = {2022}
}
```

<br>

### Related Publications

[1] [Multi-scale Attention Guided Pose Transfer](https://arxiv.org/abs/2202.06777) (arXiv 2022).

[2] [Scene Aware Person Image Generation through Global Contextual Conditioning](https://arxiv.org/abs/2206.02717) (ICPR 2022).

[3] [Text Guided Person Image Synthesis](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_Text_Guided_Person_Image_Synthesis_CVPR_2019_paper.html) (CVPR 2019).

[4] [Progressive Pose Attention Transfer for Person Image Generation](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Progressive_Pose_Attention_Transfer_for_Person_Image_Generation_CVPR_2019_paper.html) (CVPR 2019).

[5] [DeepFashion: Powering Robust Clothes Recognition and Retrieval With Rich Annotations](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html) (CVPR 2016).

<br>

### License
```
Copyright 2022 by the authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

>The DF-PASS dataset and the pretrained models are released under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) license.

<br>

##### Made with :heart: and :pizza: on Earth.
