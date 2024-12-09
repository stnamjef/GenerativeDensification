<p align="center">
  <h1 align="center">
  Generative Densification: Learning to Densify Gaussians <br> for High-Fidelity Generalizable 3D Reconstruction
  </h1>
  <p align="center">
    <a href="https://github.com/stnamjef">Seungtae Nam*</a>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=VLzxTrAAAAAJ&hl=ko&oi=ao">Xiangyu Sun*</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/Gynjn">Gyungjin Kang</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/Younggeun-L">Younggeun Lee</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/ohsngjun">Seungjun Oh</a>
    &nbsp;·&nbsp;
    <a href="https://silverbottlep.github.io/">Eunbyung Park</a>
  </p>
  <h3 align="center">
  <a>Paper</a> | 
  <a href="https://stnamjef.github.io/GenerativeDensification/">Project Page</a> |
  <a href="https://huggingface.co/Xiang12yu/GDM-object/tree/main">Checkpoints</a> 
  </h3>
  <div style="padding-top: 20px; padding-bottom: 20px;">
  <img src="./assets/teaser.jpg" style="width: 100%;">
  <p style="line-height:1.25; text-align: justify; margin-top: 5px; margin-bottom: 5px">
    Our method selectively densifies (a) coarse Gaussians from generalized feed-forward models. 
    (c) The top K Gaussians with large view-space positional gradients are selected, and (d-e) their fine Gaussians are generated in each layer. 
    (g) The final Gaussians are obtained by combining (b) the remaining (non-selected) Gaussians with (f) the union of each layer's Gaussians.
  </p>
  </div>
  <h3 align="center">Code will be released soon!</h3>
</p>

## Checkpoints

For Generative Densification Object checkpoints, we provide two versions (w/ residual and w/o redisual) model, including both fast (30 epochs) and full (50 epochs) model.
We provide the checkpoints on [hugging-face](https://huggingface.co/Xiang12yu/GDM-object/tree/main).
