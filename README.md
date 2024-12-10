<p align="center">
  <h1 align="center">
  Generative Densification: Learning to Densify Gaussians <br> for High-Fidelity Generalizable 3D Reconstruction
  </h1>
  <p align="center">
    <a href="https://github.com/stnamjef">Seungtae Nam*</a>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=VLzxTrAAAAAJ&hl=ko&oi=ao">Xiangyu Sun*</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/Gynjn">Gyeongjin Kang</a>
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
  <div style="padding-top: 5px;"></div>
</p>

|![teaser](./assets/teaser.jpg)|
|:--:|
| *Our method selectively densifies coarse Gaussians generate by generalized feed-forward models* |

<!-- ## Installation
```
docker build -t generative_densification:0.0.1 -f Dockerfile .
docker run -it -v $(pwd):/workspace --gpus all --ipc host --name generative_densification generative_densification:0.0.1
``` -->

## Datasets
* Our object-level model is trained on Gobjaverse training set, provided by [LaRa](https://github.com/autonomousvision/LaRa?tab=readme-ov-file#dataset).
* Note: 
  * The Gobjaverse dataset requires 1.4TB of storage.
  * We assume the datasets are in the `./GenerativeDensification/dataset`.

```shell
GenerativeDensification
├── dataLoader
├── dataset
│   ├── gobjaverse
│   │   ├── gobjaverse_part_01.h5
│   │   ...
│   │
│   ├── google_scanned_objects
│   │   ├── 2_of_Jenga_Classic_Game
│   │   ...
│   ...
├── lightning
...
```

## Training
* You can enable residual learning by setting `model.enable_residual_attribute=True`.
```
python train_lightning.py \
train_dataset.data_root=./dataset/gobjaverse/gobjaverse.h5 \
test_dataset.data_root=./dataset/gobjaverse/gobjaverse.h5 \
model.enable_residual_attribute=False
```

## Evaluation
* We provide two [checkpoints](https://huggingface.co/Xiang12yu/GDM-object/tree/main) (w/ residual learning and w/o it) for our object-level models.
* Note: 
  * The checkpoint 'epoch=49.ckpt' corresponds to 'Ours' model in the paper.
  * The checkpoint 'epoch=49_residual.ckpt' corresponds to 'Ours (w/ residual)' model in the paper.
```
python eval_all.py
```

## Acknowledgements
Our work is built upon the following projects.
We thank all the authors for making their amazing works publicly available.
* [LaRa](https://github.com/autonomousvision/LaRa)
* [MVSplat](https://github.com/donydchen/mvsplat)
* [MVSplat360](https://github.com/donydchen/mvsplat360)
* [SplatterImage](https://github.com/szymanowiczs/splatter-image)
* [AbsGS](https://github.com/TY424/AbsGS)

<!-- ## Citation -->
