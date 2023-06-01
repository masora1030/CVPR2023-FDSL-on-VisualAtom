# FDSL on VisualAtom

## TOC
- [Summary](#summary)
- [Updates](#updates)
- [Citation](#citation)
- [Requirements](#requirements)
- [VisualAtom Construction](#visualatom-construction-readme)
- [Pre-training](#pre-training)
  - [Pre-training with shard dataset](#pre-training-with-shard-dataset)
  - [Pre-trained models](#pre-trained-models)
- [Fine-Tuning](#fine-tuning)
- [Acknowledgements](#acknowledgements)
- [Terms of use](#terms-of-use)


## Summary
The repository contains VisualAtom Construction, Pre-training and Fine-tuning in Python/PyTorch.
The repository is based on the paper:
Sora Takashima, Ryo Hayamizu, Nakamasa Inoue, Hirokatsu Kataoka and Rio Yokota,
"Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves", IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023.
[[Project](https://masora1030.github.io/Visual-Atoms-Pre-training-Vision-Transformers-with-Sinusoidal-Waves/)] 
[[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Takashima_Visual_Atoms_Pre-Training_Vision_Transformers_With_Sinusoidal_Waves_CVPR_2023_paper.pdf)] 
[[Dataset](#visualatom-construction-readme)] 
[[Poster](https://cvpr2023.thecvf.com/media/PosterPDFs/CVPR%202023/22854.png?t=1685632285.9741583)] 
[[Supp](https://masora1030.github.io/Visual-Atoms-Pre-training-Vision-Transformers-with-Sinusoidal-Waves/CVPR2023_VisualAtom_FDSL_Supplementary_Material.pdf)]
<!-- TODO [[Oral](http://hirokatsukataoka.net/pdf/cvpr22_kataoka_oral.pdf)]  -->

## Updates
<!-- TODO update -->
**Update (Mar. 24, 2023)**
* VisualAtom Construction & Pre-training & Fine-tuning scripts are shared here.
* Downloadable pre-training models : [[Pre-trained Models](https://drive.google.com/drive/folders/1OUSmOt01K-nDsr55-w1YweGDfsVtgz4N?usp=share_link)]

**Update (Feb. 28, 2023)**
* Our paper was accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023. 
[[PDF (preprint)](https://arxiv.org/pdf/2303.01112.pdf)] 

## Citation

<!-- TODO update pages -->
If you use this scripts, please cite the following paper:
```bibtex
@InProceedings{takashima2023visual,
    author    = {Sora Takashima, Ryo Hayamizu, Nakamasa Inoue, Hirokatsu Kataoka and Rio Yokota},
    title     = {Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18579-18588}
}
``` 

<!-- ```bibtex
@article{takashima2023visual,
  title={Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves},
  author={Takashima, Sora and Hayamizu, Ryo and Inoue, Nakamasa and Kataoka, Hirokatsu and Yokota, Rio},
  journal={arXiv preprint arXiv:2303.01112},
  year={2023}
}
```  -->

## Requirements

* Python 3.x (worked at 3.8.2)
* CUDA (worked at 10.2)
* CuDNN (worked at 8.0)
* NCCL (worked at 2.7)
* OpenMPI (worked at 4.1.3)
* Graphic board (worked at single/four NVIDIA V100)

Please install packages with the following command.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## VisualAtom Construction ([README](visual_atomic_renderer/README.md))

```
$ cd visual_atomic_renderer
$ bash make_VisualAtom.sh
```

You can also download raw VisualAtom-1k here : [zenodo](https://zenodo.org/record/7945009)

## Pre-training

We used almost the same scripts as in [Kataoka_2022_CVPR](https://github.com/masora1030/CVPR2022-Pretrained-ViT-PyTorch) for our pre-training.

Run the python script ```pretrain.py```, you can pre-train with your dataset.

Basically, you can run the python script ```pretrain.py``` with the following command.

- Example : with deit_base, pre-train VisualAtom-21k, 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpirun -npernode 4 -np 4 \
      python pretrain.py /PATH/TO/VisualAtom21000 \
        --model deit_base_patch16_224 --experiment pretrain_deit_base_VisualAtom21000_1.0e-3 \
        --input-size 3 224 224 \
        --sched cosine_iter --epochs 90 --lr 1.0e-3 --weight-decay 0.05 \
        --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 \
        --batch-size 64 --opt adamw --num-classes 21000 \
        --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --remode pixel --interpolation bicubic --hflip 0.0 \
        -j 16 --pin-mem --eval-metric loss \
        --interval_saved_epochs 10 --output ./output/pretrain \
        --amp \
        --log-wandb
    ```

    > **Note**
    > 
    > - ```--batch-size``` means batch size per process. In the above script, for example, you use 4 GPUs (4 process), so overall batch size is 64×4(=256).
    > 
    > - In our paper research, for datasets with more than 21k categories, we basically pre-trained with overall batch size of 8192 (64×128).
    > 
    > - If you wish to distribute pre-train across multiple nodes, the following must be done.
    >   - Set the `MASTER_ADDR` environment variable which is the IP address of the machine in rank 0.
    >   - Set the ```-npernode``` and ```-np``` arguments of ```mpirun``` command.
    >     - ```-npernode``` means GPUs (processes) per node and ```-np``` means overall the number of GPUs (processes).

Or you can run the job script ```scripts/pretrain.sh``` (support multiple nodes training with OpenMPI). 
Note, the setup is multiple nodes and using a large number of GPUs (32 nodes and 128 GPUs for pre-train).

When running with the script above, please make your dataset structure as following.

```misc
/PATH/TO/VisualAtom21000/
    image/
        00000/
        00000_0000.png
        00000_0001.png
        ...
        00001/
        00001_0000.png
        00001_0001.png
        ...
        ...
    ...
```

After above pre-training, trained models are created like ```output/pretrain/pretrain_deit_base_VisualAtom21000_1.0e-3/model_best.pth.tar``` and ```output/pretrain/pretrain_deit_base_VisualAtom21000_1.0e-3/last.pth.tar```. 
Moreover, you can resume the training from a checkpoint by setting ```--resume``` parameter.

Please see the script and code files for details on each arguments.

### Pre-training with shard dataset

Shard dataset is also available for accelerating IO processing. 
To make shard dataset, please refer to this repository: https://github.com/webdataset/webdataset. 
Here is an Example of training with shard dataset.

- Example : with deit_base, pre-train VisualAtom-21k(shard), 4 GPUs (Batch Size = 64×4 = 256)

    ```bash
    $ mpirun -npernode 4 -np 4 \
      python pretrain.py /NOT/WORKING \
        -w --trainshards /PATH/TO/VisualAtom21000/SHARDS-{000000..002099}.tar \
        --model deit_base_patch16_224 --experiment pretrain_deit_base_VisualAtom21000_1.0e-3_shards \
        --input-size 3 224 224 \
        --sched cosine_iter --epochs 90 --lr 1.0e-3 --weight-decay 0.05 \
        --min-lr 1.0e-5 --warmup-lr 1.0e-6 --warmup-iter 5000 --cooldown-epochs 0 \
        --batch-size 64 --opt adamw --num-classes 21000 \
        --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
        --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --remode pixel --interpolation bicubic --hflip 0.0 \
        -j 1 --eval-metric loss --no-prefetcher \
        --interval_saved_epochs 10 --output ./output/pretrain \
        --amp \
        --log-wandb
    ```
​
When running with the script above with shard dataset, please make your shard dataset structure as following.

```misc
/PATH/TO/VisualAtom21000/
    SHARDS-000000.tar
    SHARDS-000001.tar
    ...
    SHARDS-002099.tar
```

### Pre-trained models

Our pre-trained models are available in this [[Link](https://drive.google.com/drive/folders/1OUSmOt01K-nDsr55-w1YweGDfsVtgz4N?usp=share_link)].

We have mainly prepared three different pre-trained models. 
These pre-trained models are ViT-Tiny/Base (patch size of 16, input size of 224) pre-trained on VisualAtom-1k/21k and Swin-Base (patch size of 7, window size of 7, input size of 224) pre-trained on VisualAtom-21k.

```misc
vit_tiny_with_visualatom_1k.pth.tar: timm model is deit_tiny_patch16_224, pre-trained on VisualAtom-1k
vit_base_with_visualatom_21k.pth.tar: timm model is deit_base_patch16_224, pre-trained on VisualAtom-21k
swin_base_with_visualatom_21k.pth.tar: timm model is swin_base_patch4_window7_224, pre-trained on VisualAtom-21k
```

## Fine-tuning

We used fine-tuning scripts based on [Nakashima_2022_AAAI](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch).

Run the python script ```finetune.py```, you additionally train other datasets from your pre-trained model.

In order to use the fine-tuning code, you must prepare a fine-tuning dataset (e.g., CIFAR-10/100, ImageNet-1k, Pascal VOC 2012). 
You should set the dataset as the following structure.

```misc
/PATH/TO/DATASET/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
    ...
  val/
    class1/
      img3.jpeg
      ...
    class2/
      img4.jpeg
      ...
    ...
```

Basically, you can run the python script ```finetune.py``` with the following command.

- Example : with deit_base, fine-tune CIFAR10 from pre-trained model (with VisualAtom-21k), 8 GPUs (Batch Size = 96×8 = 768)

    ```bash
    $ mpiexec -npernode 4 -np 8 \
        python -B finetune.py data=colorimagefolder \
        data.baseinfo.name=CIFAR10 data.baseinfo.num_classes=10 \
        data.trainset.root=/PATH/TO/CIFAR10/TRAIN data.baseinfo.train_imgs=50000 \
        data.valset.root=/PATH/TO/CIFAR10/VAL data.baseinfo.val_imgs=10000 \
        data.loader.batch_size=96 \
        ckpt=./output/pretrain/pretrain_deit_base_VisualAtom21000_1.0e-3/last.pth.tar \
        model=vit model.arch.model_name=vit_base_patch16_224 \
        model.optim.optimizer_name=sgd model.optim.learning_rate=0.01 \
        model.optim.weight_decay=1.0e-4 model.scheduler.args.warmup_epochs=10 \
        epochs=1000 mode=finetune \
        logger.entity=YOUR_WANDB_ENTITY_NAME logger.project=YOUR_WANDB_PROJECT_NAME logger.group=YOUR_WANDB_GROUP_NAME \
        logger.experiment=finetune_deit_base_CIFAR10_batch768_from_VisualAtom21000_1.0e-3 \
        logger.save_epoch_freq=100 \
        output_dir=./output/finetune
    ```

Or you can run the job script ```scripts/finetune.sh``` (support multiple nodes training with OpenMPI).

Please see the script and code files for details on each arguments.

## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST) and Tokyo Institute of Technology (TITech) are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the images and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.
