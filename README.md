# Happy: A Debiased Learning Framework for Continual Generalized Category Discovery

Official implementation of our NeurIPS 2024 paper: **Happy: A Debiased Learning Framework for Continual Generalized Category Discovery [[arXiv]](https://arxiv.org/abs/2410.06535)[[NeurIPS 2024]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5ae0f7cfd65d8e2b39da4177fef82015-Abstract-Conference.html)**

We study the under-explored setting of continual generalized category discovery (C-GCD) as follows:

![diagram](assets/CGCD-setting.png)

:bookmark: The core difference between C-GCD and class-incremental learning (CIL) is that C-GCD is unsupervised continual learning, while CIL is purely supervised. At each continual stage of C-GCD, unlabeled training data could contain both old and new classes.

We introduce our method: **Happy**, which is characterized by <ins>H</ins>ardness-<ins>a</ins>ware <ins>p</ins>rototype sampling and soft entro<ins>py</ins> regularization, as follows:

![diagram](assets/Happy.png)



## Running :running:

### 1. Datasets

We conduct experiments on 6 datasets:

* Generic datasets: CIFAR-100, TinyImageNet and ImageNet-100
* Fine-grained datasets: [CUB](https://drive.google.com/drive/folders/1kFzIqZL_pEBVR7Ca_8IKibfWoeZc3GT1), [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

### 2. Config

Set paths to datasets in `config.py`

### 3. Stage-0: Labeled Training

**CIFAR100**

```shell
CUDA_VISIBLE_DEVICES=0 python train_happy.py --dataset_name 'cifar100' --batch_size 128 --transform 'imagenet' --lr 0.1 --memax_weight 1 --eval_funcs 'v2' --num_old_classes 50 --prop_train_labels 0.8 --train_session offline --epochs_offline 100 --continual_session_num 5 --online_novel_unseen_num 400 --online_old_seen_num 25 --online_novel_seen_num 25
```

**CUB**

```shell
CUDA_VISIBLE_DEVICES=0 python train_happy.py --dataset_name 'cub' --batch_size 128 --transform 'imagenet' --lr 0.1 --memax_weight 2 --eval_funcs 'v2' --num_old_classes 100 --prop_train_labels 0.8 --train_session offline --epochs_offline 100 --continual_session_num 5 --online_novel_unseen_num 25 --online_old_seen_num 5 --online_novel_seen_num 5
```

:warning: Please specify the args `--train_session` as `offline`

### 4. Stage-1 $\sim$ T: Continual Training

**CIFAR100**

```shell
CUDA_VISIBLE_DEVICES=0 python train_happy.py --dataset_name 'cifar100' --batch_size 128 --transform 'imagenet' --warmup_teacher_temp 0.05 --teacher_temp 0.05 --warmup_teacher_temp_epochs 10 --lr 0.01 --memax_old_new_weight 1 --memax_old_in_weight 1 --memax_new_in_weight 1 --proto_aug_weight 1 --feat_distill_weight 1 --radius_scale 1.0 --hardness_temp 0.1 --eval_funcs 'v2' --num_old_classes 50 --prop_train_labels 0.8 --train_session online --epochs_online_per_session 30 --continual_session_num 5 --online_novel_unseen_num 400 --online_old_seen_num 25 --online_novel_seen_num 25 --init_new_head --load_offline_id Old50_Ratio0.8_20240418-002807 --shuffle_classes --seed 0
```

**CUB**

```shell
CUDA_VISIBLE_DEVICES=0 python train_happy.py --dataset_name 'cub' --batch_size 128 --transform 'imagenet' --warmup_teacher_temp 0.05 --teacher_temp 0.05 --warmup_teacher_temp_epochs 10 --lr 0.01 --memax_old_new_weight 1 --memax_old_in_weight 1 --memax_new_in_weight 1 --proto_aug_weight 1 --feat_distill_weight 1 --radius_scale 1.0 --eval_funcs 'v2' --num_old_classes 100 --prop_train_labels 0.8 --train_session online --epochs_online_per_session 20 --continual_session_num 5 --online_novel_unseen_num 25 --online_old_seen_num 5 --online_novel_seen_num 5 --init_new_head --load_offline_id Old100_Ratio0.8_20240506-165445 --shuffle_classes --seed 0
```

:warning: Please specify the args `--train_session` as `online`

:warning: Please change the args `--load_offline_id` according to offline training save path

:warning: Please keep the four args `--continual_session_num`, `--online_novel_unseen_num`, `--online_old_seen_num` and `--online_novel_seen_num` the same as offline training stage.



## Citing this work :clipboard:

```bibtex
@article{ma2024happy,
  title={Happy: A Debiased Learning Framework for Continual Generalized Category Discovery},
  author={Ma, Shijie and Zhu, Fei and Zhong, Zhun and Liu, Wenzhuo and Zhang, Xu-Yao and Liu, Cheng-Lin},
  journal={arXiv preprint arXiv:2410.06535},
  year={2024}
}
```



## Acknowledgements :gift:

In building this repository, we reference [SimGCD](https://github.com/CVMI-Lab/SimGCD).



## License :white_check_mark:

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mashijie1028/Happy-CGCD/blob/main/LICENSE) file for details.



## Contact :email:

If you have further questions or discussions, feel free to contact me:

Shijie Ma (mashijie2021@ia.ac.cn)
