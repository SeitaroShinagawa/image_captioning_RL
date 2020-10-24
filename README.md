# image_captioning_RL
image-captioning implementations to learn Reinforcement learning

Environment
- pytorch 1.6


# HOW TO USE
1. Download dataset (Karpathy split)

```
wget https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip  #729.4MB
unzip coco.zip
```

2. Create vocabulary

```
python vocab.py
```

3. Training starts by editing `run_train.sh` and run it (see available parameters by `python train.py --help`)

```
bash run_train.sh
```


