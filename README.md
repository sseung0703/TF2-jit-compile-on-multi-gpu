## Tensorflow2 with jit compiling on multi-gpu training.
- CIFAR and ILSVRC training code with **jit compiling** and **distributed learning** on the multi-GPU system.
- I highly recommend using Jit compiling because most of the algorithm is static and can be compiled, which gives memory usage reduction and training speed improvement.
- This repository is built by **custom layers** and **custom training loop** for my project, but if you only want to check how to use jit compiling with distributed learning, check 'train.py' and 'op_util.py'.

## Requirement
- **Tensorflow >= 2.5**
- Pillow

## How to run
- ILSVRC
```
python train.py --compile --gpu_id {} --dataset ILSVRC --data_path /path/to/your/ILSVRC/home --train_path /path/to/log
```

- CIFAR{10,100}
```
python train.py --compile --gpu_id {} --dataset CIFAR{10,100} --train_path /path/to/log
```

## Experimental results
- I used four 1080ti.
- Jit compiling gives a 40% speedup for training time.

   \       | Accuracy | Training time
------------| ------------- | -------------
Distributed only     | 75.83 | 94.61
Distributed with Jit     | 75.57 | 56.98

<p align="center">
<img width="900" alt="Training plot of ResNet-56 on ILSVRC-2012" src="https://user-images.githubusercontent.com/26036843/104848882-1c1b9400-592a-11eb-82ea-abb20bd339ba.png">
</p>
