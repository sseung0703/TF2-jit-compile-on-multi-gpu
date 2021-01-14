## Tensorflow2 with jit compiling on multi-gpu training.
- Distributed learning with **jit compiling** and **distributed learning** on multi-gpu system.
- This repository is built by **cumstom layers** and **custom training loop**.

## Requirement
- Tensorflow >= 2.5

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
- I will share my results soon :).
