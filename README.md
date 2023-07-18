# Attention_Res_BEVNet

Source code for our work **"Ground Mobile Robot Traversable Area Recognition Based on 3D CNN and Attention Mechanism"**

## TODOs
- [x] source code upload
- [x] model weights upload
- [x] dataset upload
- [ ] Instructions on dataset generation
- [x] Instructions on inference

## Datasets
Datasets should be put inside `data/`. For example, `data/rellis_4class_100x100`.

### Download links

* SemanticKITTI: [Google Drive](https://drive.google.com/file/d/1PsU0v5wC6n5gn7sK7uJS6p_8zbeK8szu/view?usp=sharing)
* RELLIS: [Google Drive](https://drive.google.com/file/d/1oOGq1e5GK-TJ_J0D4L-JNpyCv40JNYPo/view?usp=sharing)

## Running the pretrained models

### Model weights

SemanticKITTI
* Single-frame: [Google Drive](https://drive.google.com/file/d/1vtuowdWECV3agyFPQllArxpyKXio4WHl/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1TALbb1qz9NTB6bmre5UBtQ?pwd=rdl4)
* Recurrent: [Google Drive](https://drive.google.com/file/d/1jHOyH5EV_SCJupOoPpBD_0DCFhVWm-ds/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1LsipUznc6Mi2rLSDrk4LcA?pwd=ocyi)

RELLIS
* Single-frame: [Google Drive](https://drive.google.com/file/d/16WzFsO_d96k4ASDpFWsEQn6IhGbXoT59/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1XVr6t6N7fcSP-kCX992lzQ?pwd=pxgq)
* Recurrent: [Google Drive](https://drive.google.com/file/d/1c-_dCEJvRKbefRPuSYDMIpd8YZbeyMMM/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1jGpio4hy9npchNGoDoZ8Ug?pwd=6vx4)

### Run the models
First extract the model weights
``` shell
cd /path/to/bevnet/experiments
unzip /path/to/zip/file
```

To run the models on the validation set, `cd` to `bevnet/bevnet`, then run
``` shell
# Single-frame model
python test_single.py --model_file ../experiments/rellis4_100/single/resSparse-attention-logs/model.pth.15 --test_env rellis4

# Recurrent model
python test_recurrent.py --model_file ../experiments/rellis4_100/recurrent/resSparse-attention-logs/model.pth.3 --test_env rellis4
```


## Training

### AS_BEVNet-S
Example:
```
cd experiments
bash train_rellis4-unknown_single.sh rellis4_100/single/resSparse.yaml <tag> arg1 arg2 ...
```
Logs and model weights will be stored in a subdirectory of the config file like this:
`experiments/rellis4_100/single/resSparse-<tag>-logs/`
* `<tag>` is useful when you want to use the same config file but different hyperparameters. For example, if you
  want to do some debugging you can use set `<tag>` to `debug`.
* `arg1 arg2 ...` are command line arguments supported by `train_single.py`. For example, you can pass
  `--batch_size=4 --log_interval=100`, etc.


### AS_BEVNet-R
The command line formats are the same as AS_BEVNet-S
Example:
```
cd experiments
bash train_rellis4-unknown_recurrent.sh rellis4_100/recurrent/resSparse.yaml <tag> \
--n_frame=6 --seq_len=20 --frame_strides 1 10 20 \
--resume rellis4_100/single/resSparse-attention-logs/model.pth.15 \
--resume_epoch 0
```
Logs and model weights will be stored in a subdirectory of the config file
`experiments/rellis4_100/recurrent/resSparse-<tag>-logs/`.
