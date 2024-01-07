To reproduce all our results as reported bellow, you can use our [pretrained modeL](https://drive.google.com/file/d/1S2ZQmB7_OA4HgdP7A6fGGsPCmknR0MpB/view?usp=drive_link) and our source code.

| Method | All | Thing | Stuff | Single | Plural |
|--------|-----|-------|-------|--------|--------|
| MCN    | 54.2| 48.6  | 61.4  | 56.6   | 38.8   |
| PNG    | 55.4| 56.2  | 54.3  | 56.2   | 48.8   |
| EPNG   | 49.7| 45.6  | 55.5  | 50.2   | 45.1   |
| PPMN   | 59.4| 57.2  | 62.5  | 60.0   | 54.0   |
| XPNG   | 63.3| 61.1  | 66.2  | 64.0   | 56.4   |



## Environments

You need the Pytorch >= 1.10.1, and follow the command that:
```html
conda create -n xpng python=3.8
conda activate xpng
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```
After that, please follow the [instruction of detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) to install detectron2 for the enviroment with:
```html
Assuming the absolute path to the project's root directory is /xpng
cd /xpng
python -m pip install -e detectron2
```
## Dataset

Download the 2017 MSCOCO Dataset from its [official webpage](https://cocodataset.org/#download). You will need the train and validation splits' images and panoptic segmentations annotations. Download the Panoptic Narrative Grounding Benchmark from the PNG's [project webpage](https://bcv-uniandes.github.io/panoptic-narrative-grounding/#downloads).

Organize the files as follows:
```html
XPNG
|_ panoptic_narrative_grounding
|_ dataset
| |_ coco
| |  |_ val2017
| |  |_ train2017
|_ annotations
| |_ png_coco_train2017.json
| |_ png_coco_val2017.json
| |_ panoptic_segmentation
| | |_ train2017
| | |_ val2017
| |_ panoptic_train2017.json
| |_ panoptic_val2017.json
|_ data
```

## Pretrained Bert Model and PFPN

The pre-trained checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/1xrJmbBJ35M4O1SNyzb9ZTsvlYrwmkAph), and the folder should be like:

```html
pretrained_models
|_fpn
|  |_model_final_cafdb1.pkl
|_bert
|  |_bert-base-uncased
|  |  |_pytorch_model.bin
|  |  |_bert_config.json
|  |_bert-base-uncased.txt
```



## Evaluate

Set args.test_only=True in main.py with --ckpt_path as the path to the model's pth parameter file.
cd/xpng
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --nnodes=1   --master_port 29614 main.py
```
```
## Train
Set args.test_only=False in main.py  
cd/xpng
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --nnodes=1   --master_port 29615 main.py
```

```
## Acknowledge

Some of the codes are built upon [K-Net](https://github.com/ZwwWayne/K-Net) and [PNG](https://github.com/BCV-Uniandes/PNG). Thanks them for their great works!


