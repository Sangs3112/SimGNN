# [SimGNN](https://arxiv.org/abs/1808.05689):
`[WSDM 2019] SimGNN: A Neural Network Approach to Fast Graph Similarity Computation`

**This implementation is modeled exactly according to the code setup in the SimGNN paper**

![GitHub License](https://img.shields.io/github/license/Sangs3112/SimGNN)
![PyPI - Version](https://img.shields.io/pypi/v/pypi)

[Chinese](./README.md) | English
## directory structure:
```
SimGNN/
├── datasets/           
│   ├── AIDS700nef/
│   ├── ALKANE/
│   ├── IMDBMulti/
│   ├── ...(may be other datasets)
|   └── LINUX/
├── Logs/               # store log files
├── model/              # contain the model code
│   ├── layers.py       # including 'Att' and 'NTN' modules
│   ├── SimGNN.py       # the code of SimGNN
|   └── Trainer.py      # contain train, validate, test modules
├── utils/
│   ├── config.py       # System-level parameters, such as data set names
│   ├── config.yml      # Model level, data set level parameters, such as patience, num_features 
|   └── utils.py        # Tools, including loading data sets, loading configurations
└── main.py             
```
> You need to [download datasets](https://drive.google.com/drive/folders/1MOOUxxC_76Jseuc-JWaJ6B6LfU6-wNfR?usp=drive_link), which include `AIDS700nef`, `LINUX`, `IMDBMulti`, `ALKANE` datasets
>
> 1. Move the downloaded `datasets.tar.gz` compressed file to `SimGNN/`
>
> 2. Decompress: `tar -xvzf datasets.tar.gz`
>
> 3. After the decompression is complete, `cd datasets/` and use the same command to decompress the four datasets again
>
> P.s: In fact, if you don't download the dataset which I provided, you can just execute 'datasets/' in the 'SimGNN/' project root directory, and the 'GEDDataset' function will automatically download these dataset.

## Requirements:
```
pyyaml == 6.0.1
python == 3.9
numpy == 1.26
scipy == 1.11
tqdm == 4.66.1
texttable == 1.7
torch == 2.1.0
torch-geometric == 2.4.0
```

## run:
```
# AIDS700nef
python main.py
# LINUX
python main.py --dataset LINUX
# IMDBMulti
python main.py --dataset IMDBMulti
# ALKANE
python main.py --dataset ALKANE
```

## Result:
Please wait for several days~~


> If you like this project, please send us Stars ~