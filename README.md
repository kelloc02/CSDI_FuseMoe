# CSDI_FuseMoe

## 1. FuseMoe环境安装

进入 `fusemoe` 文件夹后，按照其中的 `README` 步骤配置 Python 环境并安装依赖。


## 2. FuseMoe bash 运行脚本
跑mimiciv数据集的时候bash运行src/scripts/run_mimiciv_1.sh这个脚本
```bash
bash run_mimiciv.sh 
```
其中的参数modeltype和num_modalities是控制模态数量的 file_path改成你的数据放的位置
运行的时候可能遇到无法读取bioLongformer的情况，这个时候需要去官网下载这个模型到本地，然后在util的loadbert里面把路径改为本地模型的路径（我已经下载在pretain文件夹里面了


## 3. csdi对time series进行插补
首先通过from_mimic_to_csdi.ipynb这个脚本讲fuse训练的数据转化为txt（csdi训练data用的txt文件）
根据csdi的readme配置环境
```shell
python exe_ihm_1_xxx.py --testmissingratio [missing ratio] --nsample [number of samples]
```
使用脚本训练去噪模型

训练好了之后再进行对原始数据集进行插补

```shell
python exe_ihm_1_xxx.py --modelfolder pretrained --testmissingratio [missing ratio] --nsample [number of samples]
```
最后通过from_outputnsample_to_fuse_pkl.ipynb脚本用训练好的模型对完整数据集进行插补，并且转化为fuse可读的pkl版本