# CSDI_FuseMoe

## 1. 环境安装

进入 `fusemoe` 文件夹后，按照其中的 `README` 步骤配置 Python 环境并安装依赖。


## 2. bash 运行脚本
跑mimiciv数据集的时候bash运行src/scripts/run_mimiciv_1.sh这个脚本
```bash
bash run_mimiciv.sh 
```
其中的参数modeltype和num_modalities是控制模态数量的 file_path改成你的数据放的位置
运行的时候可能遇到无法读取bioLongformer的情况，这个时候需要去官网下载这个模型到本地，然后在util的loadbert里面把路径改为本地模型的路径（我已经下载在pretain文件夹里面了