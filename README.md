# 3DPose总结
## 参考weakly [代码](https://github.com/xingyizhou/pytorch-pose-hg-3d)
## 算法结构图
<div align="center"><img src="/tools/data/gan1.png"></div>

- 整体思路是在weakly的基础上实现对抗学习，生成器G就是取消geometric loss的weakl结构,这里设计了一个全新的多源判别器D.

 
## Multi-source Discriminator

<div align="center"><img src="/tools/data/gan.png"></div>

#### 这个判别器有三个source
-  第一：图片经过卷积到256x1的向量
-  第二：Geometric Descriptor是把16个坐标点（x,y,z）两两之间做


    $$x+y$$


## Requirements
- cudnn
- [PyTorch < 0.4](http://pytorch.org/)
- Python with h5py, opencv and [progress](https://anaconda.org/conda-forge/progress)
- Optional: [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) 


 


## 数据集部分
- 数据集准备
  - HM36预处理数据集下载 [here](https://drive.google.com/open?id=0BxjtxDYaOrYPRlJJeDhfUVAzM00).
  - 运行 `python GetH36M.py` in `src/tools/` 把H36M标注数据转化为.h5文件
  - 在 `src/ref.py` 中修改数据集路径. 

  - 同理在`src/ref.py`中添加mpii图片路径
  

## 数据集预处理
- Human3.6M数据集
  - 作者预处理的h36m数据集是224x224，带有2d GT,3d GT(相机坐标)
- MPII数据集
  - mpii数据集尺寸不定，有2D GT
- 最终将两部分数据都变换处理得到一个四元组数据（这部分的实现在h36m.py mpii.py）：
  - 256x256的image，
  - 2d 热图（由2D GT得到）
  - 3D depth（3D GT映射到（-1,1），2D 数据的depth初始化为1)
  - 原始GT


#### 数据集预处理图示
