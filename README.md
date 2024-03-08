# Research-on-Sleep-Staging-Detection-Based-on-Deep-Neural-Network
基于深度神经网络的睡眠分期检测方法研究

1、在传统的睡眠分期模型的基础上， 提出一种基于多曲线融合算法的睡眠分期检测模型（Multi - curve fusion EEGNET， MCF-EEGNET）。 MCF-EEGNET 基于睡眠过程中的脑电信号数据， 采用多曲线融合算法和深度神经网络训练模型， 从而实现睡眠的自动分期。 MCF-EEGNET 利用了脑电信号中的节律波段， 给予对应的权重， 融合成一个全新的曲线。 实验结果表明， MCF-EEGNET 可以很好的与专家批注结果拟合。 并且，MCF-EEGNET 与传统模型相比在预测睡眠阶段的准确率上有所提高， 更有利于实现睡眠分期。

2、在传统的睡眠分期模型的基础上， 提出一种融入脑地形图分析的睡眠分期检测模型（Brain map- EEGNET， BM-EEGNET）。 BM-EEGNET 通过脑地形图得到不同区域的热力图， 通过聚类算法将热力中心进行聚类， 使高电位区域更加明显， 再将其嵌入包含注意力机制的 CNN 之中， 从而得到更好的预测效果。 实验结果表明， BM-EEGNET 在预测结果和 F1 分数上优于传统模型， 在实现睡眠分期方面表现突出。  

3、对比研究 BM-EEGNET 与 MCF-EEGNET 共性与差异， 分析了两者的优缺点， 实验表明 BM-EEGNET 在 N1 阶段的准确率高于传统模型， 而 MCF-EEGNET 在其他阶段的准确率均高于传统模型， 因此 BM-EEGNET 与 MCF-EEGNET 分别在提取 N1 阶段和其他睡眠阶段的信息方面有着显著优势。  

## MCF-EEGNET

1、数据集

```txt
Sleep-Edfx
```

2、实验环境

```python
numpy>=1.16.1
pandas>=0.24.1
scikit-learn>=0.20.3
scipy>=1.2.1
tensorflow-gpu==1.13.1
matplotlib>=3.0.2
pyEDFlib==0.1.19
mne==0.18.2
wget
```

3、实验步骤

（1）准备数据集，如果已经有了就不需要进行这一步

`python download_sleepedf.py`

（2）数据预处理

`python prepare_sleepedf.py`

（3）训练模型，gpu0 表示启用gpu训练，1表示用cpu训练

`python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19`

（4）模型预测 

`python predict.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best`

**运行**：如果是pycharm运行，输入下列指令即可。

```python
python download_sleepedf.py
python prepare_sleepedf.py
python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19
python predict.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best
```

