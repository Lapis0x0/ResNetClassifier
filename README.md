# ResNetClassifier
自己拿来分类图库的一个基于ResNet训练的模型demo

# 须知
在首次启动main.py时，会自动检测本地是否存在data文件夹，若不存在，则会自动创建data文件夹，并在其目录下创建分类结果、需要分类的图片、dataset和models四个子文件夹。
执行main.py后，会进行图片分类，分类结束后会自动对分类结果进行去重，删除重复图片。

* 分类结果：执行分类程序后对“需要分类的图片”的分类结果。
* 需要分类的图片：用户需要分类的图片。
* dataset：训练数据集。
* models：预训练模型，和用户训练的模型


由于GitHub repo的最大储存限制，对应的预训练ResNet模型文件太大无法上传，我放了一份在我自己的网盘里：https://data.lapis.cafe/api/raw?path=/03.%E6%96%87%E4%BB%B6%E5%88%86%E4%BA%AB/resnet18-f37072fd.pth

当然，你也可以自行下载模型文件。我默认会将预训练模型放在`./data/models/resnet18-f37072fd.pth`，
