[中文文档](https://github.com/Lapis0x0/ResNetClassifier/blob/main/README_zh_CN.md/) | [English Document](https://github.com/Lapis0x0/ResNetClassifier/blob/main/README.md) 

Here's the translated document:

# ResNetClassifier
A demo model based on ResNet for classifying image collections.

# 1. Important Information
When you run `main.py` for the first time, the program will automatically check if there is a `data` folder in the local directory. If it does not exist, the program will create the `data` folder and four subfolders within it: `classification_results`, `images_to_classify`, `dataset`, and `models`. After executing `main.py`, the program will classify images and then automatically deduplicate the classification results by deleting any duplicate images.

* Classification Results: The results of classifying the "images to be classified" after executing the classification program.
* Images to Classify: The images that the user wants to classify.
* Dataset: The training dataset.
* Models: Pre-trained models and models trained by the user.

# 2. Training the Model
If you want to train your own image classification model, create the corresponding folders under the `data/dataset` directory. The folder names will serve as your classification labels. Then, place your training data (images) into these folders. 

For example, if there are five folders in the `dataset` directory named `a. Anime Girls`, `b. Landscapes`, `c. Animals`, `d. Food`, and `e. Games`, your training images should be placed in their respective folders.

After setting up your training data in the `data/dataset` directory, run `main.py`. The program will automatically start training the model based on the pre-trained model.

Due to the storage limits of GitHub repositories, the pre-trained ResNet model file is too large to upload. I have placed a copy in my own cloud storage: [https://data.lapis.cafe/api/raw?path=/03.%E6%96%87%E4%BB%B6%E5%88%86%E4%BA%AB/resnet18-f37072fd.pth](https://data.lapis.cafe/api/raw?path=/03.%E6%96%87%E4%BB%B6%E5%88%86%E4%BA%AB/resnet18-f37072fd.pth).

Of course, you can also download the model file yourself. By default, the pre-trained model should be placed at `./data/models/resnet18-f37072fd.pth`.

Once model training is complete, the trained model will be automatically saved to the `./data/models/` folder. The file will be named `model.pth`, and a plot of the loss values will be generated and saved in the root directory as `training_loss.png`.

# 3. Starting the Classification
After your custom model has been trained, you can run `main.py`. The program will automatically start the classification process using the model you've trained. Once classification is complete, a deduplication operation will be performed to delete duplicate images. You can view the classification results in the `./data/classification_results/` folder.