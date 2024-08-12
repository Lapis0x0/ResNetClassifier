# main.py

from PyQt5 import QtWidgets
from src.train_model import train_model
from src.classify_model import run_classification
from src.remove_duplicates import remove_duplicates
import os
import sys

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Classification Tool')

        # 创建按钮
        self.trainButton = QtWidgets.QPushButton('Train Model', self)
        self.classifyButton = QtWidgets.QPushButton('Classify Images', self)
        self.clearButton = QtWidgets.QPushButton('Clear Images', self)

        # 连接按钮事件
        self.trainButton.clicked.connect(self.train_model)
        self.classifyButton.clicked.connect(self.classify_images)
        self.clearButton.clicked.connect(self.clear_images)

        # 布局
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.trainButton)
        layout.addWidget(self.classifyButton)
        layout.addWidget(self.clearButton)
        self.setLayout(layout)

    def train_model(self):
        model_save_path = 'data/models/model.pth'
        data_path = 'data/dataset'
        if os.path.exists(model_save_path):
            retrain = QtWidgets.QMessageBox.question(self, 'Retrain Model', 
                                                     "Model file found. Do you want to retrain the model?",
                                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if retrain == QtWidgets.QMessageBox.Yes:
                num_epochs = 20
                train_model(data_path, model_save_path, num_epochs)
                QtWidgets.QMessageBox.information(self, 'Info', "Model retraining completed.")
        else:
            num_epochs = 20
            train_model(data_path, model_save_path, num_epochs)
            QtWidgets.QMessageBox.information(self, 'Info', "Model training completed.")

    def classify_images(self):
        test_folder = 'data/需要分类的图片'
        output_folder = 'data/分类结果'
        run_classification('data/dataset', 'data/models/model.pth', test_folder, output_folder)
        QtWidgets.QMessageBox.information(self, 'Info', "Image classification completed.")
        remove_duplicates(output_folder)
        QtWidgets.QMessageBox.information(self, 'Info', "Duplicate removal completed.")

    def clear_images(self):
        test_folder = 'data/需要分类的图片'
        clear = QtWidgets.QMessageBox.question(self, 'Clear Images', 
                                               "Do you want to clear 'data/需要分类的图片'?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if clear == QtWidgets.QMessageBox.Yes:
            for filename in os.listdir(test_folder):
                file_path = os.path.join(test_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(self, 'Warning', f'Failed to delete {file_path}. Reason: {e}')
            QtWidgets.QMessageBox.information(self, 'Info', "'data/需要分类的图片' has been cleared.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()