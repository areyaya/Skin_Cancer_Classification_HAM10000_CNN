This project focuses on the automated classification of skin lesions using the HAM10000 dataset, a publicly available collection of dermoscopic skin images 
and associated clinical metadata. The objective is to investigate the effectiveness of transfer learning for skin cancer diagnosis by extracting deep visual 
features from a pre-trained EfficientNet model and combining them with patient metadata. These features are then used to train and evaluate an XGBoost classifier 
for multi-class skin lesion classification. The project includes data preprocessing, feature engineering, image augmentation, dimensionality reduction, model training, 
and performance evaluation, providing a complete machine learning pipeline for skin lesion analysis.

-----------HOW TO RUN THE CLASSIFIER CODE?------------
There are two groups of multimodal csv files: one group for EffNet concatenated features and the other the CustomCNN concatenated features.
This example will use CustomCNN concatenated features to follow through steps.
step 1: Download the 3 train, test and val csv named customcnn_test_multimodal.csv, customcnn_train_multimodal, customcnn_val_multimodal and save them in a folder
step 2: Download the XGboost python file
step 3: Open the XGBoost Python file and scroll down to the execution block; you should see 4 variables "TRAIN_PATH", "TEST_PATH", "VAL_PATH", and "OUTPUT_DIR"
step 4: Replace those existing values with the directory path of the corresponding multimodal csv files you downloaded and saved in step 1
NOTE: "OUTPUT_DIR" the path of this variable is where your output will be stored (the AUC, Confusion Matric, etc graphs) so select the directory carefully
step 5: Run the code

It should run :D <3
