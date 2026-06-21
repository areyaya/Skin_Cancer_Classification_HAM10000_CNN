#Supervised Feature Reduction Neural Network
#Tuning (Keras Tuner / Bayesian Optimization)  —  CSCI323 Project
#======================================================================
#Used TensorFlow / Keras
#Install if not available: python -m pip install tensorflow keras-tuner


#SECTION 1: IMPORTS
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt 
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


#SECTION 2: FILE PATHS 
#For EfficientNet extracted features
TRAIN_FEATURES_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\train_features.npy" 
TEST_FEATURES_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\test_features.npy" 
VAL_FEATURES_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\val_features.npy" 

TRAIN_META_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\train_meta.csv" 
TEST_META_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\test_meta.csv" 
VAL_META_PATH_1 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\EffNet Extracted\val_meta.csv"

#For CNN extracted features
TRAIN_FEATURES_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\train_features.npy"
TEST_FEATURES_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\test_features.npy"
VAL_FEATURES_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\val_features.npy"

TRAIN_LABELS_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\train_labels.npy"
TEST_LABELS_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\test_labels.npy"
VAL_LABELS_PATH_2 = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Extracted\val_labels.npy"

OUTPUT_DIR = r"C:\Users\shrin\OneDrive - University of Wollongong\CSCI 323\Project\TradCNN Outputs"     


#SECTION 3: GLOBAL SETTINGS
#Flip these to False, If to test only one pipeline at a time.
RUN_EFFNET_PIPELINE = True
RUN_CNN_PIPELINE    = True

NUM_CLASSES   = 7     
BATCH_SIZE    = 128   #not tuned to maanage search time
RANDOM_SEED   = 42

#Keras Tuner search
MAX_TRIALS           = 20    
EXECUTIONS_PER_TRIAL  = 1   
SEARCH_EPOCHS         = 20  
SEARCH_PATIENCE       = 5  

#Finnal taining
FINAL_EPOCHS    = 70  
FINAL_PATIENCE  = 6    

TUNER_OVERWRITE = False 
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


#SECTION 4: DATA LOADING
#For pipeline 1
def load_effnet_pipeline_data():
    print("\n \n EFFNET PIPELINE - 1 - Data Loading")
    print("=" * 45)

    train_X = np.load(TRAIN_FEATURES_PATH_1)
    val_X   = np.load(VAL_FEATURES_PATH_1)
    test_X  = np.load(TEST_FEATURES_PATH_1)

    train_meta = pd.read_csv(TRAIN_META_PATH_1)
    val_meta   = pd.read_csv(VAL_META_PATH_1)
    test_meta  = pd.read_csv(TEST_META_PATH_1)

#Extracting necessary columns only
    train_y = train_meta["dx"].values.astype(int)
    val_y   = val_meta["dx"].values.astype(int)
    test_y  = test_meta["dx"].values.astype(int)

    train_ids = train_meta["image_id"].values
    val_ids   = val_meta["image_id"].values
    test_ids  = test_meta["image_id"].values

    print(f"TRAIN: features {train_X.shape} | labels {train_y.shape}")
    print(f"VAL: features {val_X.shape}     | labels {val_y.shape}")
    print(f"TEST: features {test_X.shape}   | labels {test_y.shape}")
    return train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids

#For pipeline 2
def load_cnn_pipeline_data():
    print("\n \n SCRATCH CNN PIPELINE - 2 - Data Loading")
    print("=" * 45)

    train_X = np.load(TRAIN_FEATURES_PATH_2)
    val_X   = np.load(VAL_FEATURES_PATH_2)
    test_X  = np.load(TEST_FEATURES_PATH_2)

    train_y = np.load(TRAIN_LABELS_PATH_2).astype(int)
    val_y   = np.load(VAL_LABELS_PATH_2).astype(int)
    test_y  = np.load(TEST_LABELS_PATH_2).astype(int)

    #Placeholder IDs
    train_ids = np.arange(len(train_y))
    val_ids   = np.arange(len(val_y))
    test_ids  = np.arange(len(test_y))

    print(f"TRAIN: features {train_X.shape} | labels {train_y.shape}")
    print(f"VAL: features {val_X.shape}     | labels {val_y.shape}")
    print(f"TEST: features {test_X.shape}   | labels {test_y.shape}")
    return train_X, train_y, train_ids, val_X, val_y, val_ids, test_X, test_y, test_ids


#SECTION 5: MODEL: (SupervisedEncoder class)
def make_build_model_fn(input_dim: int, num_classes: int):
    def build_model(hp: kt.HyperParameters) -> keras.Model:

        #Hyperparameter Searching
        hidden1_units = hp.Int("hidden1_units", min_value=256, max_value=768)
        hidden2_units = hp.Int("hidden2_units", min_value=128, max_value=384)

        # Bottleneck size — searched and conttrained to 128.
        bottleneck_dim = hp.Choice("bottleneck_dim", values=[128], default=128)
        dropout_rate  = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.05, default=0.3)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)

        #Model Architecture
        inputs = keras.Input(shape=(input_dim,), name="input_features")

        #Layer 1
        x = layers.Dense(hidden1_units, name="encoder_dense_1")(inputs)
        x = layers.BatchNormalization(name="encoder_bn_1")(x)
        x = layers.ReLU(name="encoder_relu_1")(x)
        x = layers.Dropout(dropout_rate, name="encoder_dropout_1")(x)

        #Layer 2
        x = layers.Dense(hidden2_units, name="encoder_dense_2")(x)
        x = layers.BatchNormalization(name="encoder_bn_2")(x)
        x = layers.ReLU(name="encoder_relu_2")(x)
        x = layers.Dropout(dropout_rate, name="encoder_dropout_2")(x)

        x = layers.Dense(bottleneck_dim, name="bottleneck_dense")(x)
        x = layers.BatchNormalization(name="bottleneck_bn")(x)

        bottleneck_output = layers.ReLU(name="bottleneck_layer")(x)

        classifier_output = layers.Dense(num_classes, activation="softmax", name="classifier_head")(bottleneck_output)

        model = keras.Model(inputs=inputs, outputs=classifier_output, name="supervised_encoder")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",  #int labels, not one-hot
            metrics=["accuracy"],
        )
        return model
    return build_model


#SECTION 6: Scaling, Class Weights, Reduced Features Extraction, Output Saved
def run_feature_reduction_pipeline(
    pipeline_name: str,
    train_X_raw, train_y, train_ids,
    val_X_raw,   val_y,   val_ids,
    test_X_raw,  test_y,  test_ids,
):
    print("\n" + "=" * 50)
    print(f"  FEATURE REDUCTION: {pipeline_name.upper()}")
    print("=" * 50)

    #6 a. Checking the shpae
    assert len(train_X_raw) == len(train_y) == len(train_ids), \
        f"[{pipeline_name}] Train size mismatch: features={len(train_X_raw)}, labels={len(train_y)}, ids={len(train_ids)}"
    assert len(val_X_raw) == len(val_y) == len(val_ids), \
        f"[{pipeline_name}] Val size mismatch: features={len(val_X_raw)}, labels={len(val_y)}, ids={len(val_ids)}"
    assert len(test_X_raw) == len(test_y) == len(test_ids), \
        f"[{pipeline_name}] Test size mismatch: features={len(test_X_raw)}, labels={len(test_y)}, ids={len(test_ids)}"
    print(f"[{pipeline_name}] Shape assertions passed, there were no mismatches found")

    #6 b. Feature scaling — fit on train only
    print(f"\n - StandardScaler on Training Data Only")
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X_raw).astype(np.float32)
    val_X   = scaler.transform(val_X_raw).astype(np.float32)
    test_X  = scaler.transform(test_X_raw).astype(np.float32)
    print(f"{pipeline_name} Scaling is complete.")

    #6 c. Blancing imbalance classes using class weights
    print(f"\n - Computing Class Weights in Dictionary")
    weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_y,
    )

    #Plain dict class weights for Keras Tuner
    class_weight_dict = {i: float(w) for i, w in enumerate(weights_np)}
    print(f"[{pipeline_name}] - {class_weight_dict}")

    input_dim = train_X.shape[1]
    print(f"\n - Detected input dimensionality: {input_dim} \n")

    #6 d. Set up the Bayesian Optimization tuner 
    build_fn = make_build_model_fn(input_dim=input_dim, num_classes=NUM_CLASSES)

    tuner_dir = os.path.join(OUTPUT_DIR, "keras_tuner_logs")
    tuner = kt.BayesianOptimization(
        build_fn,
        objective = "val_loss",         
        max_trials = MAX_TRIALS,
        executions_per_trial = EXECUTIONS_PER_TRIAL,
        directory = tuner_dir,
        project_name = f"reduction_{pipeline_name}",
        overwrite = TUNER_OVERWRITE,
        seed = RANDOM_SEED,
    )

    print(f"\n[{pipeline_name}] - Search space details: \n")
    tuner.search_space_summary()

    search_early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=SEARCH_PATIENCE, restore_best_weights=True)

    tuner.search(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=SEARCH_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[search_early_stop],
        verbose=1,
    )

    #6 e. Finalizing the best parameters after optimization search results
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    bottleneck_dim = best_hp.get("bottleneck_dim")

    print(f"\n[{pipeline_name}] Best Parameters Found ")
    for key, value in best_hp.values.items():
        print(f"    {key}: {value}")

    # ---- 7f. Retrain the BEST config fully (real epoch budget) -------------
    print(f"\n[{pipeline_name}] Retraining best configuration up to {FINAL_EPOCHS} epochs, patience={FINAL_PATIENCE}")

    final_model = tuner.hypermodel.build(best_hp)

    final_early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=FINAL_PATIENCE, restore_best_weights=True)
    # halves the learning rate if val_loss stalls before early stopping
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    history = final_model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[final_early_stop, reduce_lr]
    )

    best_val_loss = float(min(history.history["val_loss"]))
    print(f"\n[{pipeline_name}] Final model best val_loss: {best_val_loss:.4f}")

    ''' Added Checkpoint to save the trained model before extraction to prevent loss of data and time-effciency; can reload with:
    reloaded = keras.models.load_model(checkpoint_path)
    and re-run just the extraction/save steps without retraining anything. '''

    checkpoint_path = os.path.join(OUTPUT_DIR, f"best_model_{pipeline_name}.keras")
    final_model.save(checkpoint_path)
    print(f"[{pipeline_name}] Checkpoint saved :)")

    #6 g. Reduced Features Extraction
    # Builds a NEW model that shares the trained weights but stops at the bottleneck layer. The classifier head is bypassed entirely
    encoder_model = keras.Model(
        inputs=final_model.input,
        outputs=final_model.get_layer("bottleneck_layer").output,
        name=f"encoder_{pipeline_name}",
    )

    print(f"\n[{pipeline_name}] Extracting reduced features for all 3 splits")
    train_reduced = encoder_model.predict(train_X, batch_size=BATCH_SIZE)
    val_reduced   = encoder_model.predict(val_X,   batch_size=BATCH_SIZE)
    test_reduced  = encoder_model.predict(test_X,  batch_size=BATCH_SIZE)

    print(f"[{pipeline_name}]  Train: {train_reduced.shape} | Val: {val_reduced.shape} | Test: {test_reduced.shape}")

    #6 h. Saving the reduced features extracted files
    feature_col_names = [f"feat_{i}" for i in range(bottleneck_dim)]

    def save_reduced_csv(reduced, ids, labels, split_name):
        df = pd.DataFrame(reduced, columns=feature_col_names)
        df.insert(0, "dx", labels)
        df.insert(0, "image_id", ids)

        #naming pieline into the file name for easier access
        out_name = f"{pipeline_name}_reduced_{split_name}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df.to_csv(out_path, index=False)
        print("All files saved.")

    save_reduced_csv(train_reduced, train_ids, train_y, "train")
    save_reduced_csv(val_reduced,   val_ids,   val_y,   "val")
    save_reduced_csv(test_reduced,  test_ids,  test_y,  "test")

#SECTION 7: SUMMARY OF EACH PIPELINE
    total_imgs        = len(train_ids) + len(val_ids) + len(test_ids)
    total_input_cells = total_imgs * input_dim
    total_output_cells = total_imgs * bottleneck_dim

    print("\n" + "-" * 50)
    print(f"  {pipeline_name.upper()} SUMMARY")
    print("-" * 50)
    print(f"  Input features/image  : {input_dim}")
    print(f"  Output features/image : {bottleneck_dim}")
    print(f"  Compression ratio     : {input_dim / bottleneck_dim:.1f} x")
    print(f"  Total Input feature cells   : {total_input_cells:,}")
    print(f"  Total Output feature cells  :{total_output_cells:,}")
    print(f"  Reduction achieved    : {(1 - total_output_cells/total_input_cells)*100:.1f} %")
    print(f"  Best validation loss  : {best_val_loss:.4f}")
    print("-" * 50)

    return {
        "pipeline": pipeline_name,
        "input_dim": input_dim,
        "bottleneck_dim": bottleneck_dim,
        "best_val_loss": best_val_loss,
        "best_hyperparameters": best_hp.values,
    }


#SECTION 8: MAIN — run both pipelines
all_results = []

if RUN_EFFNET_PIPELINE:
    effnet_data = load_effnet_pipeline_data()
    effnet_result = run_feature_reduction_pipeline("effnet", *effnet_data)
    all_results.append(effnet_result)

if RUN_CNN_PIPELINE:
    cnn_data = load_cnn_pipeline_data()
    cnn_result = run_feature_reduction_pipeline("cnn", *cnn_data)
    all_results.append(cnn_result)



#SECTION 9: FINAL SUMMARY
print("\n" + "-" * 50)
print(" COMPLETE FEATURE REDUCTION")
print("-" * 50)
for res in all_results:
    print(f"\n  [{res['pipeline'].upper()}]")
    print(f"    Input dim       : {res['input_dim']}")
    print(f"    Bottleneck dim  : {res['bottleneck_dim']}")
    print(f"    Best val_loss   : {res['best_val_loss']:.4f}")
    print(f"    Best HPs        : {res['best_hyperparameters']}")
print("-" * 50)
