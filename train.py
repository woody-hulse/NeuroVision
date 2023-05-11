import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import scipy
from scipy import stats
from matplotlib import pyplot as plt

import preprocessing
from preprocessing import get_behavioral_test, preprocess_behavioral_dict
import models



def rank_accuracy(model, test_data, control_model, control_test_data, test_labels, behavioral_columns, plot=True):
    """
    ranks behavioral categories by thier predictability by a model
    model               : model to compare to control
    test_data           : testing dataset
    control_model       :
    control_test_data   : testing dataset for control model
    test_labels         : behavioral labels for testing dataset
    behavioral_columns  : corresponding behavioral columns for testing dataset
    plot                : plot final graph
    """

    pred_labels = model.call(test_data)[0]
    control_pred_labels = control_model.call(control_test_data)

    squared_diff = tf.math.square(test_labels - pred_labels)
    squared_diff_control = tf.math.square(test_labels - control_pred_labels)

    mean_squared_diff = tf.math.reduce_mean(squared_diff, axis=0).numpy()
    mean_squared_diff_control = tf.math.reduce_mean(squared_diff_control, axis=0).numpy()

    predictability = mean_squared_diff - mean_squared_diff_control 

    p_values = np.zeros(squared_diff.shape[1])
    for i in range(squared_diff.shape[1]):
        p_values[i] = scipy.stats.ttest_ind(a=squared_diff[:, i], b=squared_diff_control[:, i], equal_var=False).pvalue
            
    sorted_predictability, sorted_mean_squared_diff, sorted_mean_squared_diff_control, sorted_behavioral_columns, sorted_p_values = zip(*sorted(zip(predictability, mean_squared_diff, mean_squared_diff_control, behavioral_columns, p_values)))

    data = np.array([sorted_predictability, sorted_mean_squared_diff, sorted_p_values]).T
    table_df = pd.DataFrame(data=data, index=sorted_behavioral_columns, columns=["performance against control in MSE*", "MSE", "p-value"])

    print("\nbehavioral categories ranked by predictability")
    print(table_df.to_string())
    print()

    plot_msd = list(sorted_mean_squared_diff)[:5] + list(sorted_mean_squared_diff)[-5:]
    plot_msd_c = list(sorted_mean_squared_diff_control)[:5] + list(sorted_mean_squared_diff_control)[-5:]
    plot_col = list(sorted_behavioral_columns)[:5] + list(sorted_behavioral_columns)[-5:]

    if plot:
      fig, ax = plt.subplots()
      y = np.arange(len(plot_col))
      width = 0.3
      multiplier = 0
      
      ax.barh(y - width/2, plot_msd, width, label=model.name)
      ax.barh(y + width / 2, plot_msd_c, width, label="control")
    
      ax.set_yticks(y, plot_col)
      ax.set_xlabel('mean squared error (mse)')
      ax.set_title("NeuroVision Performance vs Control")
      ax.legend(loc='upper right', ncols=2)

      plt.show()

    # false positive (p value)
    fp_p_values = [p for p, value in zip(p_values, predictability) if value < 0]
    fp_chi_sq = -2 * sum([np.log(p) for p in fp_p_values])
    fp_p = stats.chi2.cdf(fp_chi_sq, len(fp_p_values) * 2)

    # true negative
    tn_p_values = [p for p, value in zip(p_values, predictability) if value >= 0]
    tn_chi_sq = -2 * sum([np.log(p) for p in tn_p_values])
    tn_p = stats.chi2.cdf(tn_chi_sq, len(tn_p_values) * 2)

    p = 1 - fp_p / (fp_p + tn_p)
    print("joint p value of superiority over control :", p)



def print_results(models, test_data, test_labels, metrics, plot=True):
    """
    prints the results of each model after training
    models          : list of models to compare
    test_data       : corresponding list of model test data
    test_labels     : behavioral labels
    metrics         : metrics to assess labels from
    plot            : plot charts
    """

    table = []
    
    for model, data in zip(models, test_data):
        table.append([])
        for metric in metrics:
            table[-1].append(metric(test_labels, model.call(data)).numpy())
    
    table_df = pd.DataFrame(
        data=table, 
        index=[model.name for model in models], 
        columns=[str(type(metric)).split('\'')[1].split('.')[-1] for metric in metrics])
    
    print("\nmodels vs accuracy on behavioral data")
    print(table_df.to_string())
    print()

    if plot:
      losses = [metrics[0](test_labels, model.call(data)).numpy() for model, data in zip(models, test_data)]
      losses, models = zip(*sorted(zip(losses, models), reverse=True))
      print(losses)
      plt.bar([model.name for model in models], losses)
      plt.xlabel('model')
      plt.ylabel('mean squared error (mse)')
      plt.xticks(rotation=45, ha='right')
      plt.title("Performance of Models")
      plt.show()



def get_patientIDs(filepath, mri_dir, eeg_dir, sync=True):
    """
    retrieve shared list of patientIDs
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    mri_dir         : mri dir name
    """

    mri_patientIDs = set(os.listdir(filepath + mri_dir))
    eeg_patientIDs = set(os.listdir(filepath + eeg_dir))
    if ".DS_Store" in mri_patientIDs: mri_patientIDs.remove(".DS_Store")
    if ".DS_Store" in eeg_patientIDs: eeg_patientIDs.remove(".DS_Store")
    if sync: patientIDs = list(mri_patientIDs.intersection(eeg_patientIDs))
    else: patientIDs = list(mri_patientIDs.union(eeg_patientIDs))
    patientIDs = [patientID.replace(".npy", "") for patientID in patientIDs]
    patientIDs.sort()

    return patientIDs


def load_acs_mri_data(filepath, mri_dir, patientIDs):
    """
    load preprocessed acs mri data
    filepath        : path to data directory
    mri_dir         : mri dir name
    patientIDs      : patientID list
    """

    print("loading mri data from", filepath, "...")
    acs_mri_data = []
    for patientID in tqdm(patientIDs):
        patient_mri = np.load(filepath + mri_dir + patientID + ".npy")
        acs_mri_data.append(patient_mri)
    acs_mri_data = np.stack(acs_mri_data)

    return acs_mri_data


def load_eeg_data(filepath, eeg_dir, patientIDs):
    """
    load preprocessed eeg data
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    patientIDs      : patientID list
    """

    print("loading eeg data from", filepath, "...")
    eeg_data = []
    for patientID in tqdm(patientIDs):
        patient_eeg = np.load(filepath + eeg_dir + patientID + ".npy")
        eeg_data.append(patient_eeg)
    eeg_data = np.stack(eeg_data)

    return eeg_data


def load_behavioral_data(filepath, behavioral_dir, patientIDs):
    """
    load preprocessed behavioral data
    filepath        : path to data directory
    behavioral_dir  : behavioral dir name
    patientIDs      : patientID list
    """

    behavioral_path = filepath + behavioral_dir
    behavioral_tests = preprocessing.BEHAVIORAL_TESTS
    behavioral_columns = preprocessing.get_behavioral_column_names(behavioral_path, behavioral_tests)
    behavioral_tests = [preprocess_behavioral_dict(get_behavioral_test(behavioral_path, test)) for test in behavioral_tests]

    print("loading behavioral data from", filepath, "...")
    behavioral_data = []
    for patientID in tqdm(patientIDs):
        try:
            behavioral_data.append([])
            for test in behavioral_tests:
                behavioral_data[-1] += test[patientID]
            behavioral_data[-1] = np.array(behavioral_data[-1])
        except KeyError:
            print("subject removed (lack of behavioral data) :", patientID)
            continue
    behavioral_data = np.stack(behavioral_data)

    return behavioral_data, behavioral_columns


def load_data(filepath, mri_dir, eeg_dir, behavioral_dir, patientIDs):
    """
    load preprocessed data
    filepath        : path to data directory
    eeg_dir         : eeg dir name
    mri_dir         : mri dir name
    behavioral_dir  : behavioral dir name
    return          : mri_data, eeg_data, behavioral_data
    """

    print()
    acs_mri_data = load_acs_mri_data(filepath, mri_dir, patientIDs)
    eeg_data = load_eeg_data(filepath, eeg_dir, patientIDs)
    behavioral_data, behavioral_columns = load_behavioral_data(filepath, behavioral_dir, patientIDs)
    print()

    return  acs_mri_data, eeg_data, behavioral_data, behavioral_columns


def main():

    mri_dir = "data/mri/"
    eeg_dir = "data/eeg/"
    behavioral_dir = "data/behavioral/"
    
    patientIDs = get_patientIDs("", mri_dir, eeg_dir, sync=True)

    mri_data, eeg_data, behavioral_data, behavioral_columns = load_data(
        "", 
        mri_dir, # preprocessing.MRI_RESULT_DIR, 
        eeg_dir, # preprocessing.EEG_RESULT_DIR,
        behavioral_dir, # preprocessing.BEHAVIORAL_DIR,
        patientIDs)

    train_patientIDs, test_patientIDs = preprocessing.train_test_split(patientIDs)
    train_mri_data, test_mri_data = preprocessing.train_test_split(mri_data)
    train_eeg_data, test_eeg_data = preprocessing.train_test_split(eeg_data)
    train_behavioral_data, test_behavioral_data = preprocessing.train_test_split(behavioral_data)

    print("\ntraining control models ...\n")

    center_model = models.CenterModel(name="control_center", shape=test_behavioral_data.shape)
    mean_model = models.MeanModel(name="control_mean", train_labels=train_behavioral_data)
    median_model = models.MedianModel(name="control_median", train_labels=train_behavioral_data)
    simple_nn = models.SimpleNN(name="control_mlp", output_units=behavioral_data.shape[1])
    simple_nn.compile(optimizer=simple_nn.optimizer, loss=simple_nn.loss, metrics=[])
    simple_nn.fit(train_mri_data, train_behavioral_data, batch_size=4, epochs=50, validation_data=(test_mri_data, test_behavioral_data))

    print("\ntraining new models ...\n")

    # eeg model
    eegnet_model = models.EEGModel(output_units=behavioral_data.shape[1], name="eegnet")
    eegnet_model.compile(optimizer=eegnet_model.optimizer, loss=eegnet_model.loss, metrics=[])
    eegnet_model.build(train_eeg_data.shape)
    eegnet_model.summary()
    eegnet_model.fit(train_eeg_data, train_behavioral_data, batch_size=1, epochs=50, validation_data=(test_eeg_data, test_behavioral_data))

    # *unused* vgg-based acs mri model
    vgg_acs_model = models.VGGACSModel(input_shape=train_mri_data.shape[1:], output_units=behavioral_data.shape[1], name="vgg_mri")
    vgg_acs_model.compile(
      optimizer=vgg_acs_model.optimizer,
      loss=vgg_acs_model.loss,
      metrics=[],
    )
    vgg_acs_model.build(train_mri_data.shape)
    vgg_acs_model.summary()
    vgg_acs_model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=50, validation_data=(test_mri_data, test_behavioral_data))

    # inception-based acs mri model
    inception_acs_model = models.InceptionACSModel(input_shape=train_mri_data.shape[1:], output_units=behavioral_data.shape[1], name="inception_mri")
    inception_acs_model.compile(
      optimizer=inception_acs_model.optimizer,
      loss=inception_acs_model.loss,
      metrics=[],
    )
    inception_acs_model.build(train_mri_data.shape)
    inception_acs_model.summary()
    inception_acs_model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=50, validation_data=(test_mri_data, test_behavioral_data))

    # vgg-based neurovision model
    neurovision_model = models.NeuroVisionModel(mri_input_shape=train_mri_data.shape[1:], output_units=behavioral_data.shape[1], name="NeuroVision_vgg")
    neurovision_model.compile(optimizer=neurovision_model.optimizer, loss=neurovision_model.loss, metrics=[])
    neurovision_model([train_eeg_data[:2], train_mri_data[:2]])
    neurovision_model.summary()
    neurovision_model.fit([train_eeg_data, train_mri_data], train_behavioral_data, batch_size=1, epochs=50, validation_data=([test_eeg_data, test_mri_data], test_behavioral_data))

    # final inception-based neurovision model
    neurovision_model2 = models.NeuroVisionModel2(mri_input_shape=mri_data.shape[1:], eeg_input_shape=eeg_data.shape[1:], output_units=behavioral_data.shape[1], name="NeuroVision")
    neurovision_model2.compile(optimizer=neurovision_model2.optimizer, loss=neurovision_model2.loss, metrics=[])
    neurovision_model2([train_eeg_data[:2], train_mri_data[:2]])
    neurovision_model2.summary()
    neurovision_model2.fit([train_eeg_data, train_mri_data], train_behavioral_data, batch_size=1, epochs=50, validation_data=([test_eeg_data, test_mri_data], test_behavioral_data))

    """

    #   unused "dead" models

    vgg3d_model = VGG3DModel(output_units=behavioral_data.shape[1])
    vgg3d_model.compile(
		optimizer=vgg3d_model.optimizer,
		loss=vgg3d_model.loss,
		metrics=[],
	)
    vgg3d_model.build(mri_data.shape)
    vgg3d_model.summary()
    vgg3d_model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=4, validation_data=(test_mri_data, test_behavioral_data))

    vggsliced_model = VGGSlicedModel(output_units=behavioral_data.shape[1])
    vggsliced_model.compile(
		optimizer=vggsliced_model.optimizer,
		loss=vggsliced_model.loss,
		metrics=[],
	)
    vggsliced_model.build(mri_data.shape)
    vggsliced_model.summary()
    vggsliced_model.fit(train_mri_data, train_behavioral_data, batch_size=1, epochs=4, validation_data=(test_mri_data, test_behavioral_data))
    """

    print_results(
        [center_model, median_model, mean_model, simple_nn, vgg_acs_model, inception_acs_model, eegnet_model, neurovision_model, neurovision_model2], 
        [test_mri_data, test_mri_data, test_mri_data, test_mri_data, test_mri_data, test_mri_data, test_eeg_data, [test_eeg_data, test_mri_data], [test_eeg_data, test_mri_data]], 
        test_behavioral_data, [tf.keras.losses.MeanSquaredError()])

    
    rank_accuracy(neurovision_model2, [test_eeg_data, test_mri_data], mean_model, test_mri_data, test_behavioral_data, behavioral_columns)


if __name__ == "__main__":
    os.system("clear")
    main()
