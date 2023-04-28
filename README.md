# NeuroVision

### Deep Learning Multimodal Prediction of Higher-order Cognition

_Woody Hulse, Vandana Ramesh, Micah Lessnick_

## Introduction

In recent years, there has been rapid research into applications of deep neural networks to brain biometric data. Most work has been done on the classification of neurological disorder by evaluating abnormalities in either MRI or EEG data, but there has been little focus, in the conventional sense or with novel deep learning methods, on the prediction of higher-order behavioral cognition. Our goal is to predict over an array of behavioral metrics (CVLT, LPS, RWT, TAP, TMT, WST) using a supervised 3D Deep Convolutional Neural Network (DCNN) paired with an EEGNet-based 2D DCNN to interpret both MRI and EEG data. We hypothesize that a deep learning model will be able to detect and interpret structures and activities in the brain that indicate certain behavioral characteristics.

## Related Work

General overview of machine learning methods for neurological disorder classification: https://www.nature.com/articles/s41380-020-0825-2

Related paper: V. Guleva, A. Calcagno, P. Reali and A. M. Bianchi, "Personality traits classification from EEG signals using EEGNet," 2022 IEEE 21st Mediterranean Electrotechnical Conference (MELECON), Palermo, Italy, 2022, pp. 590-594, doi: 10.1109/MELECON53508.2022.9843118.

This paper implements a cutting-edge CNN EEG decoder, EEGNet, for the classification of "Big Five" (DSM-V) Personality characteristics. They were able to classify these traits in a binary way at near 90% accuracy with EEG analysis alone. However, this paper only had 38 subjects and reduced a continuous DSM-V scale to a binary.

## Data

Link to dataset: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/

Retrieved from the Max Planck Institute, Leipzig, Germany. Paired EEG/MRI with an array of behavioral metrics (CVLT, LPS, RWT, TAP, TMT, WST). We will evaluate on all of these behavioral metrics. This dataset in full is approximately ~450GB in size (mostly of time-series MRI data), but we only require about a 10GB subset to analyze structures and asynchronous activity. Reducing this data into relevant modes, conversion into appropriate data formats (reconfiguring .set, .nii, .json), and normalizing data took significant preprocessing.

## Methodology

The general structure of our model is a 3D CNN model for MRI input and a LSTM or EEGNet for EEG input fused in latent space to generate a final prediction. Our CNN will be an adaptation of cutting-edge architecture for image data, VGG-19, for 3D MRI data. We hope by using this similar architecture or a transfer learning approach we can achieve similar success with image comprehension in 3D space as VGG can with 2D. We will start with a CNN like EEGNet for EEG data, but we may switch to an RNN if it proves more accurate (although theoretically harder to combine in latent space). EEG has been shown in several cases (see Related Work) to perform well in these tasks. We will train our data on some subset of the data (150/50 k-fold cross-validated train-test split) so as to compensate for potential variability introduced by a smaller testing group. 

## Metrics

We plan to train 3 models (EEG-only model, MRI-only model, and combined EEG-MRI model) and compare L2 loss of each behavioral metric and model predictions with that of several control models. Our goal is to achieve certain levels of statistical significance--i.e., a "base goal" would be results that are within p < 0.20, a "target" of results which are p < 0.05, and a "stretch" of results which are p < 0.01. We are predicting continuous metrics, so measuring accuracy via a discretization of these tasks outside of those classifications which are well-defined in psychology is arbitrary for this task.

## Ethics

+ We think it's relevant to discuss stakeholders of a potential threat given the successful result to this study. While exciting that we can define a function of the brain and its activity to the way that it operates in the world and society, in the wrong hands these models can be destructive and a major invasion of privacy. The primary bad actor here could be advertisers, who, if armed with behavioral data that is drawn from the most fundamental structures in the brain, could launch aggressive ad campaigns which target individuals' fundamental psychology if utilized accurately. There already is a trough of user activity data through cookies, mouse tracking, keystroke tracking, etc. that make significant intrusion into the lives of potential consumers (and everyone), but this added tool would cross major human privacy lines.

+ It's also important to evaluate the role of Deep Learning here. There has been some work, with very little success, to do this kind of evaluation of neurological structures by hand. Neuroscientists haven't yet figured out why and how we can interpret the brain in a way that can determine function beyond a baseline order/disorder binary--identification of characteristics in completely functioning adults has been elusive. Deep Learning, on the other hand, can seek out these otherwise extremely subtle distinctions between sets of data, and has a superhuman ability to find trends in data. If there are some structures that do indicate trends in behavior, our model will likely find it, after which we can dissect layer outputs to determine which structures these are.
