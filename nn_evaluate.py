#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from docopt import docopt
from nn import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_ae_encoder, load_fold)
from sklearn.metrics import confusion_matrix

def nn_results(hdf5, experiment, code_size_1, code_size_2):
    str_experiment = str(experiment)
    #str_experiment = str(experiment)[2:-2]
    exp_storage = hdf5["experiments"][str_experiment]
    #exp_storage = hdf5["experiments"]["cc200_whole"]
    n_classes = 2

    results = []

    for fold in exp_storage:

        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })

        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)

        y_test = np.array([to_softmax(n_classes, y) for y in y_test])

        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.ckpt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.ckpt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })

        try:

            model = nn(X_test.shape[1], n_classes, [
                {"size": 1000, "actv": tf.nn.tanh},
                {"size": 600, "actv": tf.nn.tanh},
            ])

            init = tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session() as sess:

                sess.run(init)

                saver = tf.compat.v1.train.Saver(model["params"])
                saver.restore(sess, nn_model_path)

                output = sess.run(
                    model["output"],
                    feed_dict={
                        model["input"]: X_test,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                    }
                )

                print(output)

                y_pred = np.argmax(output, axis=1)
                y_true = np.argmax(y_test, axis=1)

                [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
                accuracy = (TP+TN)/(TP+TN+FP+FN)
                specificity = TN/(FP+TN)
                precision = TP/(TP+FP)
                sensivity = recall = TP/(TP+FN)
                fscore = 2*TP/(2*TP+FP+FN)

                results.append([accuracy, precision, recall, fscore, sensivity, specificity])
        finally:
            reset()

    return [experiment] + np.mean(results, axis=0).tolist()

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pd.set_option("display.expand_frame_repr", False)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    if arguments["--whole"]:
        hdf5_name = str("./data/abide_whole.hdf5")
    if arguments["--male"]:
        hdf5_name = str("./data/abide_male.hdf5")
    if arguments["--threshold"]:
        hdf5_name = str("./data/abide_threshold.hdf5")
    if arguments["--leave-site-out"]:
        hdf5_name = str("./data/abide_leave-site-out.hdf5")

    hdf5 = hdf5_handler(bytes(hdf5_name,encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []

    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)]

        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                site_config = {"site": site}
                experiments += [
                    format_config("{derivative}_leavesiteout-{site}",
                                  config, site_config)
                ]

    # First autoencoder bottleneck
    code_size_1 = 1000

    # Second autoencoder bottleneck
    code_size_2 = 600

    results = []

    experiments = sorted(experiments)
    for experiment in experiments:
        results.append(nn_results(hdf5, experiment, code_size_1, code_size_2))

    print(results)
    cols = ["Exp", "Accuracy", "Precision", "Recall", "F-score",
            "Sensivity", "Specificity"]
    df = pd.DataFrame(results, columns=cols)

    print(df[cols] \
        .sort_values(["Exp"]) \
        .reset_index()
    )

    results_name = hdf5_name[:-5] + "_results.txt"
    df.to_csv(results_name, header=True, index=True, sep='\t', mode='w')