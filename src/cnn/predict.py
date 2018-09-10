###########################################
# predict.py
#
# Image prediction using trained model in tensorflow
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import cv2, json, math

from pathlib import Path

import numpy as np

import tensorflow as tf

from utils import reset_tensorflow, create_logger, normalize, tofloats, tobytes, mix, resizemax, f1_score

from ds.reader import make_tfr_dataset, make_tfr_input_fn, make_predict_dataset, make_predict_input_fn

from cnn.model import resolve_topology, topology_cnn_model

logger = create_logger(__name__)


def annotate(img, size, top_label='', bottom_label='', fg=(1, 1, 1), bg=(0, 0, 0), font=cv2.FONT_HERSHEY_PLAIN, fsize=1, thickness=1):
    img = resizemax(img, size*2)
    if len(top_label) > 0:
        (w, h), base = cv2.getTextSize(top_label, font, fsize, thickness)
        x = 5
        y = 8 + h + base
        cv2.rectangle(img, (x-2, y-h-base-5), (x+w+2, y+2), bg, cv2.FILLED)
        cv2.putText(img, top_label, (x, y-base), font, fsize, fg, thickness)
    if len(bottom_label) > 0:
        (w, h), base = cv2.getTextSize(bottom_label, font, fsize, thickness)
        x = 5
        y = img.shape[0] - 5
        cv2.rectangle(img, (x-2, y-h-base-5), (x+w+2, y+2), bg, cv2.FILLED)
        cv2.putText(img, bottom_label, (x, y-base), font, fsize, fg, thickness)
    return img


def show_results(topology, results, size, verbose):
    for result in results:
        items = list()

        border = np.concatenate([np.zeros((2*size, 2, 3)), 0.5*np.ones((2*size, 2*2, 3)), np.zeros((2*size, 2, 3))], axis=1)

        name = 'input'
        if 'name' in result:
            name += ': ' + str(result['name'], 'utf-8')
        if 'variant' in result:
            name += '(' + str(result['variant'], 'utf-8') + ')'

        # s = set([
        #     'input: image_08046(0.75_none_0.0_0)',
        #     'input: image_07810(0.75_none_0.0_1)',
        #     'input: image_07119(0.75_none_0.0_3)',
        # ])
        # if name not in s:
        #     continue

        inputs = normalize(cv2.cvtColor(result['inputs'], cv2.COLOR_RGB2BGR))
        items.append(annotate(inputs, size, 'input', name))
        if 'labels' in result:
            labels = cv2.cvtColor(result['labels'], cv2.COLOR_GRAY2BGR)
            items.append(border)
            items.append(annotate(labels, size, 'label'))
        predictions = cv2.cvtColor(result['predictions'].astype(np.float32), cv2.COLOR_GRAY2BGR)
        items.append(border)
        items.append(annotate(predictions, size, 'prediction', topology))
        if 'mosaic' in result:
            r = result['mosaic'][:,:,0]
            g = result['mosaic'][:,:,1]
            b = result['mosaic'][:,:,2]
            tp = np.sum(np.bitwise_and(np.bitwise_and(r == 0,   g == 255), b == 0))
            fp = np.sum(np.bitwise_and(np.bitwise_and(r == 255, g == 0),   b == 0))
            tn = np.sum(np.bitwise_and(np.bitwise_and(r == 0,   g == 0),   b == 0))
            fn = np.sum(np.bitwise_and(np.bitwise_and(r == 255, g == 0),   b == 255))
            if tp + fp > 0:
                sp = tp / (tp + fp)
            else:
                sp = 1
            if tp + fn > 0:
                sr = tp / (tp + fn)
            else:
                sr = 1
            if sp + sr > 0:
                f1 = 2 * (sp * sr) / (sp + sr)
            else:
                f1 = 0
            items.append(border)
            items.append(annotate(cv2.cvtColor(tofloats(result['mosaic']), cv2.COLOR_RGB2BGR), size, 'iou (P:%.00f%%, R:%.00f%%, F1:%.02f)' % (100*sp, 100*sr, f1), topology))
        if 'labels' in result:
            items.append(border)
            items.append(annotate(mix(inputs, labels), size, 'target'))
        items.append(border)
        items.append(annotate(mix(inputs, predictions), size, 'result', topology))

        cv2.imshow('cv2: result', tobytes(np.concatenate(items, axis=1)))

        for key in result.keys():
            if key.startswith('mosaic_'):
                cv2.imshow('cv2: ' + key, tobytes(normalize(result[key])))

        key = cv2.waitKey()
        while key != ord('q') and key != ord('c'):
            key = cv2.waitKey()
        if key == ord('q'):
            return False
    return True


def run(device, model, sources, threads, prefetch, batch, epochs, verbose, seed):
    """
    Prediction task implementation.
    """

    logger.info("Running graph on %s", device)

    # reset tensorflow
    reset_tensorflow(seed)

    # read parameters
    paramsfile = Path(model) / "parameters.json"
    if paramsfile.exists():
        params = json.loads(paramsfile.read_text())
    else:
        raise Exception("Model parameters.json not found")

    # read topology file
    logger.info("Creating model topology V%d (%s)", params.get('topology_version', 1), params.get('topology'))
    topologyfile = Path(model) / "topology.yaml"
    if not topologyfile.exists():
        raise Exception("Model topology.yaml not found")
    topology = resolve_topology(params.get('topology_version', 1))(
        topologyfile,
        params.get("initializer", 'none'),
        params.get("regularizer", 'none'),
        params.get("constraint", 'none'),
        params.get("activation", 'none'),
        params.get("local_response_normalization", 0),
        params.get("batch_normalization", False),
        params.get("dropout_rate", 0.0)
    )

    # create tensorflow estimator
    estimator = tf.estimator.Estimator(
        model_dir=model,
        model_fn=topology_cnn_model(topology, verbose),
        params=params,
        config=tf.estimator.RunConfig(
            tf_random_seed=seed,
            device_fn=lambda op: device
        )
    )
    logger.debug("Using checkpoint: %s", estimator.latest_checkpoint())

    # predict datasets using model
    for source in sources:
        logger.info("Starting prediction: %s (batch=%d)", source, batch)

        if source.endswith('.tfr') or source.endswith('.tfr.zlib') or source.endswith('.tfr.gz'):
            results = estimator.predict(
                input_fn=make_tfr_input_fn(
                    make_tfr_dataset(source, threads=threads),
                    threads=threads,
                    prefetch=1,
                    batch=batch
                )
            )
        else:
            results = estimator.predict(
                input_fn=make_predict_input_fn(
                    make_predict_dataset(source, threads=threads),
                    topology.config['input_size'],
                    threads=threads,
                    prefetch=1,
                    batch=batch,
                    repeat=epochs-1
                )
            )
        if not show_results(params.get('topology'), results, topology.config['input_size'], verbose):
            logger.info("Aborted prediction: %s", source)
            break

        logger.info("Finished prediction: %s", source)
