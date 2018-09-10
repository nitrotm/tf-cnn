###########################################
# eval.py
#
# Model evaluation using tensorflow.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import json, time

from pathlib import Path

import tensorflow as tf

from utils import create_logger, reset_tensorflow, f1_score

from ds.reader import make_tfr_dataset, make_tfr_input_fn

from cnn.model import resolve_topology, topology_cnn_model

logger = create_logger(__name__)


def run(device, model, sources, threads, prefetch, batch, steps, verbose, seed):
    """
    Evaluation task implementation.
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
    classes = params.get('classes', [0.0, 1.0])

    # read previous evals
    evalsfile = Path(model) / "evals.json"
    if evalsfile.exists():
        evalslogs = json.loads(evalsfile.read_text())
    else:
        evalslogs = list()

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

    # evaluate datasets using model
    for source in sources:
        eval_start_time = time.time()

        logger.info("Starting evaluation: %s (steps=%d, batch=%d)", source, steps, batch)

        results = estimator.evaluate(
            input_fn=make_tfr_input_fn(
                make_tfr_dataset(source, threads=threads),
                threads=threads,
                limit=max(0, steps*batch),
                prefetch=prefetch,
                batch=batch
            )
        )

        for i in range(len(classes)):
            logger.info(
                "Class %d: precision=%02.02f%%, recall=%02.02f%%",
                i,
                results['precision_c%d' % i] * 100,
                results['recall_c%d' % i] * 100
            )
        logger.info("Evaluation completed: loss=%f, jaccard=%f (%s)", results['loss'], results['jaccard'], source)

        # write eval logs
        evalslogs.append({
            'dataset': source,
            'checkpoint': Path(estimator.latest_checkpoint()).name,
            'duration': int(time.time() - eval_start_time),
            'loss': float(results['loss']),
            'jaccard': float(results['jaccard']),
            'precision': [ float(results['precision_c%d' % i]) for i in range(len(classes)) ],
            'recall': [ float(results['recall_c%d' % i]) for i in range(len(classes)) ],
            'f1': [ f1_score(results['precision_c%d' % i], results['recall_c%d' % i]) for i in range(len(classes)) ],
        })
        evalsfile.write_text(json.dumps(evalslogs, indent=' '))
