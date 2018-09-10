###########################################
# train.py
#
# Model training using tensorflow.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import json, time

from pathlib import Path

import tensorflow as tf

from utils import reset_tensorflow, create_logger, f1_score

from ds.reader import make_tfr_dataset, make_tfr_input_fn

from cnn.model import resolve_topology, topology_cnn_model

logger = create_logger(__name__)


def run(device, model, topology, topology_version, eval_source, train_sources, threads, prefetch, batch, steps, epochs, ram_cache, initializer, regularizer, constraint, activation, batch_normalization, local_response_normalization, dropout_rate, label_weighting, loss, optimizer, learning_rate, learning_rate_decay, verbose, seed):
    """
    Training task implementation.
    """

    classes = [0.0, 1.0]
    delta_loss_threshold = 0.01
    max_stable_loss_counter = 5
    max_overfit_loss_counter = 2
    start_time = time.time()

    logger.info("Running graph on %s", device)

    # reset tensorflow
    reset_tensorflow(seed)

    # create model directory
    Path(model).mkdir(parents=True, exist_ok=True)

    # copy topology
    topologyfile = Path(model) / "topology.yaml"
    if not topologyfile.exists():
        logger.info("Topology copied from %s", topology)
        topologyfile.write_text(Path(topology).read_text())

    # read/update/store parameters
    paramsfile = Path(model) / "parameters.json"
    params = dict({
        "batch_size": batch,
        "steps": steps,
        "epochs": epochs,
        "seed": seed,
        "classes": classes,
        "topology": topology,
        "topology_version": topology_version,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "learning_rate_decay": learning_rate_decay,
        "loss": loss,
        "activation": activation,
        "local_response_normalization": local_response_normalization,
        "batch_normalization": batch_normalization,
        "dropout_rate": dropout_rate,
        "initializer": initializer,
        "regularizer": regularizer,
        "constraint": constraint,
        "label_weighting": label_weighting,
    })
    if paramsfile.exists():
        oldparams = json.loads(paramsfile.read_text())
        for (key, value) in params.items():
            if key not in oldparams:
                logger.info("Parameter added: %s=%s", key, str(value))
            elif oldparams[key] != value:
                logger.info("Parameter changed: %s=%s (initial=%s)", key, value, oldparams[key])
        for (key, value) in oldparams.items():
            if key not in params:
                logger.info("Parameter obsolete: %s=%s", key, value)
    else:
        for (key, value) in params.items():
            logger.debug("Parameter added: %s=%s", key, str(value))
    paramsfile.write_text(json.dumps(params, indent=' '))

    # read previous sessions
    epochsfile = Path(model) / "epochs.json"
    if epochsfile.exists():
        epochlogs = json.loads(epochsfile.read_text())
    else:
        epochlogs = list()
    epoch_offset = len(epochlogs)

    # read topology file
    logger.info("Creating model topology V%d (%s)", topology_version, topology)
    topology = resolve_topology(topology_version)(
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
            log_step_count_steps=10,
            save_summary_steps=10,
            save_checkpoints_secs=900,
            keep_checkpoint_max=epochs,
            tf_random_seed=seed,
            device_fn=lambda op: device
        )
    )
    logger.debug("Starting checkpoint: %s", estimator.latest_checkpoint())

    # prepare inputs
    train_datasets = [ make_tfr_dataset(train_source, threads=threads, cache=ram_cache) for train_source in train_sources ]
    eval_dataset = make_tfr_dataset(eval_source, threads=threads, cache=ram_cache)

    # evaluate model
    eval_start_time = time.time()
    logger.info("Initial evaluation (batch=%d)", batch)

    results = estimator.evaluate(
        input_fn=make_tfr_input_fn(
            eval_dataset,
            threads=threads,
            prefetch=prefetch,
            batch=batch
        ),
    )

    for i in range(len(classes)):
        logger.info(
            "Class %d: precision=%02.02f%%, recall=%02.02f%%",
            i,
            results['precision_c%d' % i] * 100,
            results['recall_c%d' % i] * 100
        )
    logger.info("Evaluation completed: loss=%f, jaccard=%f (%d[min])", results['loss'], results['jaccard'], (time.time() - eval_start_time) / 60)

    # run epochs
    train_time = time.time()
    best_loss = results['loss']
    best_epoch = 0
    best_checkpoint = estimator.latest_checkpoint()
    stable_loss_counter = 0
    overfit_loss_counter = 0
    for epoch in range(epochs):
        # abort training if eval getting worse
        if stable_loss_counter >= max_stable_loss_counter or overfit_loss_counter >= max_overfit_loss_counter:
            logger.info("Early-stop at epoch %d / %d (best=%d, checkpoint=%s)", epoch, epochs, best_epoch + 1, best_checkpoint)
            break

        # start epoch
        epoch_start_time = time.time()
        if epoch > 0:
            elapsed_time = (time.time() - train_time) / 60
            total_time = elapsed_time * (epochs / epoch)
            remaining_time = total_time - elapsed_time
            logger.info("Epoch started: %d / %d (batch=%d, remaining=%d[min], total=%d[min])", epoch + 1, epochs, batch, remaining_time, total_time)
        else:
            logger.info("Epoch started: %d / %d (batch=%d)", epoch + 1, epochs, batch)

        # train model
        for train_dataset in train_datasets:
            train_start_time = time.time()
            logger.info("Training epoch started (%d steps)", steps)

            estimator.train(
                input_fn=make_tfr_input_fn(
                    train_dataset,
                    threads=threads,
                    offset=max(0, epoch*steps*batch),
                    limit=max(0, steps*batch),
                    shuffle=max(0, steps*batch),
                    prefetch=prefetch,
                    batch=batch,
                    repeat=-1,
                    seed=seed
                ),
                steps=(steps if steps >= 0 else None)
            )

            logger.info("Training epoch completed (%d[min])", (time.time() - train_start_time) / 60)

        seed += seed * 11113 + seed

        # evaluate model
        eval_start_time = time.time()
        logger.info("Evaluation epoch started")

        results = estimator.evaluate(
            input_fn=make_tfr_input_fn(
                eval_dataset,
                threads=threads,
                prefetch=prefetch,
                batch=batch,
            )
        )
        if results['loss'] < best_loss:
            delta_loss = results['loss'] - best_loss
            best_loss = results['loss']
            best_epoch = epoch
            best_checkpoint = estimator.latest_checkpoint()
            if abs(delta_loss) >= delta_loss_threshold * abs(results['loss']):
                stable_loss_counter = 0
                status = 'learn(%.01e)' % delta_loss
            else:
                stable_loss_counter += 1
                status = 'stable(%d/%d)' % (stable_loss_counter, max_stable_loss_counter)
            overfit_loss_counter = 0
        elif results['loss'] > (best_loss * 1.1):
            stable_loss_counter += 1
            overfit_loss_counter += 1
            status = 'overfit(%d/%d)' % (overfit_loss_counter, max_overfit_loss_counter)
        else:
            stable_loss_counter += 1
            status = 'stable(%d/%d)' % (stable_loss_counter, max_stable_loss_counter)

        for i in range(len(classes)):
            logger.info(
                "Class %d: precision=%02.02f%%, recall=%02.02f%%",
                i,
                results['precision_c%d' % i] * 100,
                results['recall_c%d' % i] * 100
            )
        logger.info(
            "Evaluation epoch completed: status=%s, loss=%.03e, jaccard=%.03f (%d[min])",
            status,
            results['loss'],
            results['jaccard'],
            (time.time() - eval_start_time) / 60
        )

        # write epoch logs
        epochlogs.append({
            'epoch': epoch_offset + epoch + 1,
            'checkpoint': Path(estimator.latest_checkpoint()).name,
            'duration': int(time.time() - epoch_start_time),
            'datasets': {
                'train': train_sources,
                'eval': eval_source,
            },
            'loss': float(results['loss']),
            'learning': status.startswith('learn'),
            'status': status,
            'jaccard': float(results['jaccard']),
            'precision': [ float(results['precision_c%d' % i]) for i in range(len(classes)) ],
            'recall': [ float(results['recall_c%d' % i]) for i in range(len(classes)) ],
            'f1': [ f1_score(results['precision_c%d' % i], results['recall_c%d' % i]) for i in range(len(classes)) ],
        })
        epochsfile.write_text(json.dumps(epochlogs, indent=' '))

        logger.info("Epoch completed (%d[min])", (time.time() - epoch_start_time) / 60)

    logger.info("Training completed: best epoch=%d (%d[min])", best_epoch + 1, (time.time() - train_time) / 60)
    logger.info("Total elapsed time: %d[min]", (time.time() - start_time) / 60)
