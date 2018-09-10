###########################################
# trainmulti.py
#
# Multiple model training using tensorflow.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

import json, multiprocessing, subprocess, time, yaml

from pathlib import Path

from utils import create_logger, set_logger_file

logger = create_logger(__name__)


def cnn_train_run(*args):
    from cnn.train import run
    set_logger_file(str(Path(args[1]) / ("train-" + time.strftime("%Y%m%d%H%M%S") + ".log")))
    run(*args)

def cnn_eval_run(*args):
    from cnn.eval import run
    set_logger_file(str(Path(args[1]) / ("eval-" + time.strftime("%Y%m%d%H%M%S") + ".log")))
    run(*args)


def best_checkpoint(src):
    epochs_file = src / "epochs.json"
    checkpoint_file = src / "checkpoint"
    checkpoint = None
    if epochs_file.exists():
        for epoch in json.loads(epochs_file.read_text()):
            if not epoch['learning']:
                continue
            checkpoint = epoch['checkpoint']
    if checkpoint is None and checkpoint_file.exists():
        checkpoints = yaml.load(checkpoint_file.read_text())
        if 'model_checkpoint_path' in checkpoints:
            checkpoint = checkpoints['model_checkpoint_path']
    return checkpoint

def copy_files(dst, src, patterns):
    for pattern in patterns:
        for srcfile in sorted(src.glob(pattern)):
            if srcfile.is_dir():
                continue
            dstfile = dst.joinpath(srcfile.relative_to(src))
            logger.debug("Copying %s", srcfile.relative_to(src))
            dstfile.parent.mkdir(parents=True, exist_ok=True)
            dstfile.write_bytes(srcfile.read_bytes())

def remove_directory(root, path=None):
    if path is None:
        path = root
    logger.debug("Removing %s", path)
    for child in path.iterdir():
        if child.is_dir():
            remove_directory(root, child)
        else:
            child.unlink()
    path.rmdir()


def run(configuration, model_path, tensorboard, always, fresh, device, threads, prefetch, ram_cache, verbose):
    """
    Multiple training task implementation.
    """

    # parse configurations
    models = list()
    data = yaml.load(Path(configuration).read_text())
    configuration_name = data.get('name', '%d' % time.time())
    global_params = data.get('global', dict())
    dataset = data.get('dataset', dict())
    if 'eval' not in dataset:
        raise Exception("Missing evaluation dataset in %s." % configuration)
    if 'train' not in dataset:
        raise Exception("Missing training dataset in %s." % configuration)
    for name in sorted(data.get('models', dict()).keys()):
        model_params = data['models'][name]
        local_params = dict(global_params)
        local_params.update(model_params)
        if 'topology' not in local_params:
            raise Exception("Configuration %s in %s has no topology defined." % (name, configuration))
        local_dataset = local_params.get('dataset', dict())
        models.append({
            'name': name,
            'test_source': local_dataset.get('test', dataset.get('test')),
            'eval_source': local_dataset.get('eval', dataset.get('eval')),
            'train_source': local_dataset.get('train', dataset.get('train')),
            'batch_size': local_params.get('batch_size', 1),
            'steps': local_params.get('steps', -1),
            'epochs': local_params.get('epochs', 1),
            'seed': local_params.get('seed', 75437),
            'topology': local_params['topology'],
            'topology_version': local_params.get('topology_version', 2),
            'optimizer': local_params.get('optimizer', 'gd'),
            'learning_rate': local_params.get('learning_rate', 1e-3),
            'learning_rate_decay': local_params.get('learning_rate_decay', 'none'),
            'loss': local_params.get('loss', 'abs'),
            'activation': local_params.get('activation', 'none'),
            'local_response_normalization': local_params.get('local_response_normalization', 0),
            'batch_normalization': local_params.get('batch_normalization', False),
            'dropout_rate': local_params.get('dropout_rate', 0.0),
            'initializer': local_params.get('initializer', 'none'),
            'regularizer': local_params.get('regularizer', 'none'),
            'constraint': local_params.get('constraint', 'none'),
            'label_weighting': local_params.get('label_weighting', 0.0),
        })

    # create model parent directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # launch tensorboard
    if tensorboard:
        logger.info("Starting tensorboard...")
        tensorboard_task = subprocess.Popen(
            ["tensorboard", "--logdir", model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    else:
        tensorboard_task = None

    # execute configurations
    logger.info("Starting multiple training in sequence")

    start_time = time.time()
    failures = 0
    skipped = 0
    for i in range(len(models)):
        model = models[i]
        model_key = configuration_name + '-' + model['name']
        target_path = Path(model_path) / model_key
        checkpoint_file = target_path / "checkpoint"

        if always or not checkpoint_file.exists():
            if (i - skipped) > 0:
                elapsed_time = (time.time() - start_time) / 60
                total_time = elapsed_time * ((len(models) - skipped) / (i - skipped))
                remaining_time = total_time - elapsed_time
                logger.info("[%d / %d] Starting training of configuration %s (remaining=%d[min], total=%d[min])", i + 1, len(models), model_key, remaining_time, total_time)
            else:
                logger.info("[%d / %d] Starting training of configuration %s", i + 1, len(models), model_key)

            tmp_path = Path("tmp-%s" % time.strftime('%Y%m%d%H%M%S'))
            tmp_path.mkdir(parents=True, exist_ok=True)

            # copy latest checkpoint?
            if not fresh and checkpoint_file.exists():
                checkpoints = yaml.load(checkpoint_file.read_text())
                if 'model_checkpoint_path' in checkpoints:
                    last_checkpoint = checkpoints['model_checkpoint_path']
                    copy_files(
                        tmp_path,
                        target_path,
                        [
                            'checkpoint',
                            'parameters.json',
                            'topology.yaml',
                            last_checkpoint + '.meta',
                            last_checkpoint + '.index',
                            last_checkpoint + '.data-*'
                        ]
                    )

            # run training task
            train_task = multiprocessing.Process(
                target=cnn_train_run,
                args=(
                    device,
                    str(tmp_path),
                    model['topology'],
                    model['topology_version'],
                    model['eval_source'],
                    [model['train_source']],
                    threads,
                    prefetch,
                    model['batch_size'],
                    model['steps'],
                    model['epochs'],
                    ram_cache,
                    model['initializer'],
                    model['regularizer'],
                    model['constraint'],
                    model['activation'],
                    model['batch_normalization'],
                    model['local_response_normalization'],
                    model['dropout_rate'],
                    model['label_weighting'],
                    model['loss'],
                    model['optimizer'],
                    model['learning_rate'],
                    model['learning_rate_decay'],
                    verbose,
                    model['seed']
                )
            )
            train_task.start()
            train_task.join()

            # keep only best checkpoint
            checkpoint = best_checkpoint(tmp_path)
            logger.debug("Keeping best checkpoint: %s", checkpoint)
            for chkfile in sorted(tmp_path.glob('model.ckpt-*')):
                if chkfile.name.startswith(checkpoint):
                    continue
                chkfile.unlink()
            (tmp_path / "checkpoint").write_text('model_checkpoint_path: "%s"' % checkpoint)

            # save model
            if train_task.exitcode == 0:
                if fresh:
                    remove_directory(target_path)
                copy_files(target_path, tmp_path, ['**/*'])
            else:
                failures += 1
            if tmp_path.exists():
                remove_directory(tmp_path)

            logger.info("[%d / %d] Finished training of configuration (code=%d)", i + 1, len(models), train_task.exitcode)
        else:
            logger.info("[%d / %d] Skipping training of configuration %s (exists)", i + 1, len(models), model_key)
            skipped += 1

        # evaluate model on best checkpoint?
        checkpoint = best_checkpoint(target_path)
        if model['test_source'] and checkpoint and (target_path / (checkpoint + '.index')).exists() and (always or not (target_path / "evals.json").exists()):
            logger.info("[%d / %d] Starting evaluation of configuration %s (checkpoint=%s)", i + 1, len(models), model_key, checkpoint)

            tmp_path = Path("tmp-%s" % time.strftime('%Y%m%d%H%M%S'))
            tmp_path.mkdir(parents=True, exist_ok=True)

            # copy best checkpoint only
            copy_files(
                tmp_path,
                target_path,
                [
                    'parameters.json',
                    'topology.yaml',
                    'evals.json',
                    checkpoint + '.meta',
                    checkpoint + '.index',
                    checkpoint + '.data-*'
                ]
            )
            (tmp_path / "checkpoint").write_text('model_checkpoint_path: "%s"' % checkpoint)

            # run evaluation task
            eval_task = multiprocessing.Process(
                target=cnn_eval_run,
                args=(
                    device,
                    str(tmp_path),
                    [model['test_source']],
                    threads,
                    prefetch,
                    model['batch_size'],
                    -1,
                    False,
                    model['seed']
                )
            )
            eval_task.start()
            eval_task.join()

            # save evaluation metrics
            if eval_task.exitcode == 0:
                copy_files(target_path, tmp_path, ['evals.json', '**/eval-*.log'])
            else:
                failures += 1
            if tmp_path.exists():
                remove_directory(tmp_path)

            logger.info("[%d / %d] Finished evaluation of configuration (code=%d)", i + 1, len(models), eval_task.exitcode)
        else:
            logger.info("[%d / %d] Skipping evaluation of configuration %s (exists)", i + 1, len(models), model_key)


    logger.info("Total elapsed time: %d[min] (failures=%d)", (time.time() - start_time) / 60, failures)

    # terminate tensorboard task
    if tensorboard_task is not None:
        logger.info("Stopping tensorboard...")
        tensorboard_task.terminate()
        tensorboard_task.wait()
        logger.info("Tensorboard stopped (code=%d).", tensorboard_task.returncode)
