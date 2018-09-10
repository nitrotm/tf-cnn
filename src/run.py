###########################################
# run.py
#
# Tensorflow command-line tools to prepare datasets,
# train/evaluate cnn models and do 3rd-party predictions.
#
# author: Antony Ducommun dit Boudry
# license: GPL
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, multiprocessing, os, sys

from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


tf.logging.set_verbosity(tf.logging.WARN)
if Path('lib/s3_file_system.so').exists():
    tf.load_file_system_library('lib/s3_file_system.so')

def cnn_train(args):
    from cnn.train import run
    run(
        args.device,
        args.model,
        args.topology,
        args.version,
        args.evalsource,
        args.trainsource,
        args.threads,
        args.prefetch,
        args.batch,
        args.steps,
        args.epochs,
        args.ram,
        args.initializer,
        args.regularizer,
        args.constraint,
        args.activation,
        args.bn,
        args.lrn,
        args.dropoutrate,
        args.labelweight,
        args.loss,
        args.optimizer,
        args.learningrate,
        args.learningdecay,
        args.verbose,
        args.seed
    )

def cnn_trainmulti(args):
    from cnn.trainmulti import run
    run(
        args.config,
        args.model,
        args.tensorboard,
        args.always,
        args.fresh,
        args.device,
        args.threads,
        args.prefetch,
        args.ram,
        args.verbose
    )

def cnn_eval(args):
    from cnn.eval import run
    run(
        args.device,
        args.model,
        args.source,
        args.threads,
        args.prefetch,
        args.batch,
        args.steps,
        args.verbose,
        args.seed
    )

def cnn_predict(args):
    from cnn.predict import run
    run(
        args.device,
        args.model,
        args.source,
        args.threads,
        args.prefetch,
        args.batch,
        args.epochs,
        args.verbose,
        args.seed
    )


def ds_bake(args):
    from ds.bake import run
    run(
        args.device,
        args.name,
        args.source,
        args.compression,
        args.batch,
        args.testratio,
        args.evalratio,
        args.size,
        args.blur,
        args.blurscale,
        args.centercrop,
        args.scales,
        args.flips,
        args.rotations,
        args.crops,
        args.brightness,
        args.contrast,
        args.gnoise,
        args.unoise,
        args.seed
    )

def ds_cleanmask(args):
    from ds.cleanmask import run
    run(
        args.source,
        args.threshold,
        args.open,
        args.close,
        args.minarea,
        args.expand,
        args.preview
    )

def ds_genweight(args):
    from ds.genweight import run
    run(
        args.source,
        args.expand,
        args.ratio,
        args.border,
        args.preview
    )

def ds_importflower(args):
    from ds.importflower import run
    run(
        args.source,
        args.destination
    )

def ds_importmve(args):
    from ds.importmve import run
    run(
        args.source,
        args.destination
    )

def ds_jpegorient(args):
    from ds.jpegorient import run
    run(
        args.source
    )

def ds_merge2d(args):
    from ds.merge2d import run
    run(
        args.source,
        args.destination
    )

def ds_preview(args):
    from ds.preview import run
    run(
        args.source
    )

def ds_tfrdist(args):
    from ds.tfrdist import run
    run(
        args.source,
        args.threads,
        args.verbose,
        args.seed
    )

def ds_tfrlist(args):
    from ds.tfrlist import run
    run(
        args.source,
        args.threads,
        args.prefetch,
        args.batch,
        args.steps,
        args.epochs,
        args.verbose,
        args.seed
    )

def ds_tfrorder(args):
    from ds.tfrorder import run
    run(
        args.source,
        args.threads,
        args.prefetch,
        args.batch,
        args.steps,
        args.epochs,
        args.verbose,
        args.seed
    )

CPU_COUNT = multiprocessing.cpu_count()

mainparser = argparse.ArgumentParser(description='U-Net CNN in TensorFlow.')

subparser = mainparser.add_subparsers(title='COMMANDS', description='Commands related to dataset manipulations and cnn model training, evaluation and prediction.')

parser = subparser.add_parser('train', description='Train tensorflow model.')
parser.add_argument('model', help='target model folder')
parser.add_argument('topology', help='model topology')
parser.add_argument('evalsource', help='tensorflow tf-record file for evaluation')
parser.add_argument('trainsource', nargs='+', help='tensorflow tf-record file(s) for training')
parser.add_argument('--device', default="/cpu:0", help='tensorflow device: /cpu:0, /gpu:0')
parser.add_argument('--version', type=int, default=2, help='model topology version')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='training batch size')
parser.add_argument('--steps', type=int, default=-1, help='maximum training steps (or -1 for all in dataset)')
parser.add_argument('--epochs', type=int, default=1, help='training/evaluation epochs')
parser.add_argument('--ram', action='store_true', default=False, help='cache training/eval datasets in ram memory')
parser.add_argument('--initializer', default='none', help='kernel initializer: none, glorot_uniform, he_uniform, glorot_normal, he_normal')
parser.add_argument('--regularizer', default='none', help='kernel regularizer: none, l1:COEF, l2:COEF, l1_l2:COEF1:COEF2')
parser.add_argument('--constraint', default='none', help='kernel constraint: none, nonneg, unit, max:MAX, minmax:MIN:MAX')
parser.add_argument('--activation', default='sigmoid', help='activation function: sigmoid, tanh, relu, crelu, lrelu, elu')
parser.add_argument('--bn', action='store_true', default=False, help='enable batch normalization')
parser.add_argument('--lrn', type=int, default=0, help='local response normalization radius (0 means none)')
parser.add_argument('--dropoutrate', type=float, default=0.0, help='dropout rate: 0.0 to 1.0')
parser.add_argument('--labelweight', type=float, default=0.0, help='label weighting factor')
parser.add_argument('--loss', default='absdiff', help='loss function: absdiff, huber:DELTA, mse, log, hinge')
parser.add_argument('--optimizer', default='gd', help='optimizer function: gd, momentum:MOMENTUM, nesterov:MOMENTUM, rmsprop:DECAY:MOMENTUM, adam:BETA1:BETA2, nadam:BETA1:BETA2')
parser.add_argument('--learningrate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--learningdecay', default='none', help='learning rate inverse time decay: none, poly:STEPS:ENDRATE:POWER, idt:STEPS:RATE, exp:STEPS:RATE, cos:STEPS:ALPHA, lincos:STEPS:NUMPERIODS:ALPHA:BETA, noisylincos:STEPS:VAR:VARDECAY:NUMPERIODS:ALPHA:BETA')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=cnn_train)

parser = subparser.add_parser('trainmulti', description='Train multiple tensorflow models.')
parser.add_argument('config', help='training configuration (yaml)')
parser.add_argument('model', help='target model folder prefix')
parser.add_argument('--tensorboard', action='store_true', default=False, help='launch tensorboard during training')
parser.add_argument('--always', action='store_true', default=False, help='run configuration instance even if exists')
parser.add_argument('--fresh', action='store_true', default=False, help='don\'t import initial model data')
parser.add_argument('--device', default="/cpu:0", help='tensorflow device: /cpu:0, /gpu:0')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--ram', action='store_true', default=False, help='cache training/eval datasets in ram memory')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.set_defaults(action=cnn_trainmulti)

parser = subparser.add_parser('eval', description='Evaluate tensorflow model.')
parser.add_argument('model', help='target model folder')
parser.add_argument('source', nargs='+', help='tensorflow tf-record file(s)')
parser.add_argument('--device', default="/cpu:0", help='tensorflow device: /cpu:0, /gpu:0')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='evaluation batch size')
parser.add_argument('--steps', type=int, default=-1, help='maximum evaluation steps (or -1 for all in dataset)')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=cnn_eval)

parser = subparser.add_parser('predict', description='Make prediction using tensorflow model.')
parser.add_argument('model', help='source model folder')
parser.add_argument('source', nargs='+', help='source image folder(s) or tensorflow tf-record file(s)')
parser.add_argument('--device', default="/cpu:0", help='tensorflow device: /cpu:0, /gpu:0')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='prediction batch size')
parser.add_argument('--epochs', type=int, default=1, help='prediction epochs (number of repeats)')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=cnn_predict)

parser = subparser.add_parser('bake', description='Generate tensorflow tf-record datasets.')
parser.add_argument('name', help='tf-record name (file prefix)')
parser.add_argument('source', nargs='+', help='source images folder(s)"')
parser.add_argument('--device', default="/cpu:0", help='tensorflow device: /cpu:0, /gpu:0')
parser.add_argument('--compression', default="none", help='tf-record compression: none | zlib | gzip')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='processing batch size: 1 to n')
parser.add_argument('--testratio', type=float, default=0.2, help='test sample ratio: 0 to 1')
parser.add_argument('--evalratio', type=float, default=0.1, help='evaluation sample ratio: 0 to 1')
parser.add_argument('--size', type=int, default=256, help='square image size (pixels)')
parser.add_argument('--blur', type=int, default=0, help='mask/weight image gaussian blur radius: 0 to 15')
parser.add_argument('--blurscale', type=float, default=1.0, help='mask/weight image gaussian blur scale factor: 0 to 1.0')
parser.add_argument('--centercrop', type=float, default=1.0, help='center crop ratio: 0.0 to 1.0')
parser.add_argument('--scales', nargs='*', type=float, default=[1.0], help='image scale ratio(s): between 0 and 1')
parser.add_argument('--flips', nargs='*', default=['none'], help='image flip(s): none | x | y | xy')
parser.add_argument('--rotations', nargs='*', type=float, default=[0], help='image rotation(s) in degree: between -180 and 180')
parser.add_argument('--crops', type=int, default=1, help='number of random crops: 1 to n')
parser.add_argument('--brightness', type=float, default=0.0, help='random brightness magnitude: 0.0 to 1.0')
parser.add_argument('--contrast', type=float, default=0.0, help='random contrast magnitude: 0.0 to 1.0')
parser.add_argument('--gnoise', type=float, default=0.0, help='random gaussian noise stddev: 0.0 to inf')
parser.add_argument('--unoise', type=float, default=0.0, help='random uniform noise magnitude: 0.0 to 1.0')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=ds_bake)

parser = subparser.add_parser('cleanmask', description='Clean training/evaluation masks.')
parser.add_argument('source', nargs='+', help='images folder(s)')
parser.add_argument('--preview', action='store_true', default=False, help='only preview output')
parser.add_argument('--threshold', type=float, default=1e-6, help='binary threshold: 0.0 to 1.0')
parser.add_argument('--open', type=int, default=0, help='morphological open kernel size (in pixels)')
parser.add_argument('--close', type=int, default=0, help='morphological close kernel size (in pixels)')
parser.add_argument('--minarea', type=int, default=0, help='minimum positive area (in pixels)')
parser.add_argument('--expand', type=int, default=0, help='expand area (in pixels)')
parser.set_defaults(action=ds_cleanmask)

parser = subparser.add_parser('genweight', description='Generate training weights.')
parser.add_argument('source', nargs='+', help='images folder(s)')
parser.add_argument('--preview', action='store_true', default=False, help='only preview output')
parser.add_argument('--expand', type=int, default=0, help='expand area from mask (in pixels)')
parser.add_argument('--ratio', type=float, default=0.5, help='positive vs. negative ratio: 0.0 to 1.0')
parser.add_argument('--border', default="0:1.0", help='border size and pixel scale: SIZE:SCALE')
parser.set_defaults(action=ds_genweight)

parser = subparser.add_parser('importflower', description='Import flower dataset.')
parser.add_argument('source', help='original flower dataset folder"')
parser.add_argument('destination', help='target images folder')
parser.set_defaults(action=ds_importflower)

parser = subparser.add_parser('importmve', description='Import mve scene for 2d tagging.')
parser.add_argument('source', nargs='+', help='mve scene folder(s): "name:path"')
parser.add_argument('destination', help='target images folder')
parser.set_defaults(action=ds_importmve)

parser = subparser.add_parser('jpegorient', description='Rotate jpeg according to exif orientation tag.')
parser.add_argument('source', nargs='+', help='images folder(s)')
parser.set_defaults(action=ds_jpegorient)

parser = subparser.add_parser('merge2d', description='Merge 2d tagging folders.')
parser.add_argument('source', nargs='+', help='tagged images folder(s)')
parser.add_argument('destination', help='target images folder')
parser.set_defaults(action=ds_merge2d)

parser = subparser.add_parser('preview', description='Preview tensorflow datasets.')
parser.add_argument('source', nargs='+', help='tensorflow tf-record file(s)')
parser.set_defaults(action=ds_preview)

parser = subparser.add_parser('tfrdist', description='Estimate tensorflow dataset distribution.')
parser.add_argument('source', nargs='+', help='tensorflow tf-record file(s)')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=ds_tfrdist)

parser = subparser.add_parser('tfrlist', description='List tensorflow dataset items.')
parser.add_argument('source', nargs='+', help='tensorflow tf-record file(s)')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='batch size')
parser.add_argument('--steps', type=int, default=-1, help='maximum steps (or -1 for all in dataset)')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=ds_tfrlist)

parser = subparser.add_parser('tfrorder', description='Check ordering of tensorflow dataset items.')
parser.add_argument('source', nargs='+', help='tensorflow tf-record file(s)')
parser.add_argument('--threads', type=int, default=CPU_COUNT, help='input pipeline concurrency')
parser.add_argument('--prefetch', type=int, default=CPU_COUNT, help='input pipeline prefetch')
parser.add_argument('--batch', type=int, default=CPU_COUNT, help='batch size')
parser.add_argument('--steps', type=int, default=-1, help='maximum steps (or -1 for all in dataset)')
parser.add_argument('--epochs', type=int, default=1, help='epochs')
parser.add_argument('--verbose', action='store_true', default=False, help='verbose log/output')
parser.add_argument('--seed', type=int, default=75437, help='pseudo-random generator seed')
parser.set_defaults(action=ds_tfrorder)

args = mainparser.parse_args()
if hasattr(args, 'action'):
    args.action(args)
else:
    mainparser.print_help()
