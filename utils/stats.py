import io, json, re, sys

from pathlib import Path


def fmt_name(name):
  return name.replace('_', '\\_')

def best_fmt(fmt, value, best):
  value = fmt.format(value)
  best = fmt.format(best)
  if value == best:
    return '\\bf{' + value + '}'
  return value

def find_files(root, pattern):
  files = dict()
  for file in sorted(root.glob(pattern)):
    model = file.parent.name
    if model.startswith(root.name + '-'):
      model = model[len(root.name + '-'):]
    files[model] = file
  return files

def parse_epochs(root, files):
  best = {
    'loss': (None, 0),
    'jaccard': (None, 0),
    'c0': {
      'precision': (None, 0),
      'recall': (None, 0),
      'f1': (None, 0)
    },
    'c1': {
      'precision': (None, 0),
      'recall': (None, 0),
      'f1': (None, 0)
    }
  }

  trials = list()
  for file in files.values():
    model = file.parent.name
    if model.startswith(root.name + '-'):
      model = model[len(root.name + '-'):]

    elapsed = 0
    best_epoch = None
    for epoch in json.loads(file.read_text()):
      elapsed += epoch['duration']
      if best_epoch is None or epoch['learning']:
        best_epoch = epoch

    if best_epoch is not None:
      trial = {
        'name': model,
        'path': file.parent,
        'valid': True,
        'epoch': best_epoch['epoch'],
        'loss': best_epoch['loss'],
        'jaccard': best_epoch['jaccard'],
        'c0': {
          'precision': best_epoch['precision'][0],
          'recall': best_epoch['recall'][0],
          'f1': best_epoch['f1'][0]
        },
        'c1': {
          'precision': best_epoch['precision'][1],
          'recall': best_epoch['recall'][1],
          'f1': best_epoch['f1'][1]
        },
        'elapsed': elapsed
      }
      if best['loss'][0] is None or best['loss'][1] > best_epoch['loss']:
        best['loss'] = (model, best_epoch['loss'])
      if best['jaccard'][0] is None or best['jaccard'][1] < best_epoch['jaccard']:
        best['jaccard'] = (model, best_epoch['jaccard'])
      for i in range(2):
        for item in ['precision', 'recall', 'f1']:
          if best['c%d' % i][item][0] is None or best['c%d' % i][item][1] < trial['c%d' % i][item]:
            best['c%d' % i][item] = (model, trial['c%d' % i][item])
    else:
      trial = {
        'name': model,
        'path': file.parent,
        'valid': False,
        'epoch': 0,
        'loss': 0,
        'jaccard': 0,
        'precision': [0, 0],
        'recall': [0, 0],
        'f1': [0, 0],
        'elapsed': elapsed
      }
    trials.append(trial)

  return (trials, best)

def parse_evals(root, files):
  best = {
    'loss': (None, 0),
    'jaccard': (None, 0),
    'c0': {
      'precision': (None, 0),
      'recall': (None, 0),
      'f1': (None, 0)
    },
    'c1': {
      'precision': (None, 0),
      'recall': (None, 0),
      'f1': (None, 0)
    }
  }

  trials = list()
  for file in files.values():
    model = file.parent.name
    if model.startswith(root.name + '-'):
      model = model[len(root.name + '-'):]

    best_trial = json.loads(file.read_text())[-1]
    trial = {
      'name': model,
      'path': file.parent,
      'valid': True,
      'loss': best_trial['loss'],
      'jaccard': best_trial['jaccard'],
      'c0': {
        'precision': best_trial['precision'][0],
        'recall': best_trial['recall'][0],
        'f1': best_trial['f1'][0]
      },
      'c1': {
        'precision': best_trial['precision'][1],
        'recall': best_trial['recall'][1],
        'f1': best_trial['f1'][1]
      },
      'elapsed': best_trial['duration']
    }
    if best['loss'][0] is None or best['loss'][1] > best_trial['loss']:
      best['loss'] = (model, best_trial['loss'])
    if best['jaccard'][0] is None or best['jaccard'][1] < best_trial['jaccard']:
      best['jaccard'] = (model, best_trial['jaccard'])
    for i in range(2):
      for item in ['precision', 'recall', 'f1']:
        if best['c%d' % i][item][0] is None or best['c%d' % i][item][1] < best_trial[item][i]:
          best['c%d' % i][item] = (model, best_trial[item][i])
    trials.append(trial)

  return (trials, best)

def print_epochs(trials, best):
  for trial in sorted(trials, key=lambda t: t['name']):
    print("./task-predict %s dataset/medium-256x256-test.tfr" % trial['path'])

  print()
  print()

  for trial in sorted(trials, key=lambda t: t['name']):
    print(
      '\\textit{:20s} & {:>9s} & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>9s} & {:>9s} & {:>10s} \\\\ \\hline'.format(
        '{' + fmt_name(trial['name']) + ('' if trial['valid'] else '*') + '}',
        '{:d}'.format(trial['epoch']),
        best_fmt('{:.01f}', trial['c1']['precision'] * 100, best['c1']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['precision'] * 100, best['c0']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c1']['recall'] * 100,    best['c1']['recall'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['recall'] * 100,    best['c0']['recall'][1] * 100),
        best_fmt('{:.02f}', trial['c1']['f1'],              best['c1']['f1'][1]),
        best_fmt('{:.02f}', trial['c0']['f1'],              best['c0']['f1'][1]),
        best_fmt('{:.03f}', trial['jaccard'],               best['jaccard'][1])
      )
    )

  print()
  print()

  for trial in sorted(trials, key=lambda t: t['name']):
    if trial['elapsed'] < 0:
      continue
    print(
      '\\textit{:20s} & {:>8s} & {:>9s} & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>9s} & {:>9s} & {:>10s} \\\\ \\hline'.format(
        '{' + fmt_name(trial['name']) + '}',
        '{:d}'.format(trial['elapsed'] // 60),
        '{:d}'.format(trial['epoch']),
        best_fmt('{:.01f}', trial['c1']['precision'] * 100, best['c1']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['precision'] * 100, best['c0']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c1']['recall'] * 100,    best['c1']['recall'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['recall'] * 100,    best['c0']['recall'][1] * 100),
        best_fmt('{:.02f}', trial['c1']['f1'],              best['c1']['f1'][1]),
        best_fmt('{:.02f}', trial['c0']['f1'],              best['c0']['f1'][1]),
        best_fmt('{:.03f}', trial['jaccard'],               best['jaccard'][1])
      )
    )

  print()
  print()

def print_evals(trials, best):
  for trial in sorted(trials, key=lambda t: t['name']):
    if trial['elapsed'] < 0:
      continue
    print(
      '\\textit{:20s} & {:>8s} & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>10s}\\% & {:>9s} & {:>9s} & {:>10s} \\\\ \\hline'.format(
        '{' + fmt_name(trial['name']) + '}',
        '{:d}'.format(trial['elapsed'] // 60),
        best_fmt('{:.01f}', trial['c1']['precision'] * 100, best['c1']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['precision'] * 100, best['c0']['precision'][1] * 100),
        best_fmt('{:.01f}', trial['c1']['recall'] * 100,    best['c1']['recall'][1] * 100),
        best_fmt('{:.01f}', trial['c0']['recall'] * 100,    best['c0']['recall'][1] * 100),
        best_fmt('{:.02f}', trial['c1']['f1'],              best['c1']['f1'][1]),
        best_fmt('{:.02f}', trial['c0']['f1'],              best['c0']['f1'][1]),
        best_fmt('{:.03f}', trial['jaccard'],               best['jaccard'][1])
      )
    )

  print()
  print()


root = Path(sys.argv[1])


print("========================================")
print("TRAINING RESULTS")
print("========================================")
print()

trials, best = parse_epochs(root, find_files(root, "**/epochs.json"))
print_epochs(trials, best)


print("========================================")
print("EVAL RESULTS")
print("========================================")
print()

trials, best = parse_evals(root, find_files(root, "**/evals.json"))
print_evals(trials, best)
