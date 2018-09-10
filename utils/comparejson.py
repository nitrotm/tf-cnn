import json, os, sys

from pathlib import Path


results = dict()
for i in range(1, len(sys.argv)):
  path = Path(sys.argv[i])
  if not path.exists():
    print("File %s is missing" % path)
    sys.exit(1)
  items = set()
  for item in json.loads(path.read_text()):
    items.add(item['name'])
  results[sys.argv[i]] = items

# find shared elements
shared = None
for result in results.values():
  if shared is None:
    shared = result
  else:
    shared = shared.intersection(result)
if not shared:
  shared = set()

# find difference
for (path, result) in results.items():
  diff = result.difference(shared)
  if len(diff) > 0:
    print("%s is inconsistent: %s" % (path, ','.join(diff)))
  else:
    print("%s is consistent" % path)
