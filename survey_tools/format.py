
import json
import sys

j = json.load(open(sys.argv[1]))
print(json.dumps(j, indent=2))