import re
import sys
prefix=sys.argv[1]
filen=sys.argv[2]
with open(filen,'r') as f:
    line = f.readline()
# make sure your SacreBleu version is ver.1.5.1, otherwise you need to rewrite the RegExp below
res = re.findall('version.1.5.1 = ([0123456789\.]*)',line)[0]
print('prefix:{}, bleu:{}'.format(prefix,res))
