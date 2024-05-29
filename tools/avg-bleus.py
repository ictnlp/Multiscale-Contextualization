import re
import sys
filename=sys.argv[1]
with open(filename,'r')as f:
    content=f.readlines()
lst = []
for line in content:
    lst.append(float(re.findall('bleu:([0123456789\.]*?)\n',line)[0]))
res = sum(lst)/len(lst)
print('average bleu:{}'.format(res))
