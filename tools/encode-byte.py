from tqdm import tqdm
import sys
import re

f, o = sys.argv[1], sys.argv[2]
with open(f,'r')as f:
    content = f.readlines()
output = []
for line in content:
    newline = line.strip().split('__ ')[-1] # sentencepiece sign
    # num = line.strip().split(' ')
    num = newline.split(' ')
    num_tokens = [t if (t != '<unk>' and t != '<<unk>>') else '32' for t in num]
    str_tokens = bytes(map(int, num_tokens)).decode('utf-8', 'ignore')
    output.append(str_tokens+'\n')
with open(o, 'w')as f:
    for line in output:
        f.write(line)
