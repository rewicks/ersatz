# turns input bpe file into char bpe file
# prints to std out

import sys

file_path = sys.argv[1]

with open(file_path) as i:
    for line in i:
        line = line.strip().replace(' ', '')
        line = ' '.join(line)
        print(line)

