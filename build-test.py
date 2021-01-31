import sys
from determiner import MultilingualPunctuation

file_path = sys.argv[1]
det = MultilingualPunctuation()

def convert(input_string):
    input_string = input_string.replace(' ', '').replace('\u2581', ' ').strip()
    return input_string

last = ''
first = True
with open(file_path) as i:
    for line in i:
        if not first:
            untok = convert(line)
            if last == '' or untok == '':
                print('\n' + line.strip(), end='')
                if untok != '':
                    last = untok.strip().split()[-1]
                else:
                    last = ''
            else:
                if line == '':
                    print('')
                    last = ''
                elif det(last, ' ' + untok.split()[0]):
                    print(' ' + line.strip(), end='')
                    last = untok.strip().split()[-1]
                else:
                    print('\n' + line.strip(), end='')
                    last = untok.strip().split()[-1]
        else:
            first = False
            print(line.strip(), end = '')
            try:
                last = convert(line).strip().split()[-1]
            except:
                last = ''
