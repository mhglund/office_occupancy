#!/usr/bin/env python3
import sys
import arrow

# Konvertera datumformatet
def convert_time(text):
    conversions = [
        arrow.get,
        lambda text: arrow.get(text, 'YYYY-M-D HH:mm:ss'),
        lambda text: arrow.get(text[:10] + '.' + text[-3:]),
        lambda text: arrow.get(text[:-4], 'ddd, D MMMM YYYY HH:mm:ss')
    ]
    for convert in conversions:
        try:
            return convert(text)
        except:
            continue
    raise ValueError('Date {} using unrecognised format'.format(text))

if __name__ == '__main__':
    for line in sys.stdin:
        field_start = line.find(';')
        field_end = line.find(';', field_start+1)
        new_date = convert_time(line[field_start+1:field_end])\
            .format('YYYY-MM-DDTHH:mm:ssZZ')
        sys.stdout.write(line[:field_start+1] + new_date + line[field_end:])
