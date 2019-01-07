#!/usr/bin/env python3
import sys
import arrow

# Konvertera datumformatet
def convert_time(text):
    conversions = [
        arrow.get,
        lambda text: arrow.get(text, 'YYYY-M-D HH:mm:ss'),
        lambda text: arrow.get(text[:10] + '.' + text[-3:]),
        lambda text: arrow.get(text[:-4], 'ddd, D MMMM YYYY HH:mm:ss'),
        lambda text: arrow.get(text, 'YYYY-MM-DDTHH:mm:ssZZ')
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
        date = convert_time(line[field_start+1:field_end])
        if 7 <= date.hour < 20 and date.weekday() <= 4:
            new_date = date.format('YYYY-MM-DDTHH:mm:ssZZ')
            sys.stdout.write(
                line[:field_start+1] + new_date + line[field_end:]
            )
