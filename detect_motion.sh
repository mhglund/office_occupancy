#!/bin/sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input file> <output file>"
fi

input_file=$1
output_file=$2

new_field_order() {
    cat "$input_file"                                                  \
        | tr -d '\r'                                                   \
        | sed -E 's/ *;/;/g'                                           \
        | awk -F\; -v OFS=\; '{print $5, $2, $1, $3, $4, $6}'
}

echo "$(new_field_order | head -n 1);movement" > "$output_file"

new_field_order | awk 'NR>1{print}'                                    \
    | ./convert_dates.py                                               \
    | sort                                                             \
    | awk -F\; -v OFS=\; '
    NR == 1 {
        movement = 0;
    }
    $1 == sensor && $NF > value {
        movement = 1;
    }
    {
        sensor = $1;
        value = $NF;
        $7 = movement;
        print $0;
        movement = 0;
    }' >> "$output_file"
