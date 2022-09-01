#!/bin/bash

data_dir=$1
store_dir=$2
size=$3

if [ $# -lt 3 ]
then
    echo "Usage: $0 <data_dir> <store_dir> <size>" # data_dir is where we stored the 500GB dataset, which is scaled by scale_dataset.py
    exit 1
fi

# Fix given paths i.e. remove trailing or extra slashes
data_dir=$(realpath -s  --canonicalize-missing $data_dir)
store_dir=$(realpath -s  --canonicalize-missing $store_dir)

if [ "$size" = "16" ]; 
then
    size=17179869184
elif [ "$size" = "200" ]; 
then
    size=214748364800
elif [ "$size" = "256" ]; 
then
    size=274877906944

elif [ "$size" = "500" ];
then
    size=536870912000
else
    echo "Wrong memory size input! (only support 16, 200, 256 GB)"
    exit 1
fi


# create softlinks from /source_data to /data

total_size=0 # reset total size
for file in "${data_dir}"/*; do
    file_size=$(stat -c '%s' "$file")
    file_name=$(dirname $file)
    base_name=$(basename $file)
    file_name="${file_name}/${base_name}"
    total_size=$(($total_size + $file_size))
    store_name="${store_dir}/${base_name}"

    ln -s $file_name $store_name
    if [[ "$total_size" -ge "$size" ]] 
    then
        break
    fi
done
file_num=$(ls ${store_dir} | wc -l)
odd_even=$(($file_num%2))
if [[ ! "$odd_even" == "0" ]]
then
    echo "Found odd number of files, remove the last one: ${store_name}"
    rm -rf $store_name
fi
echo "We want "$size"B, and we got "$total_size"B" 
exit 0