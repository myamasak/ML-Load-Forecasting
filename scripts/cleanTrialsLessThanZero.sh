#!/bin/bash
# file='editors.txt'
# while read line; do
# echo $line
# done < $file

dir1="C:\Users\z003t8hn\nni\experiments\LiNYuulo\trials"

echo "Searching folder: $dir1"
# cd $dir1
# if [ -d $dir1 ]; then
# echo "Files found: $(find $dir1 -type f | wc -l)"
# echo "Folders found: $(find $dir1 -type d | wc -l)"
# echo "metrics found: $(awk '/value/' metrics)"
# else
# echo "[ERROR] Please retry with another folder."
# exit 1
# fi

# find -type f -exec awk -F: '$3 ~ /value/' {} > C:\Users\z003t8hn\code\ML\ML-Load-Forecasting\cleaningLog.txt \;
# find -name metrics -type f awk -F: '/value/'

# while IFS= read -r line; do
#     echo "Text read from file: $line"
# done < "$1"

cd $dir1
# file="metrics"
# for file in $dir1\\*; do
#     while read -r line; do
#         echo "$line"
#     done < "$file"
# done

# value=$(find -name metrics -type f -print | xargs grep "value" | awk '{print $10}')

# if value<0; then
# echo "Value $value is < 0"
# else
# echo "Value $value is >= 0"
# fi

# find -name metrics -type f -print

# find -name metrics -type f -print | xargs grep "value" | awk '$10 < 0  {print ;}' | while read -r line; do
find -type f -name 'metrics' -exec awk '$10 < 0; {gsub($10,"0}")}' {} \;
# echo "$(cut -d : -f 1)"
# echo "$(awk '{print $10}')"
# echo "$(awk '{if (NR!=1) {print substr($10, 1, length($10)-1)}}')"
# echo "$(awk '{print $10}')"
# echo "$(awk '{sub($10,"0}")}1')"
# echo $line

# awk '{gsub($10,"0}")}' $(cut -d : -f 1)

# done

# find -name metrics -type f -print | xargs grep "value" | awk '$10 < 0  {print ;}'

# echo "one potato two potato" | awk '{gsub(/potato/,"banana")}1'
# echo "./dKlQ7/.nni/metrics:ME000113{"parameter_id": 165, "trial_job_id": "dKlQ7", "type": "FINAL", "sequence": 0, "value": -2.5578110189387706e+19}" | awk '{gsub($10,"0}")}1'