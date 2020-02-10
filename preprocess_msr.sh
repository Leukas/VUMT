file=$1
cut -f 2 $file | tail -n +2 | awk "NR%2==0" > msr_2.txt
cut -f 2 $file | tail -n +2 | awk "NR%2==1" > msr_1.txt

./preprocess_file.sh msr_1.txt en
./preprocess_file.sh msr_2.txt en