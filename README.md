# ECG authentication

* the codes are not arranged yet, just for refer.
* these are for NTU 2019Fall ADL final project.
* tyu have to install tensorboardX

db data: from ECG-ID Database (physionet)

acer data: collected by ourself, by an ACER watch

most of hyperparameters setting is in the code.

--------
usage:
db preprocess:  
`
python db_merge.py --input_file unfilecgdata.csv --outputfile ./YOUR_PATH/NAME
`

db train+test:  
`
python db_train.py --train_file db_1_50_400_train.csv --test_file db_1_50_400_test.csv
`

acer train+test:  
`
python acer_train.py --train_file 0628_train.csv --test_file 0628_test.csv
`
