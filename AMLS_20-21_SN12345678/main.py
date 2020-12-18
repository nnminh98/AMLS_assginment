import tensorflow as tf
from A1.A1 import load_test_data
from A1.A1 import mainA1
from A2.A2 import mainA2
from B2.B2 import mainB2
from B1.B1 import mainB1

# ======================================================================================================================
# Data preprocessing
#data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
test_lossA1, test_accA1 = mainA1('Datasets/celeba/', 'Datasets/celeba_test/')

# ======================================================================================================================
# Task A2
test_lossA2, test_accA2 = mainA2('Datasets/celeba/', 'Datasets/celeba_test/')

# ======================================================================================================================
# Task B1
test_lossB1, test_accB1 = mainB1('Datasets/cartoon_set/', 'Datasets/cartoon_set_test/')

# ======================================================================================================================
# Task B2
test_lossB2, test_accB2 = mainB2('Datasets/cartoon_set/', 'Datasets/cartoon_set_test/')

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{};TA2:{};TB1:{};TB2:{};'.format(test_accA1, test_accA2, test_accB1,test_accB2))