#!/bin/sh
domain_file_name=$1
SVM_file_name=$2

#HERA Domain tests with 5 featrures
python3 train.py $domain_file_name  vae_mag -p hera_domain_tests -l 2
python3 train.py $domain_file_name  skip_mag_phase -p hera_domain_tests -l 2
python3 train.py $domain_file_name  vae_phase -p hera_domain_tests -l 2
python3 train.py $domain_file_name  vae_imag -p hera_domain_tests -l 2
python3 train.py $domain_file_name  vae_real -p hera_domain_tests -l 2
python3 train.py $domain_file_name  skip_real_imag -p hera_domain_tests -l 2

#SVM Embedding tests
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 2 
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 3
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 4
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 5
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 6
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 7
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 8
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 9
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 10 
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 20 
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 50 
python3 train.py $SVM_file_name  skip_mag_phase -p SVM_embedding_tests -l 100

##################################################################
##################################################################
####################### Depreciated ##############################
##################################################################
##################################################################
