# Python program to convert .tsv file to .csv file
# importing pandas library
import pandas as pd 
 
tsv_file='train.tsv'
 
# reading given tsv file
csv_table=pd.read_table(tsv_file,sep='\t')
 
# converting tsv file into csv
csv_table.to_csv('train.csv',index=False)
 
# output
print("Successfully made csv file")