# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:55:22 2020

@author: Vishwajit
"""

def write_file(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)

import sys
import os
from os import path

from Connector_mysql import connect
def read_db():
    i = 0
    try:
        conn1=connect()
        cursor1=conn1.cursor()
        if not path.exists("Suspicious_Frames"):
            os.mkdir("Suspicious_Frames")
        list = os.listdir("Suspicious_Frames") # dir is your directory path
        number_files = len(list)
        #print (number_files)
        #####################################
        sql1='select * from LOG'
        cursor1.execute(sql1)
        m = cursor1.fetchall()
        #####################################
        print("Number\tCamera ID\tSEVERITY")
        i = number_files
        for row in m:
            data_d = row[2]
            print(str(row[0])+"\t"+str(row[1])+"\t\t"+row[3])
            i = i+1
            with open('Suspicious_Frames\\test'+str(i)+'.jpg','wb') as f:
               f.write(data_d)

        cursor1.close() #closes the cursor
        count  = i - number_files 
    except NameError as e:
        print(e)
    finally:
        print("Retrieved "+ str(count)+" frames")
        cursor1.close()
        conn1.close()
    
def main():
    
    read_db()
 
if __name__ == '__main__':
    main()