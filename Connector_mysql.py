# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:55:25 2020

@author: Vishwajit
"""

import mysql.connector
from mysql.connector import Error
 
 
def connect():
    """ Connect to MySQL database """
    conn = None
    try:
        conn = mysql.connector.connect(host='localhost',
                                       database='python_test',
                                       user='root',
                                       password='password',use_pure=True)
        if conn.is_connected():
            print('Connected to MySQL database')
            return conn
 
    except Error as e:
        print(e)
 
    #finally:
     #   if conn is not None and conn.is_connected():
      #      conn.close()
 
 
if __name__ == '__main__':
    connect()
