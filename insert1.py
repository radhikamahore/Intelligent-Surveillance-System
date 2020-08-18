# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:35:33 2020

@author: Vishwajit
"""

from Connector_mysql import connect
import sys
import cv2
import os
def read_file(filename):
    with open(filename, 'rb') as f:
        photo = f.read()
    return photo

from mysql.connector import MySQLConnection, Error
 
def create_table(conn):
    #data = read_file(filename)
 
    try:
        cursor=conn.cursor()
        stmt = "SHOW TABLES LIKE 'LOG'"
        cursor.execute(stmt)
        result = cursor.fetchone()
        if not result:
            # there is a no table named "LOG"
            query= "CREATE TABLE IF NOT EXISTS LOG(Number INTEGER NOT NULL PRIMARY KEY AUTO_INCREMENT,CAM_ID INTEGER NOT NULL,IMG LONGBLOB  NOT NULL,SEVERITY varchar(50))"
            cursor.execute(query)
        '''m = read_file(filename)
        args = (m,102)
        cursor.execute("""INSERT INTO img(photo,id) VALUES(%s,%s)""",args)
        conn.commit() #ensures that its saved
        print("Inserted....")'''
    except Error as e:
        print(e)
    finally:
        cursor.close()
def insert_img(conn,image,cam_id,sev):
    
    try:
        #conn = connect()
        create_table(conn)
        a=0
        cursor=conn.cursor()
        a=a+1
        # Saves the frames with frame-count 
        #cv2.imwrite("frame%d.jpg" % count, image) 
        #sev='high'
        dim = (320, 240)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("frame%d.png" % a,resized)
        m = read_file("frame%d.png" % a)
        args = (cam_id,m,sev)
        cursor.execute("""INSERT INTO LOG(CAM_ID,IMG,SEVERITY) VALUES(%s,%s,%s)""",args)
        conn.commit() #ensures that its saved
        print("Inserted....")
        os.remove("frame%d.png" % a)
    except Error as e:
        print(e)    
    
def main():
    try:
        insert_img()
    except Error as e:
        print(e)
    finally:
        conn.close()
 
#if __name__ == '__main__':
#    main()