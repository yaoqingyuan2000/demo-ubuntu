import pymysql
import traceback
from multiprocessing import Pool,Manager,cpu_count
from multiprocessing.managers import BaseManager
import os,sys,time
import random

conn = pymysql.connect(
    host='127.0.0.1',
    port=33066,
    user='yqy',
    passwd='2000',
    db='test',
    charset='utf8')
cur = conn.cursor()

def func(id):

    # conn = pymysql.connect(
    #     host='127.0.0.1',
    #     port=33066,
    #     user='yqy',
    #     passwd='2000',
    #     db='test',
    #     charset='utf8')
    # cur = conn.cursor()


    for i in range(1000):
        name = 'yqy' + str(id)
        age = random.randint(1,100)
        tel='1'+str(random.choice([3,5,7,8]))+str(random.random())[2:11]

        sql="insert into student(name,age,tel) values('%s','%s','%s')"%(name,age,tel)

        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            print(e)

if __name__ == '__main__':

    
    time_start = time.time()

    # sql_table='''
    #     create table student(
    #       id int not null auto_increment,
    #       name varchar(20) not null,
    #       age int default 0,
    #       tel varchar(13),
    #       primary key(id)
    #     )engine=innodb character set utf8;
    # '''

    try:
        # cur.execute('drop table if exists student;')
        # cur.execute(sql_table)
        cur.execute('truncate table student;')
    except Exception as e:
        print(e)

    p = Pool(cpu_count())
    for i in range(5):
        p.apply_async(func, (i, ))
        # p.apply(func, (i, ))
        # func(i)

    p.close()
    p.join()
    cur.close()
    conn.close()
    time_end = time.time()
    time_c= time_end - time_start
    print('time cost', time_c, 's')



    


