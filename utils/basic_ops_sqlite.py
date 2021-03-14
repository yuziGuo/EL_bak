import sqlite3 as lite
import sys
import os

'''
I. Schema, or information stored I want:

<t2d_root_path>/instance_cand_cache/candidate_entities.db (tb_tid)
row_id  || col_id   ||  mention(cell value) || lookup_order || label  ||
1       ||  1       ||  fly me to the moon  ||  

entity(uri) ||
http://dbpedia.org/resource/Fly_Me_to_the_Moon  ||

classes                     || RefCount || dbo:abstract(en) || rdf:comment(en)
Song; Musical Work; Work    ||  12      ||                  ||

II. 希望做到：提高速度，并行读写
https://www.tutorialspoint.com/sqlite/sqlite_python.htm
https://www.runoob.com/sqlite/sqlite-select.html
http://zetcode.com/db/sqlitepythontutorial/
'''

'''
Example 1:
sqlite> .databased
Error: unknown command or invalid arguments:  "databased". Enter ".help" for help
sqlite> .databases
main: /home/gyh/pack_for_debug_2/EL_yh/utils/test.db
sqlite> .tables
cars
sqlite> .quit

Example 2:
(commit) Sometimes committing is implicit ! 
'''

def test_db_build_and_connect():
    '''
    Basic operations (yh):
    1.  create db: py> lite.connect(<new_db_name>)
    2.  drop/del db: sh> rm test.db # no special command; just delete file
    3.  create table: http://zetcode.com/db/sqlitepythontutorial/
    '''
    #
    try:
        con = lite.connect('test.db')
        cur = con.cursor()
        cur.execute('SELECT SQLITE_VERSION()')
        data = cur.fetchone()[0]
        print("SQLite version: {}".format(data))
    except lite.Error as e:
        print("Error {}:".format(e.args[0]))
        sys.exit(1)
    finally:
        if con:
            con.close()
    return


def test_db_get_schema():
    try:
        con = lite.connect('test.db')
        cur = con.cursor()
        # cur.execute("SELECT tbl_name FROM sqlite_master WHERE type = 'table';")
        cur.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND tbl_name = 'cars';")
        rows = cur.fetchall()
        for row in rows:
            print(row)
    except:
        pass
    return


def test_tb_get_headers(base_path, db, tbl):
    '''
    :param base_path: ./UER-spider-temp/data/spider/slim_99/farm
    :param db: farm.sqlite
    :param tbl: city
    :return: [('City_ID', 'int'), ('Official_Name', 'text'), ('Status', 'text'), ('Area_km_2', 'real'), ('Population', 'real'), ('Census_Ranking', 'text')]
    '''
    con = lite.connect(os.path.join(base_path, db))
    cur = con.cursor()
    cur.execute("PRAGMA table_info({})".format(tbl))
    header = [(_[1], _[2]) for _ in cur.fetchall()]  # ('ID', 'integer'), ('Name', 'char(35)')
    con.close()

    return header




def test_tb_create_schema_and_insert():
    try:
        import ipdb; ipdb.set_trace()
        con = lite.connect('test.db')
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS cars(id INT, name TEXT, price INT)")
        cur.execute("INSERT INTO cars VALUES(1,'Audi',52642)")
        # test_select_by_cursor()
        cur.execute("INSERT INTO cars VALUES(2,'Mercedes',57127)")
        cur.execute("INSERT INTO cars VALUES(3,'Skoda',9000)")
        cur.execute("INSERT INTO cars VALUES(4,'Volvo',29000)")
        cur.execute("INSERT INTO cars VALUES(5,'Bentley',350000)")
        cur.execute("INSERT INTO cars VALUES(6,'Citroen',21000)")
        cur.execute("INSERT INTO cars VALUES(7,'Hummer',41400)")
        cur.execute("INSERT INTO cars VALUES(8,'Volkswagen',21600)")
        con.commit()
    except lite.Error as e:
        print("Error {}:".format(e.args[0]))
        if con:
            con.rollback()
        sys.exit(1)
    finally:
        if con:
            con.close()

def test_tb_del():
    pass

def test_insert():
    pass


def test_insert_by_block_2():
    # cur.executemany  cur.executemany("INSERT INTO cars VALUES(?, ?, ?)", cars)
    cars = (
        (1, 'Audi'),
        (2, 'Mercedes')
    )
    try:
        con = lite.connect('test.db')
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS cars")
        cur.execute("CREATE TABLE cars(id INT, name TEXT, price INT)")
        cur.executemany("INSERT INTO cars(id, name) VALUES(?, ?)", cars)
        con.commit()
    except lite.Error as e:
        print("Error {}:".format(e.args[0]))
        if con:
            con.rollback()
        sys.exit(1)
    finally:
        if con:
            con.close()


def test_insert_by_block():
    # cur.executemany  cur.executemany("INSERT INTO cars VALUES(?, ?, ?)", cars)
    cars = (
        (1, 'Audi', 52642),
        (2, 'Mercedes', 57127),
        (3, 'Skoda', 9000),
        (4, 'Volvo', 29000),
        (5, 'Bentley', 350000),
        (6, 'Hummer', 41400),
        (7, 'Volkswagen', 21600)
    )

    try:
        con = lite.connect('test.db')
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS cars")
        cur.execute("CREATE TABLE cars(id INT, name TEXT, price INT)")
        cur.executemany("INSERT INTO cars VALUES(?, ?, ?)", cars)
        con.commit()
    except lite.Error as e:
        print("Error {}:".format(e.args[0]))
        if con:
            con.rollback()
        sys.exit(1)
    finally:
        if con:
            con.close()

def test_insert_in_parallel():
    pass

def test_select_by_cursor(db_name='test.db', tb_name='cars'):
    try:
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute('SELECT * FROM {}'.format(tb_name))
        rows = cur.fetchall()
        for row in rows:
            print(row)
    except lite.Error as e:
        print('Errors: {}'.format(e.args[0]))
        sys.exit(1)
    finally:
        if con is not None:
            con.close()

def test_del_rows():
    try:
        con = lite.connect('test.db')
        cur = con.cursor()
        cur.execute('DELETE FROM cars where price>0')
        con.commit()
    except lite.Error as e:
        print('Errors: {}'.format(e.args[0]))
        if con:
            con.rollback()
        sys.exit(1)
    finally:
        if con:
            con.close()
    pass



def test_1():
    # all basics
    # test_db_build_and_connect()
    # test_tb_create_schema_and_insert()
    # test_select_by_cursor()
    # test_del_rows()
    # test_select_by_cursor()
    test_insert_by_block_2()
    test_db_show_schema()
    test_select_by_cursor()
    pass

def test_2():
    base_path = './UER-spider-temp/data/spider/slim_99/farm'
    db = 'farm.sqlite'
    tbl = 'city'
    headers = test_tb_get_headers(base_path, db, tbl)
    print(headers)

if __name__=='__main__':
    # test_1()
    test_2()


