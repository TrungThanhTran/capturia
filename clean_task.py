from database import DBHandler

dbhandler = DBHandler()

del_flag = dbhandler.delete_all_db(tb_name='DONE_QUEUE')
print(del_flag)

