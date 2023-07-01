from database import DBHandler
import sys
sys.path.append('../')
dbhandler = DBHandler()

ret_task = dbhandler.check_db('TASK_QUEUE')
print('task queue = ', ret_task)
done_task = dbhandler.check_db('DONE_QUEUE')
print('done queue = ', done_task)
error_task = dbhandler.check_db('ERROR_QUEUE')
print('error queue = ', error_task)






