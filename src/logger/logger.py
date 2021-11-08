import datetime
import sys

INFO = '[INFO]'
DEBUG = '[DEBUG]'
WARNING = '[WARNING]'
ERROR = '[ERROR]'
DATEFORMAT = "%d-%m-%Y %H:%M:%S:%f"

class Logger():

    def __init__(self, logfile='default.log'):
        self.stdout_fileno = sys.stdout
        sys.stdout = open(logfile, 'a')
    
    def __del__(self): 
        sys.stdout.close() 
        sys.stdout = self.stdout_fileno
    
    def Error(self, msg):
        message = f'{ERROR}[{datetime.datetime.now().strftime(DATEFORMAT)}]: {msg}\n'
        sys.stdout.write(message) # to logfile
        self.stdout_fileno.write(message) # to console
    
    def Info(self, msg):
        message = f'{INFO}[{datetime.datetime.now().strftime(DATEFORMAT)}]: {msg}\n'
        sys.stdout.write(message) # to logfile
        self.stdout_fileno.write(message) # to console

    def Debug(self, msg):
        message = f'{DEBUG}[{datetime.datetime.now().strftime(DATEFORMAT)}]: {msg}\n'
        sys.stdout.write(message) # to logfile
        self.stdout_fileno.write(message) # to console

    def Warning(self, msg):
        message = f'{WARNING}[{datetime.datetime.now().strftime(DATEFORMAT)}]: {msg}\n'
        sys.stdout.write(message) # to logfile
        self.stdout_fileno.write(message) # to console
    