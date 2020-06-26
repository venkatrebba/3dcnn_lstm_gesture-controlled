import logging

class Logger(object):
    def __init__(self, name, logs_file="logs.log"):
        self.logger = logging.getLogger()
        self.name = name
        self.logs_file = logs_file
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        formatter = logging.Formatter(' %(levelname)s - %(message)s')

        #if(self.logger.hasHandlers()):
        #    self.logger.handlers.clear()

        # create console handler and set level to info
        logger = logging.getLogger(self.name)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # create file handler and set level to debug
        #handler = logging.FileHandler(logs_file, "w")
        logger = logging.getLogger(self.name)
        self.handler1=logging.FileHandler(self.logs_file, "w")
        self.handler1.setLevel(logging.DEBUG)
        self.handler1.setFormatter(formatter)
        self.logger.addHandler(self.handler1)

    def build(self):
        return self.logger


    def getLogger(self, name):
        return self.logger.getLogger(name)