import grpc 

from src.utils.logger import get_custom_logger


class GRPCServiceClientBase(object):

    def __init__(self, host, port, name):
        self.host = host 
        self.port = port 
        self.name = name    
        self.stub = None
        self.channel = None
        self.logget = get_custom_logger(name=__name__)
        self.start()

    def start(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        self.stub = self.create_stub()
        self.logget.info(f"GRPC Client {self.name} started on {self.host}:{self.port}")