import torch.nn as nn
import logging

class Regularization(nn.Module):
    def __init__(self, type, variable, lambda_reg):

        super().__init__()

        self.type = type
        self.variable = variable
        self.lambda_reg = lambda_reg
        self.active = True

        self.logger = logging.getLogger("regularization")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def print(self):
        self.logger.info(f"  * Adding {self.type} regularization - Variable: {self.variable} - lambda={self.lambda_reg}")

    def __call__(self, x):
        pass
