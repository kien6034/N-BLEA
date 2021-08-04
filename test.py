import logging
from os import path, mkdir

import sys


def generateLogger(inputFile):
    inputFileName = inputFile.replace('.txt', '')
    inputFileName = inputFileName.replace('Instances/', '')

    if not path.exists(f'result/{inputFileName}'):
        mkdir(f'result/{inputFileName}')
    fileLogName = f'result/{inputFileName}/{inputFileName}.log'


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    file_handler = logging.FileHandler(fileLogName)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

logger = generateLogger("Instances/10e.txt")

class Employee:
    """A sample Employee class"""

    def __init__(self, first, last):
        self.first = first
        self.last = last

        logger.info('Created Employee: {} - {}'.format(self.fullname, self.email))

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)


emp_1 = Employee('John', 'Smith')
emp_2 = Employee('Corey', 'Schafer')
emp_3 = Employee('Jane', 'Doe')