import abc


class DataLoader(object):

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loader(self):
        raise NotImplementedError
