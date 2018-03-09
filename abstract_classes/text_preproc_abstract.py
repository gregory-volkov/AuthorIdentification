from my_utils import composition


class TextPreprocessingItem(object):

    # Mapping str -> str
    @staticmethod
    def filter(self, text):
        raise NotImplementedError


class TextPreprocessing(TextPreprocessingItem):
    # Represents a composition of TextPreprocessingItem's

    def __init__(self, *prepr_items):
        self.filter = composition(*(item.filter for item in prepr_items))
