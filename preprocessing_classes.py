import re
from abstract_classes.text_preproc_abstract import TextPreprocessingItem


class PrepLowerCase(TextPreprocessingItem):
    """
    Makes string lowercase
    """

    @staticmethod
    def filter(self, text):
        return text.lower()


class PrepStringRemove(TextPreprocessingItem):
    """
    Removes substrings using string iterator or regex
    """
    regex = None
    black_list = None

    @staticmethod
    def set_black_list(str_iter):
        PrepStringRemove.black_list = str_iter

    @staticmethod
    def set_regex(pattern):
        PrepStringRemove.regex = re.compile(pattern)

    @staticmethod
    def filter(self, text):
        if PrepStringRemove.black_list:
            for sub in PrepStringRemove.black_list:
                text = text.replace(sub, '')
        else:
            text = PrepStringRemove.regex.sub(text)
        return text
