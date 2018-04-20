from utils import composition


class TextPreprocessingItem(object):

    # Mapping (str -> str) or (token_list -> token_list) or (str -> tokens)
    def mapping(self, text):
        raise NotImplementedError


# Represents a composition of TextPreprocessingItem's
class TextPreprocessing(object):

    prep_items = None
    mapping = None

    def __set_mapping__(self):
        self.prep_items = reversed(self.prep_items)
        self.mapping = composition(*(item.mapping for item in self.prep_items))
