class Posting:
    """
    Posting object
    doc_id (int): Document id
    tf (int): frequency count of the token appeared in the doc
    is_important (bool): is the token important in this doc
    """
    def __init__(self, doc_id, tf, is_important, positions=None):
        self.doc_id = doc_id
        self.tf = tf
        self.is_important = is_important
        self.positions = positions if positions else []