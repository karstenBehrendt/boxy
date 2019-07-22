""" Result object to keep track of true positives, false positives,
and false negatives
"""

class Result:
    """ Individual tp / fp / fn object with additional information """
    def __init__(self, result_type, confidence=None, size=None):
        """ Results object

        Parameters
        ----------
        result_type : str
                      'tp', 'fp', 'fn'
        confidence: float
                    detection confidence [0, 1], tp and fp only
        size : float
               size annotation, tp and fn only
        """
        self.type = result_type
        self.confidence = confidence
        self.size = size

    def __repr__(self):
        return str(self.__dict__)
