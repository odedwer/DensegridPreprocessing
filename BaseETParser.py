from abc import ABC, abstractmethod


class BaseETParser(ABC):
    """
    Base class for ET parsers, cannot be instantiated (abstract). To use, inherit from this class and override methods
    """

    # there might be a need to define more "must have" properties such as this
    @property
    def TIME(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def parse_sample(line):
        """
        parses a sample from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_msg(line):
        """
        parses a message line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_input(line):
        """
        parses a trigger line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_fixation(line):
        """
        parses a fixation line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_saccade(line):
        """
        parses a saccade line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_blinks(line):
        """
        parses a blink line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def parse_recordings(line):
        """
        parses a recording start/end line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @staticmethod
    @abstractmethod
    def is_sample(line):
        """
        checks if a line is a sample line
        :param line: line to check
        :return: True if line is sample, else False
        """
        pass
