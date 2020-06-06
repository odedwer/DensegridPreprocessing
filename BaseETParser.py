from abc import ABC, abstractmethod


class BaseETParser(ABC):
    """
    Base class for ET parsers, cannot be instantiated (abstract). To use, inherit from this class and override methods
    """

    @abstractmethod
    def parse_sample(self, line):
        """
        parses a sample from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_msg(self, line):
        """
        parses a message line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_input(self, line):
        """
        parses a trigger line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_fixation(self, line):
        """
        parses a fixation line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_saccade(self, line):
        """
        parses a saccade line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_blinks(self, line):
        """
        parses a blink line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def parse_recordings(self, line):
        """
        parses a recording start/end line from the EDF, returns a dictionary that will be a line in the DataFrame
        """
        pass

    @abstractmethod
    def is_sample(self, line):
        """
        checks if a line is a sample line
        :param line: line to check
        :return: True if line is sample, else False
        """
        pass
