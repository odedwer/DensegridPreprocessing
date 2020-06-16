# In order to add parsers, implement a new class like MonocularNoVelocityParser, singleton that overrides
# BaseETParser functions according to the data file format
from enum import Enum

import EyeLinkProcessor
from BaseETParser import BaseETParser

# add entries for added parsers

Parsers = Enum('Parsers', 'MonocularNoVelocity')


def EyeTrackingParserFactory(parser_type: Parsers):
    """
    Factory for ET parsers
    :param parser_type: of type Parsers. Type of parser to return instance of
    """
    if parser_type == Parsers.MonocularNoVelocity:
        return MonocularNoVelocityParser()
        # add elif to extend to other parsers
    else:
        raise Exception('No support for ' + str(parser_type))


class MonocularNoVelocityParser:
    """
    singleton wrapper of parser
    """

    class __MonocularNoVelocityParser(BaseETParser):
        """
        actual parser for EyeLink 1000 Monocular recording with no velocity data
        """
        TYPE = "type"
        PEAK_VELOCITY = "peak velocity"
        AMPLITUDE = "amplitude"
        END_Y = "end y"
        END_X = "end x"
        START_Y = "start y"
        START_X = "start x"
        AVG_PUPIL_SIZE = "avg pupil size"
        AVG_Y = "avg y"
        AVG_X = "avg x"
        DURATION = "duration"
        END_TIME = "end time"
        START_TIME = "start time"
        EYE_STR = "eye"
        TRIGGER = "trigger"
        MSG = "message"
        RIGHT_PUPIL_SIZE = "right pupil size"
        RIGHT_Y = "right y"
        RIGHT_X = "right x"
        LEFT_PUPIL_SIZE = "left pupil size"
        LEFT_Y = "left y"
        LEFT_X = "left x"
        TIME = "time"

        def __init__(self):
            self.parse_line_by_token = {'INPUT': self.parse_input, 'MSG': self.parse_msg,
                                        'ESACC': self.parse_saccade,
                                        'EFIX': self.parse_fixation, "EBLINK": self.parse_blinks}

        def parse_sample(self, line):
            """
            parses a sample line from the EDF
            """
            return {self.TIME: int(line[0]), self.LEFT_X: float(line[1]), self.LEFT_Y: float(line[2]),
                    self.LEFT_PUPIL_SIZE: float(line[3]), self.RIGHT_X: float(line[4]),
                    self.RIGHT_Y: float(line[5]), self.RIGHT_PUPIL_SIZE: float(line[6])}

        def parse_msg(self, line):
            """
            parses a message line from the EDF
            """
            return {self.TIME: int(line[1]), self.MSG: "".join(line[2:-1])}

        def parse_input(self, line):
            """
            parses a trigger line from the EDF
            """
            return {self.TIME: int(line[1]), self.TRIGGER: int(line[2])}

        def parse_fixation(self, line):
            """
            parses a fixation line from the EDF
            """
            return {self.EYE_STR: line[1], self.START_TIME: int(line[2]), self.END_TIME: int(line[3]),
                    self.DURATION: int(line[4]), self.AVG_X: float(line[5]), self.AVG_Y: float(line[6]),
                    self.AVG_PUPIL_SIZE: float(line[6])}

        def parse_saccade(self, line):
            """
            parses a saccade line from the EDF
            """
            return {self.EYE_STR: line[1], self.START_TIME: int(line[2]), self.END_TIME: int(line[3]),
                    self.DURATION: int(line[4]), self.START_X: float(line[5]), self.START_Y: float(line[6]),
                    self.END_X: float(line[6]), self.END_Y: float(line[7]), self.AMPLITUDE: float(line[8]),
                    self.PEAK_VELOCITY: float(line[9])}

        def parse_blinks(self, line):
            """
            parses a blink line from the EDF
            """
            return {self.EYE_STR: line[1], self.START_TIME: int(line[2]), self.END_TIME: int(line[3]),
                    self.DURATION: int(line[4])}

        def parse_recordings(self, line):
            """
            parses a recording start/end line from the EDF
            """
            return {self.TYPE: line[0], self.TIME: int(line[1])}

        def is_sample(self, line):
            return line[-2] == '.....'

        def get_type(self):
            return EyeLinkProcessor.Eye.BOTH

    instance = None

    def __init__(self):
        """
        create instance if one does not exist yet
        """
        if not MonocularNoVelocityParser.instance:
            MonocularNoVelocityParser.instance = MonocularNoVelocityParser.__MonocularNoVelocityParser()

    def __getattr__(self, name):
        """
        delegation to instance
        """
        return getattr(self.instance, name)
