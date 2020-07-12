# In order to add parsers, implement a new class like MonocularNoVelocityParser
from BaseETParser import BaseETParser
from EyeEnum import Eye


class BinocularNoVelocityParser(BaseETParser):
    """
    parser for EyeLink 1000 Monocular recording with no velocity data
    """

    # constants for column names, allow for quick and easy changes
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

    @staticmethod
    def parse_sample(line):
        """
        parses a sample line from the EDF
        """
        return {BinocularNoVelocityParser.TIME: int(line[0]), BinocularNoVelocityParser.LEFT_X: float(line[1]),
                BinocularNoVelocityParser.LEFT_Y: float(line[2]),
                BinocularNoVelocityParser.LEFT_PUPIL_SIZE: float(line[3]),
                BinocularNoVelocityParser.RIGHT_X: float(line[4]),
                BinocularNoVelocityParser.RIGHT_Y: float(line[5]),
                BinocularNoVelocityParser.RIGHT_PUPIL_SIZE: float(line[6])}

    @staticmethod
    def parse_msg(line):
        """
        parses a message line from the EDF
        """
        return {BinocularNoVelocityParser.TIME: int(line[1]), BinocularNoVelocityParser.MSG: "".join(line[2:-1])}

    @staticmethod
    def parse_input(line):
        """
        parses a trigger line from the EDF
        """
        return {BinocularNoVelocityParser.TIME: int(line[1]), BinocularNoVelocityParser.TRIGGER: int(line[2])}

    @staticmethod
    def parse_fixation(line):
        """
        parses a fixation line from the EDF
        """
        return {BinocularNoVelocityParser.EYE_STR: line[1], BinocularNoVelocityParser.START_TIME: int(line[2]),
                BinocularNoVelocityParser.END_TIME: int(line[3]),
                BinocularNoVelocityParser.DURATION: int(line[4]), BinocularNoVelocityParser.AVG_X: float(line[5]),
                BinocularNoVelocityParser.AVG_Y: float(line[6]),
                BinocularNoVelocityParser.AVG_PUPIL_SIZE: float(line[6])}

    @staticmethod
    def parse_saccade(line):
        """
        parses a saccade line from the EDF
        """
        return {BinocularNoVelocityParser.EYE_STR: line[1], BinocularNoVelocityParser.START_TIME: int(line[2]),
                BinocularNoVelocityParser.END_TIME: int(line[3]), BinocularNoVelocityParser.DURATION: int(line[4]),
                BinocularNoVelocityParser.START_X: float(line[5]), BinocularNoVelocityParser.START_Y: float(line[6]),
                BinocularNoVelocityParser.END_X: float(line[6]), BinocularNoVelocityParser.END_Y: float(line[7]),
                BinocularNoVelocityParser.AMPLITUDE: float(line[8]),
                BinocularNoVelocityParser.PEAK_VELOCITY: float(line[9])}

    @staticmethod
    def parse_blinks(line):
        """
        parses a blink line from the EDF
        """
        return {BinocularNoVelocityParser.EYE_STR: line[1], BinocularNoVelocityParser.START_TIME: int(line[2]),
                BinocularNoVelocityParser.END_TIME: int(line[3]),
                BinocularNoVelocityParser.DURATION: int(line[4])}

    @staticmethod
    def parse_recordings(line):
        """
        parses a recording start/end line from the EDF
        """
        return {BinocularNoVelocityParser.TYPE: line[0], BinocularNoVelocityParser.TIME: int(line[1])}

    @staticmethod
    def is_sample(line):
        return line[-2] == '.....'

    @staticmethod
    def get_type():
        return Eye.BOTH

    # has to be last in order to find the parsing methods
    parse_line_by_token = {'INPUT': parse_input, 'MSG': parse_msg, 'ESACC': parse_saccade, 'EFIX': parse_fixation,
                           "EBLINK": parse_blinks}
