# In order to add parsers, implement a new class like MonocularNoVelocityParser, singleton that overrides
# BaseETParser functions according to the data file format
from enum import Enum

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

        def __init__(self):
            self.parse_line_by_token = {'INPUT': self.parse_input, 'MSG': self.parse_msg,
                                        'ESACC': self.parse_saccade,
                                        'EFIX': self.parse_fixation, "EBLINK": self.parse_blinks}

        def parse_sample(self, line):
            """
            parses a sample line from the EDF
            """
            return {"time": int(line[0]), "left x": float(line[1]), "left y": float(line[2]),
                    "left pupil size": float(line[3]), "right x": float(line[4]),
                    "right y": float(line[5]), "right pupil size": float(line[6])}

        def parse_msg(self, line):
            """
            parses a message line from the EDF
            """
            return {"time": int(line[1]), "message": "".join(line[2:-1])}

        def parse_input(self, line):
            """
            parses a trigger line from the EDF
            """
            return {"time": int(line[1]), "trigger": int(line[2])}

        def parse_fixation(self, line):
            """
            parses a fixation line from the EDF
            """
            return {"eye": line[1], "start time": int(line[2]), "end time": int(line[3]),
                    "duration": int(line[4]), "avg x": float(line[5]), "avg y": float(line[6]),
                    "avg pupil size": float(line[6])}

        def parse_saccade(self, line):
            """
            parses a saccade line from the EDF
            """
            return {"eye": line[1], "start time": int(line[2]), "end time": int(line[3]),
                    "duration": int(line[4]), "start x": float(line[5]), "start y": float(line[6]),
                    "end x": float(line[6]), "end y": float(line[7]), "amplitude": float(line[8]),
                    "peak velocity": float(line[9])}

        def parse_blinks(self, line):
            """
            parses a blink line from the EDF
            """
            return {"eye": line[1], "start time": int(line[2]), "end time": int(line[3]), "duration": int(line[4])}

        def parse_recordings(self, line):
            """
            parses a recording start/end line from the EDF
            """
            return {"type": line[0], "time": int(line[1])}

        def is_sample(self, line):
            return line[-2] == '.....'

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
