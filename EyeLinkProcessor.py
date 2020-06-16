from os.path import exists, basename
from re import split
from sys import stderr

import numpy as np
import pandas as pd

from EyeTrackingParsers import *


class Eye(Enum):
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class EyeLinkProcessor:
    """
    This class loads & parses Eyelink 1000 .asc data file.
    This class assumes the recording is binocular without velocity
    This class can extract ET events from the samples and synchronize to mne.Raw file by triggers
    """
    NOISE_THRESHOLD_LAMBDA = 5

    def __init__(self, filename, parser_type):
        """
        initialize a parser
        :param filename: path of the .asc file to parse
        """
        self.name = basename(filename)[:-4]
        self._filename = filename
        self._samples = []
        self._messages = []
        self._triggers = []
        self._fixations = []
        self._saccades = []
        self._blinks = []
        self._recording_data = []
        self._parser = EyeTrackingParserFactory(parser_type)
        # method dictionary for parsing
        self._parse_line_by_token = {'INPUT': self._parse_input, 'MSG': self._parse_msg, 'ESACC': self._parse_saccade,
                                     'EFIX': self._parse_fixation, "EBLINK": self._parse_blinks}
        self._parse_et_data()  # parse the file
        self._dt = self._samples.loc[1, self._parser.TIME] - self._samples.loc[0, self._parser.TIME]
        self._velocity_extracted_saccades = self._detect_saccades()

    def _parse_sample(self, line):
        """
        parses a sample line from the EDF
        """
        self._samples.append(self._parser.parse_sample(line))

    def _parse_msg(self, line):
        """
        parses a message line from the EDF
        """
        self._messages.append(self._parser.parse_msg(line))

    def _parse_input(self, line):
        """
        parses a trigger line from the EDF
        """
        self._triggers.append(self._parser.parse_input(line))

    def _parse_fixation(self, line):
        """
        parses a fixation line from the EDF
        """
        self._fixations.append(self._parser.parse_fixation(line))

    def _parse_saccade(self, line):
        """
        parses a saccade line from the EDF
        """
        self._saccades.append(self._parser.parse_saccade(line))

    def _parse_blinks(self, line):
        """
        parses a blink line from the EDF
        """
        self._blinks.append(self._parser.parse_blinks(line))

    def _parse_recordings(self, line):
        """
        parses a recording start/end line from the EDF
        """
        self._recording_data.append(self._parser.parse_recordings(line))

    def _parse_et_data(self):
        """
        parses the .asc file whose path is self._filename
        """
        if not exists(self._filename):
            raise Exception("ET File does not exist!")
        if self._filename[-4:] != ".asc":
            raise Exception("given file is not an .asc file!")
        with open(self._filename, 'r') as file:
            all_lines = file.readlines()
        for line in all_lines:
            line = split("[ \n\t]+", line)
            if self._parser.is_sample(line):
                line = [-1 if i == '.' else i for i in line]
                self._parse_sample(line)
            elif line[0] in self._parser.parse_line_by_token:
                line = [-1 if i == '.' else i for i in line]
                self._parse_line_by_token[line[0]](line)
        self._samples = pd.DataFrame(self._samples)
        self._messages = pd.DataFrame(self._messages)
        self._triggers = pd.DataFrame(self._triggers)
        self._fixations = pd.DataFrame(self._fixations)
        self._saccades = pd.DataFrame(self._saccades)
        self._blinks = pd.DataFrame(self._blinks)
        self._recording_data = pd.DataFrame(self._recording_data)

    def get_eye_locations(self, eye=Eye.BOTH):
        """
        returns the samples of the given eye
        :param eye: the eye to retrieve data for
        :type eye: Eye
        :return: pandas DataFrame containing time, x & y positions and pupil size per timepoint for given eye
        """
        if eye == Eye.RIGHT:
            return self._samples[["time"] + [col for col in self._samples.columns if "right" in col]]
        elif eye == Eye.LEFT:
            return self._samples[["time"] + [col for col in self._samples.columns if "left" in col]]
        elif eye == Eye.BOTH:
            return self._samples
        else:
            print("%d Not supported. Please choose one of the following:\n" % eye +
                  "".join(["Eye.%s ," % e.name for e in Eye]), file=stderr)

    @staticmethod
    def _get_velocities(x, y, kernel):
        vx = np.convolve(x, kernel, 'valid')
        vy = np.convolve(y, kernel, 'valid')
        return vx, vy

    @staticmethod
    def _get_sigmas(vx, vy):
        sigma_x = np.median(np.power(vx, 2)) - np.power(np.median(vx), 2)
        sigma_y = np.median(np.power(vy, 2)) - np.power(np.median(vy), 2)
        return sigma_x, sigma_y

    def _detect_saccades(self):
        """
        detect saccades based on velocity
        refs:
            Engbert & Kliegl 2002 https://doi.org/10.1016/S0042-6989(03)00084-1
            Engbert & Mergenthaler 2006 https://doi.org/10.1073/pnas.0509557103

        """
        velocity_transform_kernel = (1 / (6 * self._dt)) * np.array([-1, -1, 0, 1, 1])
        velocities, sigmas, thresholds = [], [], []
        if self._parser.get_type() == Eye.BOTH or self._parser.get_type() == Eye.LEFT:
            v_left_x, v_left_y = EyeLinkProcessor._get_velocities(self._samples[self._parser.LEFT_X],
                                                                  self._samples[self._parser.LEFT_Y],
                                                                  velocity_transform_kernel)
            velocities.extend((v_left_x, v_left_y))
            sigma_left_x, sigma_left_y = EyeLinkProcessor._get_sigmas(v_left_x, v_left_y)
            sigmas.extend((sigma_left_x, sigma_left_y))
            thresholds.extend((EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_left_x,
                               EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_left_y))
        if self._parser.get_type() == Eye.BOTH or self._parser.get_type() == Eye.RIGHT:
            v_right_x, v_right_y = EyeLinkProcessor._get_velocities(self._samples[self._parser.RIGHT_X],
                                                                    self._samples[self._parser.RIGHT_Y],
                                                                    velocity_transform_kernel)
            velocities.extend((v_right_x, v_right_y))
            sigma_right_x, sigma_right_y = EyeLinkProcessor._get_sigmas(v_right_x, v_right_y)
            sigmas.extend((sigma_right_x, sigma_right_y))
            thresholds.extend((EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_right_x,
                               EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_right_y))

        velocities = np.asarray(velocities)
        sigmas = np.asarray(sigmas)
        thresholds = np.asarray(thresholds)
        return velocities
