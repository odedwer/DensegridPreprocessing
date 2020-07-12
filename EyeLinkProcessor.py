from os.path import exists, basename
from re import split
from sys import stderr

import numpy as np
import pandas as pd

from EyeEnum import Eye


class EyeLinkProcessor:
    """
    This class loads & parses Eyelink 1000 .asc data file.
    This class assumes the recording is binocular without velocity
    This class can extract ET events from the samples and synchronize to mne.Raw file by triggers
    """
    NOISE_THRESHOLD_LAMBDA = 5
    MISSING_VALUE = -1

    def __init__(self, filename, parser_type, saccade_detector):
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
        self._parser = parser_type.value  # enum, get class from enum value
        # method dictionary for parsing
        self._parse_line_by_token = {'INPUT': self._parse_input, 'MSG': self._parse_msg, 'ESACC': self._parse_saccade,
                                     'EFIX': self._parse_fixation, "EBLINK": self._parse_blinks}
        self._parse_et_data()  # parse the file
        # get sampling frequency
        self._sf = 1000. / self._samples.loc[1, self._parser.TIME] - self._samples.loc[0, self._parser.TIME]
        self._saccade_detector = saccade_detector.value  # enum, get class from enum value
        self._detected_saccades = []
        self._detect_saccades()

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
        # validity checks, raise exception in case of invalid input
        if not exists(self._filename):
            raise Exception("ET File does not exist!")
        if self._filename[-4:] != ".asc":
            raise Exception("given file is not an .asc file!")
        # read all lines
        with open(self._filename, 'r') as file:
            all_lines = file.readlines()
        # parse every line
        for line in all_lines:
            line = split("[ \n\t]+", line)
            # ignore lines that are not samples or have a first token corresponding to one of the tokens in the
            # parsing methods dictionary
            if self._parser.is_sample(line):
                line = [EyeLinkProcessor.MISSING_VALUE if i == '.' else i for i in line]
                self._parse_sample(line)
            elif line[0] in self._parser.parse_line_by_token:
                line = [EyeLinkProcessor.MISSING_VALUE if i == '.' else i for i in line]
                self._parse_line_by_token[line[0]](line)
        # create pandas data frames from the lists of dictionaries
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
            return self._samples[[self._parser.TIME] + [col for col in self._samples.columns if "right" in col]]
        elif eye == Eye.LEFT:
            return self._samples[[self._parser.TIME] + [col for col in self._samples.columns if "left" in col]]
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

    def _add_saccades_to_self(self, x, y):
        velocity_transform_kernel = (1 / (6 * self._dt)) * np.array([-1, -1, 0, 1, 1])
        vx, vy = EyeLinkProcessor._get_velocities(x, y, velocity_transform_kernel)
        sigma_x, sigma_y = EyeLinkProcessor._get_sigmas(vx, vy)
        threshold_x, threshold_y = EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_x, \
                                   EyeLinkProcessor.NOISE_THRESHOLD_LAMBDA * sigma_y
        tmp = (np.power((vx / threshold_x), 2) + np.power((vy / threshold_y), 2)) > 1
        tmp[np.isnan(tmp) | np.isnan(tmp)] = 0
        above_threshold = np.ones((tmp.size - self._k + 1,), dtype=int)
        for i in range(self._k):
            above_threshold = (
                    above_threshold & tmp[i:(-(self._k - i - 1) if i != self._k - 1 else len(tmp))].astype(int))
        l_pad = (len(self._samples[self._parser.LEFT_X]) - above_threshold.size) // 2
        r_pad = (len(self._samples[self._parser.LEFT_X]) - above_threshold.size) - l_pad
        above_threshold = np.pad(above_threshold, ((l_pad, r_pad),))
        self._velocity_extracted_saccades.append(above_threshold)

    def _detect_saccades(self):
        """
        detect saccades based on velocity
        refs:
            Engbert & Kliegl 2002 https://doi.org/10.1016/S0042-6989(03)00084-1
            Engbert & Mergenthaler 2006 https://doi.org/10.1073/pnas.0509557103

        """

        if self._parser.get_type() == Eye.BOTH or self._parser.get_type() == Eye.LEFT:
            left_data = self._samples[[self._parser.LEFT_X, self._parser.LEFT_Y]]
            self._detected_saccades = self._saccade_detector.detect_saccades(left_data, self._sf)
        if self._parser.get_type() == Eye.BOTH or self._parser.get_type() == Eye.RIGHT:
            right_data = self._samples[[self._parser.RIGHT_X, self._parser.RIGHT_Y]]
            stack_function = np.hstack if isinstance(self._detected_saccades, list) else np.vstack
            self._detected_saccades = stack_function(
                [self._detected_saccades, self._saccade_detector.detect_saccades(right_data, self._sf)]).T
