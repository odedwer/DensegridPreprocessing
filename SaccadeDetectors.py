import numpy as np

from BaseSaccadeDetector import BaseSaccadeDetector

NUM_OF_COLUMS_ERROR_MSG = "Saccade data should have 2 columns - X, Y positions, but has %d columns"

INPUT_SHAPE_ERROR_MSG = "Shape of saccade data is incorrect! should be a 2D matrix yet has %d dimensions"


class EngbertAndMergenthalerMicrosaccadeDetector(BaseSaccadeDetector):
    NOISE_THRESHOLD_LAMBDA = 5

    @classmethod
    def detect_saccades(cls, saccade_data, sf):
        # validate input
        saccade_data = np.asarray(saccade_data)
        if len(saccade_data.shape) != 2:
            raise Exception(INPUT_SHAPE_ERROR_MSG % len(saccade_data.shape))
        if saccade_data.shape[1] != 2:
            raise Exception(NUM_OF_COLUMS_ERROR_MSG % saccade_data.shape[1])
        velocities = cls._get_velocities(saccade_data, sf)
        msdx, msdy = cls._get_SD_thresholds(velocities)
        radius_x = cls.NOISE_THRESHOLD_LAMBDA * msdx
        radius_y = cls.NOISE_THRESHOLD_LAMBDA * msdy
        # threshold the data
        thresholded_data = (velocities[:, 0] / radius_x) ** 2 + (velocities[:, 1] / radius_y) ** 2
        thresholded_data = thresholded_data > 1

        # calculate row numbers of saccade starts
        saccade_start_indices = cls._get_saccade_start_indices(thresholded_data)
        event_vector = np.zeros((saccade_data.shape[0],), dtype=int)
        event_vector[saccade_start_indices] = 1  # 1 in every saccade start, 0 elsewhere
        return event_vector

    @classmethod
    def _get_saccade_start_indices(cls, thresholded_data):
        possible_saccade_indices = np.nonzero(thresholded_data)[0]  # all non-zero indices
        consecutive_saccade_indices = np.diff(possible_saccade_indices) == 1  # only consecutive indices
        # only indices in which 2 previous indices are detected saccades
        detected_saccades_indices = np.insert((consecutive_saccade_indices[:-1:] & consecutive_saccade_indices[1::]), 0,
                                              [False, False])
        # get run length of 0 and 1's (for [1,1,1,0,0,1,1,1,1] we'll get [3,4] and [2])
        non_zero = np.nonzero(np.diff(detected_saccades_indices) != 0)[0]
        diff = np.diff(non_zero)
        no_saccade_run_lengths = diff[1::2]
        saccade_run_lengths = diff[::2]
        first_saccade_index = np.argmax(detected_saccades_indices)  # index of first 1
        saccade_starts = np.cumsum(
            np.hstack([[first_saccade_index], saccade_run_lengths[:-1] + no_saccade_run_lengths]))
        saccade_start_indices = possible_saccade_indices[saccade_starts]
        return saccade_start_indices

    @staticmethod
    def _get_velocities(saccade_data, sf):
        velocities = np.zeros((saccade_data.shape[0], 2))
        velocities[3:-2, :] = (sf / 6.) * (
                saccade_data[5:, :] + saccade_data[4:-1, :] - saccade_data[2:-3, :] - saccade_data[1:-4, :])
        velocities[2, :] = (sf / 2.) * (saccade_data[3, :] - saccade_data[1, :])
        velocities[-2, :] = (sf / 2.) * (saccade_data[-1, :] - saccade_data[3, :])
        return velocities

    @staticmethod
    def _get_SD_thresholds(velocities):
        msdx = np.sqrt((np.median(velocities[:, 0] ** 2)) - (np.median(velocities[:, 0]) ** 2))
        msdy = np.sqrt((np.median(velocities[:, 1] ** 2)) - (np.median(velocities[:, 1]) ** 2))
        return msdx, msdy
