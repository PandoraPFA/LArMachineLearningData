from abc import ABC, abstractmethod

import logging
logger = logging.getLogger("the_logger")

class BaseResultParser(ABC):
    @abstractmethod
    def on_new_trial(self, trial, filepath):
        pass

    @abstractmethod
    def parse(self, filepath):
        pass

class SimpleParser(BaseResultParser):
    def on_new_trial(self, trial, filepath):
        return

    def parse(self, filepath, overtime_factor):
        with open(filepath, "r") as f:
            result = float(f.read().strip("\n"))
        if overtime_factor != 1:
            logger.info(f"Overtime factor {overtime_factor} applied!")
        return result * overtime_factor

class SimpleInverseParser(BaseResultParser):
    def on_new_trial(self, trial, filepath):
        return

    def parse(self, filepath, overtime_factor):
        with open(filepath, "r") as f:
            result = float(f.read().strip("\n"))
        if overtime_factor != 1:
            logger.info(f"Overtime factor {overtime_factor} applied!")
        return (1 - result) * overtime_factor

class TrackShowerParser(BaseResultParser):
    """
    Penalise the track/shower only Rand Index dropping below the value for the default params.
    """
    def __init__(self):
        self.default_shower_result = None
        self.default_track_result = None

    def on_new_trial(self, trial, filepath):
        if "is_default" not in trial.user_attrs or not trial.user_attrs["is_default"]:
            return
        _, self.default_track_result, self.default_shower_result = self._partial_parse(filepath)
        logger.info(
            f"Default track result it {self.default_track_result}, "
            f"default shower result is {self.default_shower_result}"
        )

    def parse(self, filepath, overtime_factor):
        all_result, track_result, shower_result = self._partial_parse(filepath)
        if overtime_factor != 1:
            logger.info(f"Overtime factor {overtime_factor} applied!")
        track_term = min(track_result - self.default_track_result, 0)
        shower_term = min(shower_result - self.default_shower_result, 0)
        logger.info(
            f"all {all_result}, "
            f"track {track_result} ({track_term}), shower {shower_result} ({shower_term})"
        )
        return (all_result + track_term + shower_term) * overtime_factor

    def _partial_parse(self, filepath):
        with open(filepath, "r") as f:
            all_result, track_result, shower_result = [
                float(val) for val in f.read().strip("\n").split(",")
            ]
        return all_result, track_result, shower_result

class TrackPurityCompletenessParser(BaseResultParser):
    """
    Penalise any drops in track purity and completeness as well as a lower all ARI
    """
    def __init__(self, penalty_factor=1.0):
        self.default_track_purity = None
        self.default_track_completeness = None

        self.penalty_factor = penalty_factor

    def on_new_trial(self, trial, filepath):
        if "is_default" not in trial.user_attrs or not trial.user_attrs["is_default"]:
            return
        _, self.default_track_purity, self.default_track_completeness = self._partial_parse(filepath)
        logger.info(
            f"Default track purity is {self.default_track_purity}, "
            f"default track completeness is {self.default_track_completeness}"
        )

    def parse(self, filepath, overtime_factor):
        all_ari, track_purity, track_completeness = self._partial_parse(filepath)
        if overtime_factor != 1:
            logger.info(f"Overtime factor {overtime_factor} applied!")
        track_purity_term = min(track_purity - self.default_track_purity, 0) * self.penalty_factor
        track_completeness_term = min(track_completeness - self.default_track_completeness, 0) * self.penalty_factor
        logger.info(
            f"all ARI {all_ari}, "
            f"track purity {track_purity} ({track_purity_term}), "
            f"track completeness {track_completeness} ({track_completeness_term})"
        )
        return (all_ari + track_purity_term + track_completeness_term) * overtime_factor

    def _partial_parse(self, filepath):
        with open(filepath, "r") as f:
            all_ari, track_purity, track_completeness = [
                float(val) for val in f.read().strip("\n").split(",")
            ]
        return all_ari, track_purity, track_completeness

class TrackARIShowerPurityParser(BaseResultParser):
    """
    Penalise any drops in shower purity as well as a lower track ARI
    """
    def __init__(self):
        self.default_shower_purity = None

    def on_new_trial(self, trial, filepath):
        if "is_default" not in trial.user_attrs or not trial.user_attrs["is_default"]:
            return
        _, self.default_shower_purity = self._partial_parse(filepath)
        logger.info(f"Default shower purity is {self.default_shower_purity}")

    def parse(self, filepath, overtime_factor):
        track_ari, shower_purity = self._partial_parse(filepath)
        if overtime_factor != 1:
            logger.info(f"Overtime factor {overtime_factor} applied!")
        shower_purity_term = min(shower_purity - self.default_shower_purity, 0)
        logger.info(f"track ARI {track_ari}, shower purity {shower_purity} ({shower_purity_term})")
        return (track_ari  + shower_purity_term) * overtime_factor

    def _partial_parse(self, filepath):
        with open(filepath, "r") as f:
            track_ari, shower_purity = [ float(val) for val in f.read().strip("\n").split(",") ]
        return track_ari, shower_purity

result_parsers = {
    "simple" : SimpleParser(),
    "simple_inverse" : SimpleInverseParser(),
    "trackshower" : TrackShowerParser(),
    "track_purity_completeness" : TrackPurityCompletenessParser(),
    "track_purity_completeness_penalty5" : TrackPurityCompletenessParser(penalty_factor=5.0),
    "track_ari_shower_purity" : TrackARIShowerPurityParser()
}
