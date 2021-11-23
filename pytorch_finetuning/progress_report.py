from datetime import datetime as dt
from datetime import timedelta

from tqdm.auto import tqdm


def get_progress_reporter(progress_reporting, logger):
    if progress_reporting is None:
        return no_progress_report
    elif progress_reporting == "tqdm":
        return tqdm
    elif progress_reporting == "eta":
        return ETA(logger)
    raise ValueError('progress_reporting should be either None, "tqdm" or "eta"')


def no_progress_report(iterable, *args, **kwargs):
    return iterable


class ETA:
    """Estimate and log duration of a loop

    Wraps an iterable similarly to tqdm, but doesn't provide a progress bar,
    just logs estimated time of arrival. Tries to do so not too often and only
    if the estimate has changed since last report.

    :param logger: logging.Logger
    :param iterable: iterable (if without len(), no ETA will be logged)
    :param desc: str optional description of the loop (e.g. "epoch")
    :return: iterable

    Usage:
    for i in ETA(logger, range(num_epochs), desc="training"):
        train()
    """

    def __init__(self, logger, iterable=None, desc=""):
        self.logger = logger
        self.iterable = iterable
        self.desc = desc

    def __call__(self, iterable, desc=""):
        return ETA(self.logger, iterable, desc)

    def __iter__(self):
        try:
            len(self.iterable)
        except Exception:
            # we cannot estimate duration
            return iter(self.iterable)

        self.i = 0
        self.start_time = dt.now()
        self.next_estimation = 1
        self.next_min_pause = timedelta(seconds=5)
        self.last_estimation_time = self.start_time
        self.last_eta = dt.now() - timedelta(days=1)

        self.iterator = iter(self.iterable)
        return self

    def get_remaining_time(self):
        step_duration = (dt.now() - self.start_time) / self.i
        remaining_steps = len(self.iterable) - self.i
        return remaining_steps * step_duration

    def __next__(self):
        if self.i >= self.next_estimation:
            now = dt.now()
            if now - self.last_estimation_time > self.next_min_pause:
                remaining = self.get_remaining_time()
                eta = now + remaining
                if abs(eta - self.last_eta) > timedelta(seconds=5 * 60):
                    self.logger.info(
                        "ETA {} {} ({})".format(
                            self.desc,
                            eta.replace(microsecond=0),
                            timedelta(seconds=round(remaining.total_seconds())),
                        )
                    )
                    self.last_eta = eta
                self.next_estimation *= 2
                self.next_min_pause *= 2
                self.last_estimation_time = now
        self.i += 1
        return next(self.iterator)
