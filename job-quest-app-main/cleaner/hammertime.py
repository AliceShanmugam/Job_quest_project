import time

# Timer Class
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
class Timer():
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError("Timer is still running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise TimeoutError("Timer is not running. Use .stop() to stop it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elpased time : {elapsed_time:0.4f}\n")
