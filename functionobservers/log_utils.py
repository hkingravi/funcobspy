"""
Utils used across the repository.
"""

import logging
from logging.handlers import SysLogHandler

# set up logger
LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: [%(processName)s:%(process)d] %(message)s'


def configure_logger(level='INFO', name=None):
    """
    This function configures the logger for the entire repo.

    :param level: logging level string: one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
    :param name: string to be passed to the logging.getLogger method.
    :return: logger object
    """
    level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING':logging.WARNING,
                 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}
    level = level.upper()
    if level not in level_map:
        print("ERROR: Invalid value {} for the logging level.".format(level))
        level = 'INFO'

    logging.basicConfig(level=level_map[level], format=LOG_FORMAT)  # perform basic configuration
    if isinstance(name, basestring):
        logger_out = logging.getLogger(name)
    else:
        logger_out = logging.getLogger(__name__)

    need_sys = True
    for handler in logger_out.handlers:
        if isinstance(handler, SysLogHandler):
            need_sys = False
            break

    if need_sys:
        sh = SysLogHandler(address='/dev/log', facility=SysLogHandler.LOG_LOCAL0)
        sh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger_out.addHandler(sh)

    return logger_out
