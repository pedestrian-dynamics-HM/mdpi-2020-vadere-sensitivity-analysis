import logging
""" ------------------------------------- init logging -------------------------------------------------"""


def init_logging(log_file_path):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.NOTSET,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path + "\debug_file.log",
                        filemode='w')

    #  set level to warning for existing loggers
    [logging.getLogger(name).setLevel(logging.WARNING) for name in logging.root.manager.loggerDict]
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    #  define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger("utils.init_logging")
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("fabric").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info("Started logging.")


def finish_logging():
    logging.shutdown()

