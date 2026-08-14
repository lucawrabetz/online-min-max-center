import os
import re
import shutil
import logging
import numpy as np
from gurobipy import Model
from util import append_date, DATETIME_FORMAT


def setup_logging():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    basename = "onlinekcenter"
    filename = append_date(basename, time=True) + ".log"
    filepath = os.path.join(logs_dir, filename)
    terminal_format = "[%(levelname)s]: %(message)s"
    file_format = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format=file_format,
        datefmt=DATETIME_FORMAT,
    )

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter for the console handler
    console_formatter = logging.Formatter(terminal_format)
    console_handler.setFormatter(console_formatter)

    # Get the root logger and add the console handler to it
    logging.getLogger().addHandler(console_handler)


class InfoLogger:
    INFO_CHARLEN: int = len("[INFO]: ")
    DEBUG_CHARLEN: int = len("[DEBUG]: ")
    WARN_CHARLEN: int = len("[WARNING]: ")
    SPACE_INDENT: int = 4
    FINAL_INDENT_CHARLEN: int = max(INFO_CHARLEN, DEBUG_CHARLEN) + SPACE_INDENT
    INFO_INDENT: str = f"{' ' * (FINAL_INDENT_CHARLEN - INFO_CHARLEN)}-> "
    DEBUG_INDENT: str = f"{' ' * (FINAL_INDENT_CHARLEN - DEBUG_CHARLEN)}>>>> "
    WARN_INDENT: str = f"{' ' * (FINAL_INDENT_CHARLEN - WARN_CHARLEN)}>>>> "

    def __init__(self, logger=logging, special_char="-", precision=2):
        self.logger = logger
        self.special_char = special_char
        self.precision = precision

    def blank_line(self):
        print()

    def separator_line(self):
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        num_characters = terminal_width // 2
        separator_line = (
            f"{self.special_char} " * num_characters + f"{self.special_char}"
        )
        print(separator_line)

    def separator_block(self):
        self.separator_line()
        self.separator_line()

    def clear_page(self):
        for i in range(4):
            self.blank_line()

    def format_numbers_in_string(self, msg: str) -> str:
        if type(msg) != str:
            logging.warning(
                f"Expected string in format_numbers_in_string, got {type(msg)}. Cannot guarantee proper formatting."
            )
            return msg
        pattern = re.compile(r"(\d+\.\d+)")

        def repl(match):
            return f"{float(match.group()):.{self.precision}f}"

        return pattern.sub(repl, msg)

    def log_matrix(self, matrix: np.ndarray, name: str, index: int = 0):
        """
        Prints a 2D numpy ndarray in a readable format with specified decimal places.

        Parameters:
        matrix (np.ndarray): 2D numpy array to be printed.
        decimal_places (int): Number of decimal places for formatting the numbers.
        """
        if not isinstance(matrix, np.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if matrix.ndim != 2:
            raise ValueError("Input array must be 2-dimensional.")
        self.log_debug(f"{name} matrix", ":")
        for row in matrix:
            formatted_row = []
            for x in row:
                formatted_row.append(str(x))
            self.log_debug(" ".join(formatted_row), "")

    def log_header(self, msg: str):
        self.blank_line()
        self.separator_block()
        msg = self.format_numbers_in_string(msg)
        logging.info(f"{msg}.")
        self.separator_block()
        self.blank_line()

    def log_subheader(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        self.blank_line()
        logging.info(f"{msg}.")
        self.blank_line()

    def log_bodydebug(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.debug(f"{self.INFO_INDENT}{msg}.")

    def log_body(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.info(f"{self.INFO_INDENT}{msg}.")

    def log_special(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.info(f"--> {msg}! <--")

    def log_debug(self, msg: str, final_punc: str = "."):
        msg = self.format_numbers_in_string(msg)
        logging.debug(f"{self.DEBUG_INDENT}{msg}{final_punc}")

    def log_warning(self, msg: str, final_punc: str = "."):
        msg = self.format_numbers_in_string(msg)
        logging.warn(f"{self.WARN_INDENT}{msg}{final_punc}")


_LOGGER = InfoLogger()


def throwaway_gurobi_model():
    """
    Create an empty gurobi model and destroy it.
    This is a hacky workaround to get rid of the dreaded "academic license", and "set parameter username" printouts from gurobi which also print twice in this case because of the log config. These only print the first time gurobi is called in memory. Just call this function from any class that runs a series of gurobi models in the __init__ method to ensure your logging works as expected and is readable.
    """
    model = Model("fake")
    del model


def gurobi_log_file():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    basename = "onlinekcenter_gurobi"
    filename = append_date(basename, time=True) + ".log"
    filepath = os.path.join(logs_dir, filename)
    return filepath
