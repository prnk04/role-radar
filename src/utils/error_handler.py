import os
import logging
import datetime


# Setup logging
# logging.basicConfig(
#     level=logging.INFO, format='%(acstime)s - %(levelname)s - %(message)s'
# )

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger(__name__)


def log_error(error_message):
    dir_name = "logs"
    os.makedirs(dir_name, exist_ok=True)
    file_name = datetime.datetime.today().date()
    with open(f"{dir_name}/error_logs_{file_name}.txt", "a") as f:
        f.writelines(
            f"{datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\t{error_message}\n"
        )
