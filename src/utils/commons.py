import functools
import hashlib
import time


def get_hashed(data: str):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def clean_text_list(thisList):
    data_to_send = list()
    for thisData in thisList:
        modified_data = ",".join([x.strip() for x in thisData.split("(")])
        modified_data = ",".join([x.strip() for x in modified_data.split(")")])
        modified_data = ",".join([x.strip() for x in modified_data.split("&")])
        data_to_send.extend(
            [x.strip() for x in modified_data.split(",") if len(x) > 0])

    data_to_send = list([x.strip()
                        for x in data_to_send if len(x.strip()) > 0])

    return data_to_send


def logging_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Inside function {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Finished function {func.__name__} in {end_time - start_time}")
        return result
    return wrapper
