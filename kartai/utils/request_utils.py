

import time
import requests


def do_request(url, params, stream=True, headers=None, timeout=None):
    req = None
    wait = 1
    number_of_retries = 0
    while number_of_retries < 15:
        try:
            req = requests.get(url, stream=stream, params=params,
                               headers=headers, timeout=timeout)
        except Exception:
            print("Error in request - try again. Retry number: ", number_of_retries)

        if not req:
            time.sleep(wait)
            wait = increase_wait_time(wait, number_of_retries)
        else:
            if req.status_code == 200:
                return req
            else:
                print("Error in req – status code: ", req.status_code)
                time.sleep(wait)
                wait = increase_wait_time(wait, number_of_retries)
        number_of_retries += 1

    raise ValueError(
        "Something went very wrong in request, even after 15 retries")


def increase_wait_time(wait, number_of_retries):
    wait += 1 if number_of_retries < 10 else 30
    return wait
