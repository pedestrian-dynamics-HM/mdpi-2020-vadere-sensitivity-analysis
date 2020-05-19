from subprocess import Popen, PIPE, check_output, call
from typing import Union
import sys


# from suqc - general.py

def get_current_uq_state():
    git_commit_hash = check_output(["git", "rev-parse", "HEAD"])
    git_commit_hash = git_commit_hash.decode().strip()

    uncommited_changes = check_output(["git", "status", "--porcelain"])
    uncommited_changes = uncommited_changes.decode()  # is returned as a byte sequence -> decode to string

    if uncommited_changes:
        print("WARNING: THERE ARE UNCOMMITED CHANGED IN THE REPO")
        # print("In order to have a reproducible scenario run you should check if untracked changes in the following "
        #      "files should be commited before: \n")
        # print(uncommited_changes)

    return {"git_hash": git_commit_hash, "uncommited_changes": uncommited_changes}


# todo: move to suq controller when switching to latest version
def get_current_vadere_version(jar_path: str) -> str:
    return run_utils_function_vadere("getVersion", jar_path)


def run_utils_function_vadere(function: str, jar_path: str):
    # from https://stackoverflow.com/questions/1996518/retrieving-the-output-of-subprocess-call
    p = Popen(["java", "-jar", jar_path, "utils", "-m", function], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode
    output_str = output.decode("utf-8")
    function_output = [s for s in str.split(output_str, "\n") if s and "WARNING" not in s]
    if function_output is not None:
        function_output = function_output[0]
    return function_output


# todo: move to suq controller when switching to latest version
def get_current_vadere_commit(jar_path: str) -> str:
    return run_utils_function_vadere("getCommitHash", jar_path)
