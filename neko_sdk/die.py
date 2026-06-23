import os
import signal
import sys
import psutil


def die(exitval=0):
    """
    Kills all subprocesses and Python parents with SIGKILL,
    then exits itself with the specified exitval.
    """
    current_process = psutil.Process(os.getpid())

    # 1. Collect all subprocesses (children)
    children = current_process.children(recursive=True)

    # 2. Trace back Python parents only (stopping at PyCharm/Shell)
    python_parents = []
    temp_parent = current_process.parent()
    while temp_parent is not None:
        try:
            # Only target parents if they are python interpreters
            if "python" in temp_parent.name().lower():
                python_parents.append(temp_parent)
                temp_parent = temp_parent.parent()
            else:
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break

    # 3. Kill children and parents first
    for proc in children + python_parents:
        try:
            proc.send_signal(signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 4. Exit the current process with the provided exit value
    # We use os._exit to bypass any catch-all try/except or finally blocks
    os._exit(exitval)