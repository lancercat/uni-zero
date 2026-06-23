import sys
import time


def err(text):
    print(text);
def warn(*args):
    print("WARN:",*[str(i) for i in args]);
def info(*args):
    print("INFO:",*[str(i) for i in args]);

def fatal(text=""):
    print("FATAL:",text);
    sys.stdout.flush();
    sys.stderr.flush();
    time.sleep(1);
    exit(9);

