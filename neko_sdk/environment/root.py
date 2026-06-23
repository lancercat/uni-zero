import getpass;
import os;
def find_data_root():
    username = getpass.getuser()
    if(username=="lasercat"):
        print("own PC?")
        return os.path.join("/home",username,"ssddata");
    else:
        print("lab pc!")
        return os.path.join("/home",username,"ssddata");

def find_model_root(root_override=None):
    if(root_override is not None):
        return root_override;
    username = getpass.getuser()

    os.path.join("/home", username, "hydra_saves");

def find_export_root(root_override=None):
    if(root_override is not None):
        return root_override;
    username = getpass.getuser()
    if (username == "lasercat"):
        print("own PC?")
        return os.path.join("/home", username, "ssddata");
    else:
        print("lab pc!")
        return os.path.join("/home", username, "ssddata");
