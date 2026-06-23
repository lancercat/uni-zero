import os
def get_dev_meta(branch="neko_2022_soai_zero"):
    return {
    "MEOWS-MegaDimension":
    {
        "ltype":"dash",
        "name": "MEOWS-MegaDimension",
        "username": "lasercat",
        "root": os.path.join("/home/lasercat/cat/neko_wcki/", branch),
        "port": "59000",
        "addr": "md.meows"

    },
    "318prirC":
    {
        "ltype": "solid",
        "name":"318prirC",
        "username": "prir1005",

        "root":os.path.join("/home/prir1005/cat/neko_wcki/", branch),
        # "port":"22",
        # "addr":"202.204.62.148",
        # "port": "9379",
        "port": "9379",
        "addr": "localhost"
    },
    "MEOWS-HeartDimension":
    {
        "ltype": "dot",
        "name": "MEOWS-HeartDimension",
        "username": "lasercat",
        "root": os.path.join("/home/lasercat/cat/neko_wcki/", branch),
        "port": "9339",
        "addr": "localhost"
    },
    "MEOWS-ZeroDimension":
        {
            "ltype": "dot",
            "name": "MEOWS-ZeroDimension",
            "username": "lasercat",
            "root": os.path.join("/home/lasercat/cat/neko_wcki/", branch),
            "port": "22",
            "addr": "zd.meows",

        },
        "MEOWS-Gamarket2":
            {
                "ltype": "dot",
                "name": "MEOWS-Gamarket2",
                "username": "lasercat",
                "root": os.path.join("/run/media/lasercat/data/neko_wcki/", branch),
                # "port": "22",
                # "addr": "zd.meows",
                "port": "22",
                "addr": "localhost"
            }
}
