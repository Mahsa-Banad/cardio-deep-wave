import wfdb

def download_mitdb():
    wfdb.dl_database(
        "mitdb",
        dl_dir="data/mit-bih"
    )
    print(" Dataset downloaded")