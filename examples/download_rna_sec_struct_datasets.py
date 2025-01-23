from rinalmo.utils.download import download_spot_rna_bprna, download_archiveII_fam_splits
from pathlib import Path


data_root = Path('../data')


TRAIN_DIR_NAME = "train"
VAL_DIR_NAME = "valid"
TEST_DIR_NAME = "test"


download_spot_rna_bprna(
                data_root/"bpRNA", train_dir_name=TRAIN_DIR_NAME,
                val_dir_name=VAL_DIR_NAME, test_dir_name=TEST_DIR_NAME
            )


