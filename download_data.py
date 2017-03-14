"""Download data relevant to train the KittiSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
import subprocess

import zipfile


from six.moves import urllib
from shutil import copy2

import argparse

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(1, 'incl')

# Please set kitti_data_url to the download link for the Kitti DATA.
#
# You can obtain by going to this website:
# http://www.cvlibs.net/download.php?file=data_road.zip
#
# Replace 'http://kitti.is.tue.mpg.de/kitti/?????????.???' by the
# correct URL.


vgg_url = 'ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy'

copyfiles = ["train_2.idl", "train_3.idl", "train_4.idl",
             "val_2.idl", "val_3.idl", "val_4.idl",
             "train.txt", "val.txt", "testing.txt", "train3.txt", "val3.txt"]

copydirs = ["KittiBox", "KittiBox", "KittiBox",
            "KittiBox", "KittiBox", "KittiBox",
            "KittiBox", "KittiBox", "data_road", "data_road", "data_road"]

kitti_download_files = ["data_object_image_2.zip", "data_object_label_2.zip",
                        "data_road.zip"]

data_sub_dirs = ["KittiBox", "KittiBox", ""]


def get_pathes():
    """
    Get location of `data_dir` and `run_dir'.

    Defaut is ./DATA and ./RUNS.
    Alternativly they can be set by the environoment variabels
    'TV_DIR_DATA' and 'TV_DIR_RUNS'.
    """

    if 'TV_DIR_DATA' in os.environ:
        data_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        data_dir = "DATA"

    if 'TV_DIR_RUNS' in os.environ:
        run_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        run_dir = "RUNS"

    return data_dir, run_dir


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    logging.info("   Download URL: {}".format(url))
    logging.info("   Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
                prog = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, prog))
                sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_url', default='', type=str)
    args = parser.parse_args()

    kitti_data_url = args.kitti_url

    data_dir, run_dir = get_pathes()

    if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    vgg_weights = os.path.join(data_dir, 'vgg16.npy')

    # Download VGG DATA
    if not os.path.exists(vgg_weights):
        download_command = "wget {} -P {}".format(vgg_url, data_dir)
        logging.info("Downloading VGG weights.")
        download(vgg_url, data_dir)
    else:
        logging.warning("File: {} exists.".format(vgg_weights))
        logging.warning("Please delete to redownload VGG weights.")

    # Checking whether the user provides an URL
    if kitti_data_url == '':
        logging.error("Data URL for Kitti Data not provided.")
        url = "http://www.cvlibs.net/download.php?file=data_road.zip"
        logging.error("Please visit: {}".format(url))
        logging.error("and request Kitti Download link.")
        logging.error("Rerun scrpt using"
                      "'python download_data.py --kitti_url [url]'")
        exit(1)

    # Copy txt files
    for file, subdir in zip(copyfiles, copydirs):
        filename = os.path.join('data', file)
        target_dir = os.path.join(data_dir, subdir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        copy2(filename, target_dir)
        logging.info("Copied {} to {}.".format(filename, target_dir))

    # Checking whether the user provides the correct URL
    if not kitti_data_url[13:33] == 'is.tue.mpg.de/kitti/':
        logging.error("Wrong url.")
        url = "http://www.cvlibs.net/download.php?file=data_road.zip"
        logging.error("Please visit: {}".format(url))
        logging.error("and request Kitti Download link.")
        logging.error("You will receive an Email with the kitti download url")
        logging.error("Rerun and enter the received [url] using"
                      "'python download_data.py --kitti_url [url]'")
        exit(1)

    for zip_file, subdir in zip(kitti_download_files, data_sub_dirs):
        # Get name of downloaded zip file
        file_dir = os.path.join(data_dir, subdir)
        # Creating dirs if nessasary
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        final_file = os.path.join(file_dir, zip_file)
        # Check whether file exists
        if os.path.exists(final_file):
            logging.info("File exists: {}".format(final_file))
            logging.info("Skipping Download and extraction.")
            logging.info("Remove: {} if you wish to download that data again."
                         .format(final_file))
            logging.info()
            continue
        # Make Kitti_URL
        kitti_main = os.path.dirname(kitti_data_url)
        kitti_zip_url = os.path.join(kitti_main, zip_file)
        kitti_zip_url = os.path.join(kitti_main,
                                     os.path.basename(zip_file))
        logging.info("Starting to download: {}".format(zip_file))
        download(kitti_zip_url, file_dir)
        logging.info("Extracting: {}".format(final_file))
        zipfile.ZipFile(final_file, 'r').extractall(file_dir)

    logging.info("All data have been downloaded successful.")


if __name__ == '__main__':
    main()
