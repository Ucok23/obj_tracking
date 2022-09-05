import argparse


# construct the argument parse and parse the arguments
def get_cli_input():
    # initalize argiment parser
    ap = argparse.ArgumentParser()

    # add first argument, input video file name
    # default to pedestrians.mp4 file, don't delete this file
    ap.add_argument("-i", "--input", type=str, default="0",
        help="path to (optional) input video file")

    # add second argument, output video file name
    # default to empty string, indicating
    # that we didn't output to file
    ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")

    # add third argument, whether to display frame
    # while processing, default to displaying
    ap.add_argument("-d", "--display", type=int, default=1,
        help="whether or not output frame should be displayed")

    # return dict of parsed args
    return vars(ap.parse_args())