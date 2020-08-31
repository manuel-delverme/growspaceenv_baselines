import os, sys
import imageio
import cv2

def make_movie_list():
    png_dir = './hittiyas/growspaceenv_braselines/scripts/imgs/'

    step = 0
    movie_files = []
    print(os.listdir(png_dir))
    for i in range(0, len(os.listdir(png_dir))):
        if os.listdir(png_dir)[i].startswith("step"):
            movie_files.append(os.listdir(png_dir)[i])


    return movie_files

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]