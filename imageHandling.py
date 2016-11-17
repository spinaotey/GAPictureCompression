from PIL import Image
from PIL import ImageDraw
import numpy as np

def alphaComposite(src, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    src = np.asarray(src)
    dst = np.asarray(dst)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    out[rgb] = (src[rgb]*src_a + dst[rgb]*dst_a*(1-src_a))/out[alpha]
    np.seterr(**old_setting)    
    out[alpha] *= 255
    np.clip(out,0,255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype('uint8')
    out = Image.fromarray(out, 'RGBA')
    return out

def makePoly(rgba,coords,size):
    out = Image.new("RGBA",size)
    polydraw = ImageDraw.Draw(out)
    polydraw(coords,fill=rgba)
    return out
    
    
def GAPic(f_in,size):
    #Creates black background
    out = Image.new("RGBA",size,(0,0,0,255))
    with open(f_in,"rb") as csvfile:
        reader = csv.DictReader(csvfile,fieldnames=["color","coords"], delimiter=';')
        for row in reader:
            rgba = tuple(eval(row["color"]))
            coords = eval(row["coords"])
            poly = makePoly(rgba,coords,size)
            alphaComposite(poly,out)
    return out

def PicFitness(p1,p2)
    return np.sum(np.abs(np.array(list(p1.getdata()))-np.array(list(p2.getdata()))))
