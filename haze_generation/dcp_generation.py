import numpy as np 
import os 
import cv2 
import math

def DarkChannel(im,sz):
    r,g,b = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    gray = np.float64(gray)/255
    r = 100
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

if __name__ == '__main__': 
    exmaple_path = '/data/URHI/hazy/'
    target_path = '/data/URHI/dcp/'
    # A_path = '/data/URHI/A/'
    os.makedirs(target_path, exist_ok=True)
    # os.makedirs(A_path, exist_ok=True)
    imgs = os.listdir(exmaple_path)
    for name in imgs: 
        hazy = cv2.imread(os.path.join(exmaple_path, name))[:,:,::-1]
        I = hazy / 255.0
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        
        A = np.array([[1.0, 1.0, 1.0]])
        te = TransmissionEstimate(I,A,15)
        t = TransmissionRefine(hazy, te)
        
        t = np.clip(t*255, 0, 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(target_path, name), t)
        
        # np.save(os.path.join(A_path, name.replace('.png', '.npy')), A[0])