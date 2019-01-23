import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
import cv2

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint
from PIL import Image


def add_elastic_transform(image, alpha, sigma, pad_size=30, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : alpha is a scaling factor
        sigma :  sigma is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    #image_size = int(image.shape[0])
    image = np.pad(image, pad_size, mode="symmetric")
    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
#    return cropping(map_coordinates(image, indices, order=1).reshape(shape), 572, pad_size, pad_size), seed
    return map_coordinates(image, indices, order=1).reshape(shape)

def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
        # no effect
    return image


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image

    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image)
    return noise_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    #image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 1024] = 1024
    image[image < -1024] = -1024
    #image = image.astype("uint8")
    return image


def approximate_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def normalization1(image, mean, std):
    """ Normalization using mean and std
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """

    image = image / 1024  # values will lie between 0 and 1.
    image = (image - mean) / std

    return image


def normalization2(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new


def stride_size(image_len, crop_num, crop_size):
    """return stride size
    Args :
        image_len(int) : length of one size of image (width or height)
        crop_num(int) : number of crop in certain direction
        crop_size(int) : size of crop
    Return :
        stride_size(int) : stride size
    """
    return int((image_len - crop_size)/(crop_num - 1))


def multi_cropping(image, crop_size, crop_num1, crop_num2):
    """crop the image and pad it to in_size
    Args :
        images : numpy arrays of images
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
    Return :
        cropped_imgs : numpy arrays of stacked images
    """

    img_height, img_width = image.shape[0], image.shape[1]
    assert crop_size*crop_num1 >= img_width and crop_size * \
        crop_num2 >= img_height, "Whole image cannot be sufficiently expressed"
    assert crop_num1 <= img_width - crop_size + 1 and crop_num2 <= img_height - \
        crop_size + 1, "Too many number of crops"

    cropped_imgs = []
    # int((img_height - crop_size)/(crop_num1 - 1))
    dim1_stride = stride_size(img_height, crop_num1, crop_size)
    # int((img_width - crop_size)/(crop_num2 - 1))
    dim2_stride = stride_size(img_width, crop_num2, crop_size)
    for i in range(crop_num1):
        for j in range(crop_num2):
            cropped_imgs.append(cropping(image, crop_size,
                                         dim1_stride*i, dim2_stride*j))
    return np.asarray(cropped_imgs)


# IT IS NOT USED FOR PAD AND CROP DATA OPERATION
# IF YOU WANT TO USE CROP AND PAD USE THIS FUNCTION
"""
def multi_padding(images, in_size, out_size, mode):
    '''Pad the images to in_size
    Args :
        images : numpy array of images (CxHxW)
        in_size(int) : the input_size of model (512)
        out_size(int) : the output_size of model (388)
        mode(str) : mode of padding
    Return :
        padded_imgs: numpy arrays of padded images
    '''
    pad_size = int((in_size - out_size)/2)
    padded_imgs = []
    for num in range(images.shape[0]):
        padded_imgs.append(add_padding(images[num], in_size, out_size, mode=mode))
    return np.asarray(padded_imgs)
"""


def cropping(image, crop_size, dim1, dim2):
    """crop the image and pad it to in_size
    Args :
        images : numpy array of images
        crop_size(int) : size of cropped image
        dim1(int) : vertical location of crop
        dim2(int) : horizontal location of crop
    Return :
        cropped_img: numpy array of cropped image
    """
    cropped_img = image[dim1:dim1+crop_size, dim2:dim2+crop_size]
    return cropped_img


def add_padding(image, in_size, out_size, mode):
    """Pad the image to in_size
    Args :
        images : numpy array of images
        in_size(int) : the input_size of model
        out_size(int) : the output_size of model
        mode(str) : mode of padding
    Return :
        padded_img: numpy array of padded image
    """
    pad_size = int((in_size - out_size)/2)
    padded_img = np.pad(image, pad_size, mode=mode)
    return padded_img


def division_array(crop_size, crop_num1, crop_num2, dim1, dim2):
    """Make division array
    Args :
        crop_size(int) : size of cropped image
        crop_num2 (int) : number of crop in horizontal way
        crop_num1 (int) : number of crop in vertical way
        dim1(int) : vertical size of output
        dim2(int) : horizontal size_of_output
    Return :
        div_array : numpy array of numbers of 1,2,4
    """
    div_array = np.zeros([dim1, dim2])  # make division array
    one_array = np.ones([crop_size, crop_size])  # one array to be added to div_array
    dim1_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim2_stride = stride_size(dim2, crop_num2, crop_size)  # horizontal stride
    for i in range(crop_num1):
        for j in range(crop_num2):
            # add ones to div_array at specific position
            div_array[dim1_stride*i:dim1_stride*i + crop_size,
                      dim2_stride*j:dim2_stride*j + crop_size] += one_array
    return div_array


def image_concatenate(image, crop_num1, crop_num2, dim1, dim2):
    """concatenate images
    Args :
        image : output images (should be square)
        crop_num2 (int) : number of crop in horizontal way (2)
        crop_num1 (int) : number of crop in vertical way (2)
        dim1(int) : vertical size of output (512)
        dim2(int) : horizontal size_of_output (512)
    Return :
        div_array : numpy arrays of numbers of 1,2,4
    """
    crop_size = image.shape[1]  # size of crop
    empty_array = np.zeros([dim1, dim2]).astype("float64")  # to make sure no overflow
    dim1_stride = stride_size(dim1, crop_num1, crop_size)  # vertical stride
    dim2_stride = stride_size(dim2, crop_num2, crop_size)  # horizontal stride
    index = 0
    for i in range(crop_num1):
        for j in range(crop_num2):
            # add image to empty_array at specific position
            empty_array[dim1_stride*i:dim1_stride*i + crop_size,
                        dim2_stride*j:dim2_stride*j + crop_size] += image[index]
            index += 1
    return empty_array


def create_addnoise_file(noisetype,imagetype,x,y):
    """create a file of
    Args :
        noisetype: 'gaussian' or 'uniform'
        imagetype:'masks' or 'images'
        x:first parameter
        y:second parameter
    Return :
        
    """
    if (imagetype=='images' or imagetype=='masks') and (noisetype=='gaussian' or noisetype=='uniform'):
        
        debutnom=noisetype+str(x)+'_'+str(y)+'_'
        dossier='./'+noisetype+'/'+debutnom+imagetype
        os.mkdir(dossier)
        for nomf in os.listdir('./datapfe/'+imagetype):
            if imagetype=='images':
                im = Image.fromarray(np.loadtxt("./datapfe/images/"+nomf))
                i = np.array(im)
                g = Image.fromarray(add_gaussian_noise(i, x, y))  
                np.savetxt(dossier+'/'+debutnom+nomf, np.array(g))
            else:
                i = img.imread("./datapfe/masks/"+nomf)                
                g = Image.fromarray(add_gaussian_noise(i, x, y))  
                cv2.imwrite(dossier+'/'+debutnom+nomf, np.array(g))
 
        
def save_gaussian_dossier(t,x,y):
    if t=='images' or t=='masks':
        debutnom="gaussian"+str(x)+'_'+str(y)+'_'
        dossier='./gaussian/'+debutnom+t
        os.mkdir(dossier)
        for nomf in os.listdir('./datapfe/'+t):
            if t=='images':
                i = np.loadtxt("./datapfe/images/"+nomf)
                g = add_gaussian_noise(i, x, y)
                np.savetxt(dossier+'/'+debutnom+nomf, g)
            else:
                i = img.imread("./datapfe/masks/"+nomf)                
                g = add_gaussian_noise(i, x, y)  
                cv2.imwrite(dossier+'/'+debutnom+nomf, g)

def save_uniform_dossier(t,x,y):
    if t=='images' or t=='masks':
        debutnom="uniform"+str(x)+'_'+str(y)+'_'
        dossier='./uniform/'+debutnom+t
        os.mkdir(dossier)
        for nomf in os.listdir('./datapfe/'+t):
            if t=='images':
                i = np.loadtxt("./datapfe/images/"+nomf)
                g = add_uniform_noise(i, x, y)
                np.savetxt(dossier+'/'+debutnom+nomf, g)
            else:
                i = img.imread("./datapfe/masks/"+nomf)                
                g = add_uniform_noise(i, x, y)  
                cv2.imwrite(dossier+'/'+debutnom+nomf, g)
                
def save_elastic_dossier(t,a,s,p):
    if t=='images' or t=='masks':
        debutnom="elastic"+str(a)+'_'+str(s)+'_'+str(p)+'_'
        dossier='./elastic/'+debutnom+t
        os.mkdir(dossier)
        for nomf in os.listdir('./datapfe/'+t):
            if t=='images':
                i = np.loadtxt("./datapfe/images/"+nomf)
                g = add_elastic_transform(i, a, s, p) 
                np.savetxt(dossier+'/'+debutnom+nomf, g)
            else:
                i = img.imread("./datapfe/masks/"+nomf)                
                g = add_elastic_transform(i, a, s, p)   
                cv2.imwrite(dossier+'/'+debutnom+nomf, g)     


def random_uniform(k):
    """Create a file imagesk and masksk
    Ces dossiers contiennent des images et masks obtenus en ajoutant 
    un bruit unifome aléatoire.
    Arg :
        k : the number of the file created
    """    
    x = randint(-100,0)
    y = randint(x,100)
    print('lower boundary of output interval: ', x,'\nupper boundary of output interval: ', y)
    os.mkdir("./datapfe/images"+str(k))
    os.mkdir("./datapfe/masks"+str(k))

    for file in os.listdir('./datapfe/images'):
                i = np.loadtxt("./datapfe/images/"+file)
                ei = add_uniform_noise(i,x,y)
                np.savetxt("./datapfe/images"+str(k)+'/'+str(k)+'_'+file, ei)
    for file in os.listdir('./datapfe/masks'):
                m = img.imread("./datapfe/masks/"+file)                
                em = add_uniform_noise(m,x,y)
                cv2.imwrite("./datapfe/masks"+str(k)+'/'+str(k)+'_'+file, em)    
                

def random_elastic(k):
    """Create a file imagesk and masksk
    Ces dossiers contiennent des images et masks obtenus en effectuant
    une déformation élastique aléatoire.
    Arg :
        k : the number of the file created
    """  
    a = randint(0,100)
    s = a//10 + randint(0,30)
    print('alpha: ', a, '\nsigma: ', s)
    os.mkdir("./datapfe/images"+str(k))
    os.mkdir("./datapfe/masks"+str(k))
    for file in os.listdir('./datapfe/images'):
                i = np.loadtxt("./datapfe/images/"+file)
                ei = add_elastic_transform(i, a, s)
                np.savetxt("./datapfe/images"+str(k)+'/'+str(k)+'_'+file, ei)
                
    for file in os.listdir('./datapfe/masks'):
                m = img.imread("./datapfe/masks/"+file)                
                em = add_elastic_transform(m, a, s)  
                cv2.imwrite("./datapfe/masks"+str(k)+'/'+str(k)+'_'+file, em)    
            

#def save_gaussian(fichier,x,y): 
#    if fichier[-1]=='m':
#         dossier=1
#    if fichier[-1]=='t':    
#         dossier=str(x)+'_'+str(y)
#         im = Image.fromarray(np.loadtxt("./datapfe/images/"+nom))
#         i = np.array(im)
#         g = Image.fromarray(add_gaussian_noise(i, x, y))  
#         np.savetxt('./gaussian/'+dossier+'_'+nom, np.array(g))
#        

if __name__ == "__main__":


    #a: mask
    #b: image    
    a = img.imread("./datapfe/masks/01_000188.pgm")
    ma = Image.fromarray(a)
    im = Image.fromarray(np.loadtxt("./datapfe/images/01_000132.txt"))
    b = np.array(im)
    
    
    mc = Image.fromarray(np.loadtxt("./datapfe/images1/1_01_000245.txt"))
#    mc.show()
    
#   ea = Image.fromarray(add_elastic_transform(b,50,5))
#    ea.show()
    
    
    
    #add gaussian noise images
    ga = Image.fromarray(add_gaussian_noise(a, 100, 25))
    gb = Image.fromarray(add_gaussian_noise(b, 100, 200))    
    
    
    
    #add uniform noise images
    ua = Image.fromarray(add_uniform_noise(a, -150, 100))
    ub = Image.fromarray(add_uniform_noise(b, -150, 150))
    ub.show()

    #change brightness images  
    cba = Image.fromarray(change_brightness(a, 240))
    cbb = Image.fromarray(change_brightness(b, 250))

    
    #add elastic transform noise images 
    
#    for alpha in range (0, 100,10):
#    alpha = 0
#    for sigma in range (5,50,5):
#            for p in range (5,50,5):
#                                eb = Image.fromarray(add_elastic_transform(b,alpha,sigma,p))
#                                cv2.imwrite('./imagetest/elastic transform/ielastic'+str(alpha)+'_'+str(sigma)+'_'+str(p)+'.pgm',np.array(eb))

#    alpha=0
#    sigma=2
#    p=2
#    ea = Image.fromarray(add_elastic_transform(a,-50,2,1))
#    eb = Image.fromarray(add_elastic_transform(b,alpha,sigma,p))
#    #eb.show()
#    cv2.imwrite('./imagetest/elastic transform/ielastic'+str(alpha)+'_'+str(sigma)+'_'+str(p)+'.pgm',np.array(eb))
    

    
    #normalization 2 images
    n2b = Image.fromarray(normalization2(b, max=250, min=-50))
    
    #approximate images
    apb = Image.fromarray(approximate_image(b))
    

        