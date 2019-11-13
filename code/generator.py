## Keras Generator for Rey images
import numpy as np
import keras
from imgaug import augmenters as iaa

from skimage.measure import find_contours, approximate_polygon
from skimage.transform import probabilistic_hough_line, rescale, resize, downscale_local_mean

from skimage.filters import sobel, scharr, threshold_otsu, gaussian
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.draw import polygon_perimeter
from skimage.io import imread,imsave
from skimage.color import grey2rgb

class ROCFGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,path_ROCF, full_set_size, batch_size=32, dim=(30,40), n_channels=1, shuffle=True):
        
        'Initialization'
        self.full_set_size = full_set_size
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

        self.seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(1, 2.0)),
            iaa.CropAndPad(px=100),
            #iaa.Fliplr(0.01),
            #iaa.Flipud(0.01),
            iaa.ElasticTransformation(alpha=(0, 1), sigma=(0, 0.5)),
            iaa.PiecewiseAffine(scale=(0.001, 0.005)),
            iaa.PerspectiveTransform(scale=(0.01, 0.09)),
            ])

        
        self.h,self.w = self.dim
      
        self.path_ROCF = path_ROCF
        self.img_ROCF = self.__get_img_ROCF(self.path_ROCF)
        self.img_ROCF_h = self.img_ROCF.shape[0]
        self.img_ROCF_w = self.img_ROCF.shape[1]
        
        #forzamos las dimensiones a la de la imagen ROCF <<<----No mas!
        
        #self.dim = self.img_ROCF.shape
        #self.h = self.img_ROCF.shape[0]
        #self.w = self.img_ROCF.shape[1]
        
        self.polygons_ROCF, img_ = self.__get_polygons(self.img_ROCF, 
                                                rmv_small=True, 
                                                rmv_redund=False, 
                                                level=0.2, 
                                                thresh=0.1, 
                                                tol = 0.1, 
                                                min_dist=10)        
        
        print('__Init__: Using Keras Generator of Rey Polygons images for autoencoders V.1')


    def __len__(self):
        'Denotes the number of batches per epoch'
        #print('__len__: Denotes the number of batches per epoch')
        return int(np.floor(self.full_set_size / self.batch_size))

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        #print('__getitem__: Generate one batch of data')
        # Generate indexes of the batch
        batch_indexes = self.indexes[batch_index*self.batch_size:(batch_index+1)*self.batch_size]

        #print('\rindex:',batch_index)
        #print('\rindexes:',batch_indexes)

        # Find list of IDs
        list_IDs_temp = batch_indexes

        #print('list_IDs_temp: ',list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.full_set_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        #print('on_epoch_end: Updates indexes after each epoch')

        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, Id in enumerate(list_IDs_temp):

            # Store sample
            
            #print('X[i,].shape',X[i,].shape)
            
            #FA original#X[i,] = np.load('data/' + Id + '.npy')
#            X[i,] = np.expand_dims(self.__Rey_load_image_data_by_Id(Id), axis=4)
            X[i,] = self.__Rey_load_image_data_by_Id(Id)

            #print('X[i,].shape',X[i,].shape)

            # Store class
            #y[i] = self.labels[Id]

        #print('X batch shape:', X.shape)

        return X,None

    def __remove_redundant_polygons(self,polygons, min_dist):
        centroids = [(np.mean(p[:,0]), np.mean(p[:,1])) for p in polygons]
        to_remove = []
        for i,j in enumerate(centroids):
            for ii,jj in enumerate(centroids):
                x0, y0 = centroids[i]
                x1, y1 = centroids[ii]
                dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                smallest = np.argmin([polygons[i].shape[0], polygons[ii].shape[0]])
                if dist < min_dist:
                    to_remove.append([polygons[i], polygons[ii]][int(smallest)].flatten())

        return [p for p in polygons if ~np.isin(p, to_remove).any()]
    
    def __get_polygons(self,img, rmv_small=True, rmv_redund=False, thresh = 50, level=0.5, min_dist=10, tol=0.5, sort=False):
        
        h,w = img.shape

        contours = find_contours(img, level)
        polygons = []

        if rmv_small:
            contours = [c for c in contours if c.shape[0] > thresh]

        for c in contours:
            polygon = approximate_polygon(c, tolerance=tol)
            polygons.extend([polygon])

        if sort:
            polygons.sort(key=lambda x: x[0,0])

        l0 = len(polygons)
        if rmv_redund:
            polygons = __remove_redundant_polygons(polygons, min_dist=min_dist)    
#        print(f'Eliminados {l0-len(polygons)} polígonos')

        img = self.__draw_polygons(polygons, h, w)

        if sort:
            polygons.sort(key=lambda x: x[0,0])

        return polygons, img    
    
    
    def __Image_Rey_polygons_combinations(self,base_polygons, h, w, n, priors= None, sigma = 1):
        if priors == None:
            priors = [poly.shape[0]/sum(poly.shape[0] for poly in base_polygons) for poly in base_polygons]
        
        polygons = np.random.choice(base_polygons, replace = False, size = n, p = priors)
        
        img = self.__draw_polygons(polygons, h, w)
 
        return gaussian(img, sigma)    

    def __Rey_load_image_data_by_Id(self,Id):

        #print('__Rey_load_image_data_by_Id:','self.h:',self.h,'self.w:',self.w)
        
        comb = self.__Image_Rey_polygons_combinations(self.polygons_ROCF, self.img_ROCF_h, self.img_ROCF_w,10)
        Rey_img = self.seq.augment_image(comb)
        
        #imsave('Img_'+str(Id)+'.png', Rey_img)
        
        return  self.__Rey_preprocess_image(Rey_img,Id)

    def __Rey_preprocess_image(self,img,Id):

        # Rescale Image
        # Rotate Image
        # Resize Image
        # Flip Image
        # PCA etc.
        #img = img.reshape((self.h,self.w,self.n_channels))


        img = resize(img,self.dim)

        
#        img = rescale_intensity(img, out_range=(0, 255))
        img = img.astype('float32') / 255

        #imsave('Img_'+str(Id)+'.png', img)
        
        img = img.reshape((self.h,self.w,self.n_channels))

        
        #return a numpy array

        return img
    
    def __get_img_ROCF(self,path):
        
        return imread(path, as_grey=True, origin='lower')
    
    def __draw_polygons(self,polygons, h, w, t=1):
        canv = np.zeros(shape=(h,w))
        for poly in polygons:
            rr, cc = polygon_perimeter(poly[:,0],poly[:,1], shape=(h-1,w-1))
            canv[rr,cc] = 1
            canv[rr+t,cc-t] = 1
            canv[rr-t,cc+t] = 1
            canv[rr-t,cc-t] = 1
            canv[rr+t,cc+t] = 1
            
        return canv
