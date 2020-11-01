"""
Saliency detection module
Author : obanmarcos
"""

import cv2 
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import functools

class Saliency():
    
    def __init__(self, waveletType, scalingNumber, scalingFactor, kernelSize = 5, sigmaSize = 0, channelNumber = 3):
        """
        Initializes saliency module.
        """
        self._waveletType = waveletType    
        self._scalingNumber = scalingNumber
        self._kernelSize = kernelSize
        self._sigmaSize = sigmaSize
        self._channelNumber = channelNumber
        self._scalingFactor = scalingFactor
        
        self._GFM = None
        self._FM = None
        
    def __call__(self, raw_image):
        
        img = self._applyCieConversion(raw_image)   # conversion RGB - CIE Lab
        img = self._applyGaussianBlur(img)          # Gaussian pre filtering
        img = self._applyNormalization(img)         # 0-255 normalization

        self._generateFeatureMap(img)
        self._getFeatureMaps()                   # Gets all feature maps 
        
        self.localSaliencyMap()                     # Local saliency map
        self.globalSaliencyMap()                    # Global saliency map

        # Final saliency
        self.Saliency = self._applyGaussianBlur(self._NLnormalization(np.multiply(np.exp(self.GS), self.LS)))
        
        return self.Saliency

    def _applyGaussianBlur(self, image):
        """
        Applies Gaussian filtering with kernelSize and sigmaSize parameters
        """
        return cv2.GaussianBlur(image, (self._kernelSize, self._kernelSize), self._sigmaSize)
    
    def _applyCieConversion(self, image):
        """
        Applies RGB -> CIE lab conversion for histogram uniformity.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    
    def _applyNormalization(self, image):
        """
        Applies normalization between 0,255 on each channel in order to enhance contrast.
        """
        img = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) 
        
        return img
    
    def _getFeatureMaps(self):
        """
        Gets each feature map, starting from the coarsest level. Output to FM
        """
        assert(self._GFM is not None)
        
        FM = []  # FM is an array of feature maps (scaling number, channel, height, width)

        for s in range(self._scalingNumber, 0, -1):

            FM.append(self._createFeatureMap(s))

        self._FM = np.array(FM)
        
    def _generateFeatureMap(self, image):
        """
        Forward pass wavelet transform : On each image channel, a wavelet decomposition is performed, obtaining four images containing a quater subsampled approximation and vertical, horizontal and diagonal detail information. This is peformed scalingNumber times, subsequently using in cascade a coarser approximation. Outputs float feature maps.
        """
        
        GFM = [] # GFM is a list of scale number arrays with different heights and weights [scale number] (channel, detail, height, width)

        for s in range(self._scalingNumber):
            
            W_c = []
            
            for c in range(self._channelNumber):
                        
                if s == 0:
                    cA, (cH, cV, cD) = pywt.dwt2(image[:,:,c], self._waveletType)
                    W_c.append(np.array((cA, cH,  cV, cD)))
                    
                else :
                    
                    # passing previous scale, 0 represents approximation
                    cA, (cH, cV, cD) = pywt.dwt2(GFM[s-1][c, 0 ,:, :], self._waveletType)
                    W_c.append(np.array((cA, cH,  cV, cD)))
            # appends each channel
            GFM.append(np.array(W_c))
        
        # Whole Wavelet set of transformations 
        self._GFM = GFM
    
    def _createFeatureMap(self, waveletLevel):
        """
        Inverse wavelet transform. With the wavelet decomposition, waveletLevel upsamplings are done for each channel, obtaining a feature map FM. Each feature map is the square of the upsampling divided by the scaling factor nu. Resize issues are managed in idwtChannels
        """

        assert(self._GFM is not None)

        for s in range(waveletLevel, 0, -1):
            
            if s == waveletLevel:
                
                # Inverse wavelet upscaling doesnt use coarsest level
                fm = self.__idwtChannels(None, self._GFM[s-1][:, 1, :,:], self._GFM[s-1][:, 2, :,:], self._GFM[s-1][:,3, :,:], self._waveletType)
                
            else:
            
                fm = self.__idwtChannels(fm, self._GFM[s-1][:, 1, :,:], self._GFM[s-1][:, 2, :,:], self._GFM[s-1][:,3, :,:], self._waveletType)
            
        return fm

    def __idwtChannels(self, Wa, Wh, Wv, Wd, wavelet_name, channels = 3):
        
        fm = [] # fm contains [channel, height, width]
        
        # not using coarsest level
        if Wa is not None:
            Wa_aux = []
            for w in Wa:
                
                # setting shape problem : extend approximation to horizontal detail (could've chosen any other detail)
                if w.shape != Wh[0,:,:].shape:
                        
                    Wa_aux.append(cv2.resize(w, Wh[0,:,:].shape[::-1]))

                else:
                    
                    Wa_aux.append(w)
            
            # calculate inverse wavelet transform with scaling factor limit
            for c in range(self._channelNumber):
                
                fm.append(pywt.idwt2((Wa_aux[c], (Wh[c, :,:], Wv[c, :,:], Wd[c, :,:])), wavelet_name)**2/self._scalingFactor)
                
        else :
            
            # setting shape problem : doesnt happen with previous extension
            for c in range(self._channelNumber):
                
                fm.append(pywt.idwt2((None, (Wh[c, :,:], Wv[c, :,:], Wd[c, :,:])), self._waveletType)**2/self._scalingFactor)
                
        return np.array(fm)
    
    def localSaliencyMap(self):
        """
        Obtains local saliency map from feature maps.
        """
        assert(self._FM is not None)
        
        self.LS = np.max(self._FM, axis = 1)
        self.LS = np.sum(self.LS, axis = 0)
        
        self.LS = self._applyGaussianBlur(self.LS)
        
    def _covFM(self):
        """
        Calculates covariance for feature maps
        """
        
        assert(self._FM is not None)
        
        return np.cov(self._FM.reshape(self._channelNumber*self._scalingNumber, -1), rowvar = True)
    
    def _meanFM(self):
        """
        Calculates mean for feature maps
        """
        assert(self._FM is not None)
        
        return np.mean(self._FM.reshape(self._channelNumber*self._scalingNumber, -1), axis = 1)
    
    def _NLnormalization(self, image):
        """
        Calculates non-linear normalization
        """        
        return ((image)**(np.log(2**0.5)))/(2**0.5)
        
    def globalSaliencyMap(self):
        """
        Calculate global saliency map
        """
        # should add feature map subsampling
        img_shape = self._FM.shape[2:4]
        N = self._channelNumber*self._scalingNumber
        FM_unravel = self._FM.reshape(N, -1)
        
        mean = self._meanFM()
        cov = self._covFM()
        icov = np.linalg.pinv(cov)
        det = np.linalg.det(cov)
        
        FM_unravel = (FM_unravel- mean[:, np.newaxis]).T
        # This is the most expensive operation (x-mu) sigma (x-mu)^T (already transposed in second einsum operation)
        exp_arg = np.einsum('ij,ij->i', np.einsum('ij,jk->ik', FM_unravel, icov), FM_unravel)
        
        # multivariate gaussian pdf - global saliency
        self.GS = (1/((2*np.pi)**(N/2)*det**(1/2))*np.exp(-1/2*exp_arg)).reshape(img_shape)

    
    
        
          
        