# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:24:18 2021

@author: Utilisateur
"""
from skimage import io
import numpy as np

#pour tester les fonctions
from ciel import ciel
img=np.copy(ciel((100,100),0.01,(1,2),0,35))

def fichier_image(nom_fichier):
    '''convertis un fichier image en tableau de numpy'''
    image=io.imread(nom_fichier)
    return image

def sous_image(image,i,j,ker,lk,lk0,l,lm):
    ''' créé une image de la taille de ker qui contient 
    le voisinage du pixel (i,j) et 0 à l'exterieur de l'image''' 
    s_im=np.zeros_like(ker,np.float32)
    for k in range(lk):
        if i+k<l:
            for n in range(lk0):
                if j+n<lm:
                    s_im[k,n]=image[i+k,j+n]
    return s_im

def erosion(image,ker):
    '''realise une érosion de l'image par l'element ker''' 
    im=np.copy(image)
    #creation de l'image de sortie
    lk,lk0,l,lm=len(ker),len(ker[0]),len(im),len(im[0])
    #tailles de l'image et de l'element structurant
    sous_image1=np.copy(ker)
    for i in range(-lk//2,l-lk//2):
        for j in range(-lk0//2,lm-lk0//2):
            #(i,j) correspond au coin haut gauche du voisinage étudié
            sous_image1=sous_image(image,i,j,ker,lk,lk0,l,lm)
            #le voisinage rectangle du pixel (i+lk//2,j+lk0//2)
            im[i+lk//2,j+lk0//2]=min([min(sous_image1[k]) for k in range(len(sous_image1))])
            #le minimum du voisinage est affecté au pixel central sur l'image de sortie
    return im
         
def dilate(image,ker):
    '''comme erosion, mais prends le maximum du voisinage plutot que le minimum'''
    im=np.copy(image)
    lk,lk0,l,lm=len(ker),len(ker[0]),len(im),len(im[0])
    sous_image1=np.copy(ker)
    for i in range(-lk//2,l-lk//2):
        for j in range(-lk0//2,lm-lk0//2):
            sous_image1=sous_image(image,i,j,ker,lk,lk0,l,lm)
            im[i+lk//2,j+lk0//2]=max([max(sous_image1[k]) for k in range(len(sous_image1))])
    return im
    


def filtre_satellites_par_voisinage(img,n,m):#si n<1, n=3. de même pour m.
    '''observe les voisinages de chaques pixels et ne conserves que leurs points d'intersection. m>n>=0 m taille des lignes, n celle du carré'''
    image1=np.copy(img)
    image2=np.copy(img)
    
    #elements structurants
    dker=np.ones((n,n))#element structurant de dilatation
    eker2=np.array((m)*[[1]])#element structurant d'erosion 1
    eker1=np.array([(m)*[1]])#element structurant d'erosion 2
    
    #premiere image
    image1=dilate(image1,dker)
    image1=erosion(image1,eker1)
    
    #deuxieme image
    image2=dilate(image2,dker)
    image2=erosion(image2,eker2)
    
    #superposition des resultats
    image=image1+image2   
    return image