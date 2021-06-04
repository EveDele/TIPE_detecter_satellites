# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:00:42 2021

@author: Utilisateur
"""

import cv2
import numpy as np

#pour tester les fonctions
from ciel import ciel
img=np.copy(ciel((100,100),0.01,(1,2),0,3))


def mask(img,lines):
    '''créé le masque à partir de la transformée de Hough d'une image'''
    #image de sortie
    im2=np.zeros_like(img)
    
    #variables utiles
    l,l1=len(img),len(img[0])
    
    if lines is not None :
        for line in lines:
            #r=x*cos(t)+y*sin(t) 
            r,t = line[0]

            #on observe en x=0 et x=l
            if np.sin(t)!=0:
                x1,y1=0,r/np.sin(t)
                x2,y2=l,(r-l*np.cos(t))/np.sin(t)
            else :
                x1,y1=r,0
                x2,y2=r,l1
            x1,x2,y1,y2=int(x1),int(x2),int(y1),int(y2)
            
            #tracé des lignes
            im2=cv2.line(im2, (x1, y1), (x2, y2), 1, 2)
    return im2

def filtre_satellites_par_hough(image,points_alignés_min,fond_du_ciel):
    '''observe les droites présentes dans l'image pour ne conserver qu'elles'''
    #seuil
    img=np.copy(image)+1-fond_du_ciel
    gray=np.uint8(img)
    
    #transformée de Hough : lines contient les parametres r et t des droites présentes
    lines=cv2.HoughLines(gray,1,np.pi/180,points_alignés_min)#image,resolution de r, de t, niveau
    
    #création de l'image contenant les droites
    masque=mask(img,lines)
    
    #image de sortie
    img=masque*image
    
    return img