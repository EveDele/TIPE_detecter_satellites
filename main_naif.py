# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:22:39 2021

@author: Utilisateur
"""
from skimage import io
import numpy as np


def rogner(im,l0,l1,L0,L1): #l->hauteur L->longueur rogne l'image
    imi=im[l0:l1]
    new=[]
    for i in imi:
        new.append(i[L0:L1])
    return new


def nb(im): 
    #met im image(liste de listes de pixels) rgb en noir et blanc (1pixel=>1float)
    rtn=[]
    for i in im:
        li=[]
        for j in i:
            li.append(0.227*j[0]+0.587*j[1]+0.0114*j[2])
        rtn.append(li)
    return rtn

def conversion_binaire(nb,taux=6):
    #le seuil est fixé à taux*l'écrt type de l'image.
    #nb en noir et blanc
    thresh=taux*np.std(nb)
    image_convertie=nb>thresh
    return image_convertie

#reperer les objets lumineux___________________________________________________


#etoileveprecis (regroupe les pixels lumineux en étoiles)____________________________________________________________________________
def pixels_lumineux(yv): 
    '''image en binaire renvoi la liste des pixels alumés triée par 
    premier(ordonée) puis deuxieme element(abscice)'''
    L=[]
    tl=len(yv)
    tL=len(yv[0])
    for i in range (tl):
        for j in range (tL):
            if yv[i][j]==True:
                L.append([i,j])
    return L

def lignes_indicées(liste_pixels):
    '''créé la liste des pixels voisins (gauche ou droite) à partir de la liste des pixels'''
    newlist=[]
    if liste_pixels!=[]:
        
        Ligne=[liste_pixels[0]]
        tr=len(liste_pixels)-1
        
        for i in range (tr):
            if liste_pixels[i+1]==[liste_pixels[i][0],liste_pixels[i][1]+1]:
                #si le pixel i+1 est bien à droite du pixel i
                #ils sont sur la meme ligne
                Ligne.append(liste_pixels[i+1])
            else:
                #la ligne est finie, on en crée une nouvelle, le pixel i+1 est sur une nouvelle ligne
                newlist.append([Ligne[0][0],Ligne[0][1],Ligne[-1][1]])
                #chaque ligne est reperée par [hauteur,début,fin]
                Ligne=[liste_pixels[i+1]]
        
        newlist.append([Ligne[0][0],Ligne[0][1],Ligne[-1][1]])
    return newlist

def superpose(a1,a2,b1,b2): 
    '''return true <=> les intervalle [a1,a2] et [b1,b2] se superposent'''
    if (a1<=b2 and a1>=b1) or (a2<=b2 and a2>=b1) or (b1<=a2 and b1>=a1) or (b2<=a2 and b2>=a1):
        return True
    else:
        return False

def paquets(liste_lignes):# créé la liste des lignes voisines (haut ou bas) sur la liste des lignes
    liste_a_vider=liste_lignes[0:len(liste_lignes)]
    L=[]
    #on vide liste_a_vider pour remplir L
    while liste_a_vider!=[]:
        etoile=[liste_a_vider[0]]
        #liste de lignes les unes sur les autres.
        ligne_supr=[0]
        #liste des indices des lignes à enlever de liste_a_vider
        lv=len(liste_a_vider)
        
        #ajouter l'étoile à L
        for j in range (ligne_supr[-1]+1,lv):    
            if liste_a_vider[ligne_supr[-1]][0]+1==liste_a_vider[j][0] and superpose(liste_a_vider[ligne_supr[-1]][1],liste_a_vider[ligne_supr[-1]][2],liste_a_vider[j][1],liste_a_vider[j][2]):
                #si deux lignes sont l'une au dessus de l'autre
                #elles sont dans la même étoile
                etoile.append(liste_a_vider[j])
                ligne_supr.append(j)        
        L.append(etoile)
        
        #enlever les lignes qui la composent de liste_a_vider
        for k in range(len(ligne_supr)-1,-1,-1):
            liste_a_vider.pop(ligne_supr[k])
        
    return L#liste de paquets (liste de listes de lignes)

#d'une image couleur fait tout le traitement précedant
def etoiles(image,taux=6):#d'une image rgb des pixels présents groupés en étoiles
    noirblanc=nb(image)
    image_binaire=conversion_binaire(noirblanc,taux)
    pixels=pixels_lumineux(image_binaire)
    lignes=lignes_indicées(pixels)
    liste_etoiles=paquets(lignes)
    return liste_etoiles

def etoilesnb(image):#d'une image noir et blanc des pixels présents groupés en étoiles
    image_binaire=conversion_binaire(image)
    pixels=pixels_lumineux(image_binaire)
    lignes=lignes_indicées(pixels)
    liste_etoiles=paquets(lignes)
    return liste_etoiles

def localisation(etoile): #pour une étoile, donne la localisation du centre de l'etoile.
    xs=0#abscisse moyen
    ys=0#ordonée moyenne
    sl=0#somme des pixels sur les lignes
    for ligne in etoile:
        ys=ys+(ligne[1]+ligne[2])#moyenne faite avec le max et le min de chaque ligne
        l=ligne[2]-ligne[1]+1
        xs=xs+l*ligne[0]#chaque ligne a le poids de son nombre de pixels
        sl=sl+l
    xm,ym=xs/sl,ys/(2*len(etoile))
    return [xm,ym]

    
def liste_localisations(etoiles):#à une liste d'etoiles renvoi la liste des localisations des étoiles
    return[localisation(k) for k in etoiles]
#distance_des_etoiles_au_repere_______________________________________________________________________

def plus_grosse(etoiles):
    '''donne la plus grosse etoile d'une liste d'étoiles'''
    npm=(0,-1)
    t=len(etoiles)
    for k in range(t):
        np=0
        #nombre de pixels dans l'étoile
        for p in etoiles[k]:
            np+=p[2]-p[1]
        if np>npm[0]:
            #si elle est plus grosse que la plus grosse des précedentes
            #c'est la plus grosse
            npm=(np,etoiles[k])
    return npm[1]

def distance(list_loc,loc_ref):
    '''la liste des positions des étoiles, la place du repere dans cette liste. donne la distance de chaque etoile à ce repere.'''
    L=[]
    for loca in list_loc:
        L.append(((loc_ref[0]-loca[0])**2+(loc_ref[1]-loca[1])**2)**0.5)
        #norme euclidienne
    return L

#comparaison_entre_les_images___________________________________________________

def reperes(liste_etoiles1,liste_etoiles2,reperes,prox=10):
    ''''donne la position des reperes de l'image 1 sur la deuxieme image, en supposant que c'est la plus grosse etoile de son voisinage'''
    p1=liste_localisations(reperes)

    letm2=liste_localisations(liste_etoiles2)
    tl2=len(letm2)
    p2=[]
    for j in p1:
        
        #creation d'un voisinage de taille prox dans l'image 2
        lprox=[]
        for i in range(tl2):
            if (letm2[i][0]<j[0]+prox and letm2[i][0]>j[0]-prox and letm2[i][1]<j[1]+prox and letm2[i][1]>j[1]-prox):
                lprox.append(liste_etoiles2[i])
        
        #la plus grosse etoile de ce voisinage est associée au repere de l'image 1
        p2.append(localisation(plus_grosse(lprox)))
    return p2

def liste_distances_2images(liste_etoile1,liste_etoile2,prox=10):
    '''calcule les distances de chaques etoile à la plus grosse 
    sur deux images à la suite  '''
    #repere les plus grosses étoiles comme références
    plus_grosse1a=plus_grosse(liste_etoile1)
    plus_grosse2=reperes(liste_etoile1,liste_etoile2,[plus_grosse1a],prox)[0]
    plus_grosse1=localisation(plus_grosse1a)
    
    #localise chaque etoile
    locas1,locas2=liste_localisations(liste_etoile1),liste_localisations(liste_etoile2)
    
    #fournit la liste des distances des etoiles aux reperes
    distances1,distances2=distance(locas1,plus_grosse1),distance(locas2,plus_grosse2)
    return distances1,distances2

def erreure(dis,x,tolerance=1):
    '''dis : liste de distances sur une image
    x : distance
    return True <=> une etoile de dis se trouve à peu près à cette distance x'''
    for i in dis:
        if abs(x-i)<=tolerance:
            return True
    return False


def erreures(list_dist_2im,tolerance=1):
    ''''repere les etoiles qui ne se répetent pas entre deux images'''
    erreures1=[]
    erreures2=[]
    ld1=len(list_dist_2im[0])
    ld2=len(list_dist_2im[1])
    for j in range(ld1):
    #dans la liste 1
        if not(erreure(list_dist_2im[1],list_dist_2im[0][j],tolerance)):
            #si une etoile est à une distance différente
            #c'est une aberration
            erreures1.append(j)
    for k in range(ld2):
    #dans la liste 2, de même
        if not(erreure(list_dist_2im[0],list_dist_2im[1][k])):
            erreures2.append(k)
    return erreures1,erreures2            

#fonction_finale_______________________________________________________________

def comparaison_naive(im1,im2,taux=6,tolerance=1,prox=10):
    '''compare les distances entre les étoiles sur deux images à la suite 
    pour voir ce qui a bougé.
    
    taux définit à quel point le seuil est elevé
    
    tolérance determine l'écart de distance max 
    entre 2 etoiles sur 2 images à la suite, en pixels
    
    prox determine de combien de pixels l'image 2 pourrait etre décalée
    par rapport à l'image 1'''
    
    #voir les etoiles
    etoiles1=etoiles(im1,taux)
    etoiles2=etoiles(im2,taux)
    
    #lister les distances et etablir les indices des aberrations
    listes_distances=liste_distances_2images(etoiles1,etoiles2,prox)
    indices_aberrations=erreures(listes_distances,tolerance)
    
    #en déduire les coordonées des pixels problématiques
    aberrations=[[],[]]
    for k in indices_aberrations[0]:
        aberrations[0].append(etoiles1[k])
    for l in indices_aberrations[1]:
        aberrations[1].append(etoiles2[l])
    
    return aberrations
