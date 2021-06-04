# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:27:34 2021

@author: Valentin GUILLET
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
import scipy.special
import scipy.signal
import time

def generer_fond_moyen(taille_x, taille_y, valeur, limites = (0,1), plot = True):
    """
    Permet de générer une image uniforme de taille (taille_x, taille_y)
    Les pixels ont pour valeur "valeur"
    

    Parameters
    ----------
    taille_x : Entier strictement positif
        Taille de l'image selon l'axe horizontal
    taille_y : Entier strictement positif
        Taille de l'image selon l'axe vertical
    valeur : float
        Valeur du fond moyen
    limites : (min, max), optional
        limites pour l'affichage des figures
    plot : Bool, optional
        Si True, on affiche l'image. The default is True.
         
    Returns
    -------
    image : np.array
        l'image de fond moyen

    """
    t = time.time()
    image = valeur*np.ones((taille_y,taille_x))
    
    if plot:
        try:
            plt.figure("Fond moyen").clf()
        except:
            pass
        plt.figure("Fond moyen")
        plt.title("Fond moyen")
        plt.imshow(image,cmap = "gist_gray")
        plt.clim(limites[0],limites[1])
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    print(f'Le temps pour générer le fond moyen est de {time.time()-t} secondes')
    return image


def ajouter_etoiles(image_entree, densite=0.1, minimum = 0, maximum = 1, limites = (0,1), plot = True, noyau_aleatoire = 0):
    """
    modifie une image de fond moyen pour y ajouter des étoiles avec une certaine densité, un minimum et un maximum

    Parameters
    ----------
    image_entree : np.array()
        image de fond moyen
    densite : float compris entre 0 et 1, optional
        densité d'étiles dans l'image finale. The default is 0.1.
    minimum : float, optional
        valeur minimale pour la luminosité des étoiles. The default is 0.
    maximum : float, optional
        valeur maximale pour la luminosité des étoiles. The default is 1.
    limites : (min, max), optional
        limites pour l'affichage des figures
    plot : Bool, optional
        Si True, on affiche l'image de sortie. The default is True.
    noyau_aleatoire : int, optional
        Si différent de 0, on prend ce noyau pour l'aléatoire de cette fonction

    Returns
    -------
    image_sortie : np.array
        image contenant le fond moyen et les étoiles.

    """
    t = time.time()
    
    taille_x = np.shape(image_entree)[1] #taille de l'image selon x
    taille_y = np.shape(image_entree)[0] #taille de l'image selon y

    nombre_etoiles = round(densite*taille_x*taille_y) #le nombre d'étoiles à ajouter dans l'image
    
    if noyau_aleatoire:
        rand.seed(noyau_aleatoire) # initialise le générateur aléatoire de nombres
    else:
        rand.seed()
        
    # On génère l'ensemble des coordonnées des étoiles dans l'image
    N = 0 # initialise le compteur d'étoiles
    coords_etoiles = [] #contiendra le couple de coordonnées des étoiles
    points_comptes_plusieurs_fois = 0 #on compte le nombre de fois où un point est sélectionné plusieurs fois par le générateur aléatoire
    while N < nombre_etoiles: # tant qu'on n'a pas le bon nombre d'étoiles
        x_etoile = rand.choice(range(taille_x)) # on génère aléatoirement une coordonnée x
        y_etoile = rand.choice(range(taille_y)) # et une coordonnée y
        coord_etoile = [x_etoile, y_etoile] # on construit le couple de coordonénes
        if coord_etoile not in coords_etoiles: # s'il n'est pas déjà dans la liste des coordonnées
            coords_etoiles.append(coord_etoile) # on l'y ajoute
            N+=1
        else:
            points_comptes_plusieurs_fois+=1
    print(f"Il y a eu {points_comptes_plusieurs_fois} points comptés plusieurs fois sur un total de {nombre_etoiles} étoiles")
    # on génère un tableau aléatoire avec une loi de probabilité uniforme pour la luminosité des étoiles
    luminosites = [rand.uniform(minimum, maximum) for k in range(nombre_etoiles)]
    
    # on place les étoiles dans l'image de sortie
    image_sortie = image_entree[:]
    for i in range(nombre_etoiles):
        x = coords_etoiles[i][0]
        y = coords_etoiles[i][1]
        image_sortie[y,x] = luminosites[i]
    
    if plot:
        try:
            plt.figure("fond étoilé").clf()
        except:
            pass
        plt.figure("fond étoilé")
        plt.title("fond étoilé")
        plt.imshow(image_sortie, cmap = "gist_gray")
        plt.clim(limites[0],limites[1])
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    print(f'Le temps pour ajouter les étoiles est de {time.time()-t} secondes')
    return image_sortie
    

def ajouter_trainees(image_entree, intervalle_nombre, intervalle_longeur, intervalle_luminosite, limites = (0,1), plot = True, noyau_aleatoire = 0):
    """
    ajoute les trainées du passage d'un satellite dans l'image. On pondère la "luminosité" du pixel par le temps durant lequel le satellite
    a survolé le pixel considéré (modèle plus réaliste)

    Parameters
    ----------
    image_entree : np.array
        image d'entrée, avec par exemple un fond étoilé
    intervalle_nombre : (min, max)
        définit l'intervalle dans lequel sera tiré au hasard le nombre de trainées dans l'image
    intervalle_longeur : (min, max)
        définit l'intervalle dans lequel sera tirée au hasard la longueur de la trajectoire du satellite
        La normalisation est fixée à 1 pour la diagonale de l'image
    intervalle_luminosite : (min, max)
        définit l'intervalle dans lequel sera tirée au hasard la luminosité du satellite
        compris entre 0 et 1. 1 correspondant au maximum de la dynamique de l'image
    limites : (min, max), optional
        limites de la dynamique de l'image. The default is (0,1).
    plot : Bool, optional
        Si True, on affiche l'image de sortie. The default is True.
    noyau_aleatoire : int, optional
        Si différent de 0, on prend ce noyau pour l'aléatoire de cette fonction

    Returns
    -------
    image_sortie : np.array
        image de sortie contenant les trainées de satellites

    """
    t = time.time()
    
    # Calcule la taille de l'image d'entrée
    taille_x = np.shape(image_entree)[1] #taille de l'image selon x
    taille_y = np.shape(image_entree)[0] #taille de l'image selon y
    # Calcul de la diagonale
    diag = math.sqrt(taille_x**2 + taille_y**2)
    
    if noyau_aleatoire:
        rand.seed(noyau_aleatoire) # initialise le générateur aléatoire de nombres
    else:
        rand.seed()
        
    
    # Genère le nombre de trainées qui seront dans l'image
    nombre_trainees = rand.randint(intervalle_nombre[0],intervalle_nombre[1]) 
    
    # image de sortie
    image_sortie = image_entree[:]
    
    # pour chaque trainée
    for n in range(nombre_trainees):
        # on tire au hasard:
        # une longeur [px]
        longueur = round(rand.uniform(intervalle_longeur[0],intervalle_longeur[1])*diag)
        # une orientation [rad]
        orientation = rand.uniform(0,2*np.pi)
        # un point de départ [px]
        x_init = rand.choice(range(taille_x+1))
        y_init = rand.choice(range(taille_y+1))
        # une luminosité
        luminosite = rand.uniform(intervalle_luminosite[0],intervalle_luminosite[1])
        
        # on calcule l'équation de la droite définie par (x_init,y_init) et orientation
        if orientation == 0 or orientation == 2*np.pi:
            # l'équation de la droite est de la forme y = u
            points_intersection = [[x,y_init] for x in range(x_init, min(x_init+longueur,taille_x)+1)]
        elif orientation == np.pi/2:
            points_intersection = [[x_init,y] for y in range(x_init, min(y_init+longueur,taille_y)+1)]
        elif orientation == np.pi:
            points_intersection = [[x,y_init] for x in range(max(0,x_init-longueur), x_init+1)]
        elif orientation == 3*np.pi/2:
            points_intersection = [[x_init,y] for y in range(max(0,y_init-longueur), y_init+1)]
        else:
            #l'equation est de la forme y = ax+b avec a nécessairement différent de 0
            a = math.tan(orientation)
            b = y_init-a*x_init
        
            # on calcul l'ensemble des points d'intersection avec la grille de pixels
            points_intersection = [[x_init,y_init]]
            X = x_init
            Y = y_init
                # on cherche à savoir si le prochain point sera en bas ou en haut, à gauche ou à droite
            gauche = True if orientation >= np.pi/2 and orientation <= 3*np.pi/2 else False
            droite = not gauche
            haut = True if orientation <= np.pi else False
            bas = not haut
                # tant que les points d'intersection sont dans l'image (car on peut éventuellement en sortir)
                # ou que la distance au point d'origine est plus petit que "longueur"
            while (0 < X and X < taille_x) and (0 < Y and Y < taille_y) and math.sqrt((X-x_init)**2+(Y-y_init)**2) < longueur:
                if droite and haut:
                    # coordonnées du coin supérieur droit du pixel considéré
                    x_coin = math.floor(X+1)
                    y_coin = math.floor(Y+1)
                    # angle défini par le point d'origine et le coin supérieur droit du pixel
                    angle_limite = math.atan((y_coin-Y)/(x_coin-X))
                    # selon le cas, on les coordonnées du point d'intersection suivant ne sont pas les mêmes (faire un dessin)
                    if orientation < angle_limite:
                        X = x_coin
                        Y = a*X+b
                    else:
                        Y = y_coin
                        X = (Y-b)/a
                # on poursuit
                elif gauche and haut:
                    x_coin = math.ceil(X-1)
                    y_coin = math.floor(Y+1)
                    angle_limite = math.atan((y_coin-Y)/(x_coin-X))+np.pi
                    if orientation < angle_limite:
                        Y = y_coin
                        X = (Y-b)/a
                    else:
                        X = x_coin
                        Y = a*X+b
                elif gauche and bas:
                    x_coin = math.ceil(X-1)
                    y_coin = math.ceil(Y-1)
                    angle_limite = math.atan((y_coin-Y)/(x_coin-X))+np.pi
                    if orientation < angle_limite:
                        X = x_coin
                        Y = a*X+b
                    else:
                        Y = y_coin
                        X = (Y-b)/a
                elif droite and bas:
                    x_coin = math.floor(X+1)
                    y_coin = math.ceil(Y-1)
                    angle_limite = math.atan((y_coin-Y)/(x_coin-X))+2*np.pi
                    if orientation < angle_limite:
                        Y = y_coin
                        X = (Y-b)/a
                    else:
                        X = x_coin
                        Y = a*X+b
                else:
                    print("il y a un problème dans la disjonction des cas")

                points_intersection.append([X,Y])
        
            
            
        
            # on peut maintenant insérer la trainée dans l'image
            for i in range(len(points_intersection)-1):
                # calcule les coordonnées du pixel qu'il faut modifier
                x_pixel = int(math.floor(points_intersection[i][0]) if droite else int(math.floor(points_intersection[i][0]))-1)
                y_pixel = int(math.floor(points_intersection[i][1]) if haut else int(math.floor(points_intersection[i][1]))-1)

                # calcule la distance parcourue par le satellite dans le pixel
                # on normallise la valeur par la diagonale d'un pixel, qui vaut racine de 2
                x1 = points_intersection[i][0]
                x2 = points_intersection[i+1][0]
                y1 = points_intersection[i][1]
                y2 = points_intersection[i+1][1]
                distance = math.sqrt((x2-x1)**2+(y2-y1)**2)/math.sqrt(2)
                
                luminosite_pixel = distance*luminosite*limites[1]
                
                # on ajoute la partie de la trainée du pixel considéré à l'image de sortie
                # on ne peut pas dépasser la dynamique de la caméra, si c'est le cas, on sature à la limite (définie à 1 par défaut)
                if image_sortie[y_pixel,x_pixel] + luminosite_pixel < limites[1]:
                    image_sortie[y_pixel,x_pixel] += luminosite_pixel
                else:
                    image_sortie[y_pixel,x_pixel] = limites[1] 
                    
        
                
    if plot:
        try:
            plt.figure("fond étoilé et trainées").clf()
        except:
            pass
        plt.figure("fond étoilé et trainées")
        plt.figure("fond étoilé et trainées")
        plt.imshow(image_sortie, cmap = "gist_gray")
        plt.clim(limites[0],limites[1])
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    print(f'Le temps pour ajouter les trainées est de {time.time()-t} secondes')
    return image_sortie


def convoluer_PSF(image_entree, rayon = 8, mode = "Airy", fond_moyen = 0, limites = (0,1), plot = True):
    """
    Permet de calculer l'image après qu'elle soit passée par un instrument optique, qui, au premier ordre, réalise une convolution de l'image avec la PSF, à cause de la diffraction

    Parameters
    ----------
    image_entree : np.array
        une image d'un objet qui n'a pas encore été imagé par un instrument
    rayon : float, optional
        un nombre strictement positif qui définit le rayon de la PSF. The default is 8.
    mode : str, optional
        plusieurs modes possible : 
            "Airy" : Si l'instrument est en limite de diffraction.
            "Focus": The default is "Airy".
    fond_moyen : float, optional
        valeur du fond moyen. The default is 0.
    limites : (min,max), optional
        limites de la dynamique de l'image. The default is (0,1).
    plot : Bool, optional
        Si True, on affiche les images. The default is True.

    Returns
    -------
    image_sortie : np.array
        image d'entrée convoluée par la PSF

    """
    t = time.time()
    
    # Calcule la taille de l'image d'entrée
    taille_x = np.shape(image_entree)[1] #taille de l'image selon x
    taille_y = np.shape(image_entree)[0] #taille de l'image selon y
    
    
    #Si on est en limite de diffraction
    if mode == "Airy":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((8*math.ceil(rayon)+1, 8*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)/rayon*1.219670
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = (2*scipy.special.jv(1,np.pi*r)/(np.pi*r))**2 if r else 1
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Airy").clf()
            except:
                pass
            plt.figure("PSF Airy")
            plt.title("PSF Focus")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    # si la mise au point n'est pas parfaite
    elif mode == "Focus":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((2*math.ceil(rayon)+1, 2*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2),int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)
                #pondérer les points du bords du cercle par l'intégrale de la portion de disque sur ce pixel
                if abs(r-rayon) < np.sqrt(2)/2:
                    value = 0
                    M = 10
                    for u in np.linspace(i-0.5,i+0.5, M):
                        for v in np.linspace(j-0.5, j+0.5, M):
                            if math.sqrt(u**2+v**2) < rayon:
                                value +=1
                    value = float(value)/(M**2)
                elif r > rayon:
                    value = 0
                elif r < rayon:
                    value = 1
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = value
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Focus").clf()
            except:
                pass
            plt.figure("PSF Focus")
            plt.title("PSF Focus")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    elif mode == "Araignée":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((20*math.ceil(rayon)+1, 20*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        angle_araignee = 10 #° [entre 0 et 89.9°]
        rayon_sinc = 0.2
        araignee = np.sinc(np.linspace(0, 1, int((N-1)/2)+1))**2
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)/rayon*1.219670
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = (2*scipy.special.jv(1,np.pi*r)/(np.pi*r))**2 if r else 1
                M = 10
                valeur = 0
                a = np.tan(angle_araignee*np.pi/180)
                b = rayon_sinc
                a2 = -a
                if abs(a*j-i) <= b+0.5+0.5*a  or abs(a2*i-j) <= b+0.5+0.5*a2:
                    for u in np.linspace(j-0.5, j+0.5,M):
                        for v in np.linspace(i-0.5, i +0.5, M):
                            if abs(a*u-v) <=rayon_sinc:
                                valeur += 1
                            if abs(a2*v-u) <= rayon_sinc:
                                valeur += 1
                    valeur /= 2*M**2
                    R = math.sqrt(i**2+j**2)
                    if R <= (N-1)/2:
                        PSF[i+int((N-1)/2),j+int((N-1)/2)] += valeur*araignee[round(R)]
                
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Araignée").clf()
            except:
                pass
            plt.figure("PSF Araignée")
            plt.title("PSF Araignée")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    else:
        print("Ce mode n'est pas pris en compte pour le moment")
        raise NameError(f"le mode {mode} n'est pas encore défini")
        
    # on réalise la convolution, qui revient à faire un produit matriciel
    # les effets de bords sont gérés ici en considérant prolongeant l'image par le fond moyen
    # c'est une version très naïve du produit de convolution, au sens où la complexité de cet algo n'est pas idéale.
    # une autre version est possible en passant dans l'espace de Fourier (quasi équivalent de l'espace de Laplace utilisé en SI)
    # dans cet espace, le produit de convolution se transforme en produit simple, la complexité est bien meilleure dans ce cas :) 
    image_sortie = np.zeros((taille_y,taille_x))
    for i in range(taille_x):
        for j in range(taille_y):
            pixel = 0
            for k in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                for l in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                    if i+k < 0 or i+k > taille_x-1 or j+l < 0 or j+l > taille_y-1 :
                        pixel += PSF[l+int((N-1)/2), k+int((N-1)/2)]*fond_moyen
                    else:
                        pixel += PSF[l+int((N-1)/2), k+int((N-1)/2)]*image_entree[j+l,i+k]
            image_sortie[j,i] = pixel
    
    # on corrige l'exposition de l'image, en se disant que le max de l'image d'entrée doit correspondre au max de l'image de sortie
    maximum_image_entree = np.amax(image_entree)
    maximum_image_sortie = np.amax(image_sortie)
    image_sortie *= maximum_image_entree/maximum_image_sortie
    
    if plot:
        try:
            plt.figure(f"Image convoluée par la PSF {mode} de manière naïve").clf()
        except:
            pass
        plt.figure(f"Image convoluée par la PSF {mode} de manière naïve")
        plt.title(f"Image convoluée par la PSF {mode} de manière naïve")
        plt.imshow(image_sortie, cmap = "gist_gray")
        plt.clim(limites[0],limites[1])
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    
    print(f'Le temps pour convoluer avec la méthode naïve est de {time.time()-t} secondes')
    return image_sortie

def convoluer_PSF_en_mieux(image_entree, rayon = 8, mode = "Airy", fond_moyen = 0, limites = (0,1), plot = True):
    """
    Permet de calculer l'image après qu'elle soit passée par un instrument optique, qui, au premier ordre, réalise une convolution de l'image avec la PSF, à cause de la diffraction
    Cette version est optimisée d'un point de vue temps de calcul. La manière dont fonctionne cette convolution utilise la propriété suivante :
        Si on désigne par * la convolution et par . le produit 
        Soient f et g deux fonctions, et TF(f) et TF(g) leur transformée de Fourier respectives
        TF(f*g) = TF(f).TF(g)
        soit f*g = TF^-1(TF(f).TF(g))
        avec TF^-1 la transformée de Fourier inverse (voir quand elle existe, mais avec tes fonctions, tu n'auras jamais de souci, elle est toujours définie)
    
    Parameters
    ----------
    image_entree : np.array
        une image d'un objet qui n'a pas encore été imagé par un instrument
    rayon : float, optional
        un nombre strictement positif qui définit le rayon de la PSF. The default is 8.
    mode : str, optional
        plusieurs modes possible : 
            "Airy" : Si l'instrument est en limite de diffraction.
            "Focus": The default is "Airy".
    fond_moyen : float, optional
        valeur du fond moyen. The default is 0.
    limites : (min,max), optional
        limites de la dynamique de l'image. The default is (0,1).
    plot : Bool, optional
        Si True, on affiche les images. The default is True.

    Returns
    -------
    image_sortie : np.array
        image d'entrée convoluée par la PSF

    """
    t = time.time()
    
    # Calcule la taille de l'image d'entrée
    taille_x = np.shape(image_entree)[1] #taille de l'image selon x
    taille_y = np.shape(image_entree)[0] #taille de l'image selon y
    
    
    #Si on est en limite de diffraction
    if mode == "Airy":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((8*math.ceil(rayon)+1, 8*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)/rayon*1.219670
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = (2*scipy.special.jv(1,np.pi*r)/(np.pi*r))**2 if r else 1
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Airy").clf()
            except:
                pass
            plt.figure("PSF Airy")
            plt.title("PSF Airy")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    # si la mise au point n'est pas parfaite
    elif mode == "Focus":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((2*math.ceil(rayon)+1, 2*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2),int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)
                #pondérer les points du bords du cercle par l'intégrale de la portion de disque sur ce pixel
                if abs(r-rayon) < np.sqrt(2)/2:
                    value = 0
                    M = 10
                    for u in np.linspace(i-0.5,i+0.5, M):
                        for v in np.linspace(j-0.5, j+0.5, M):
                            if math.sqrt(u**2+v**2) < rayon:
                                value +=1
                    value = float(value)/(M**2)
                elif r > rayon:
                    value = 0
                elif r < rayon:
                    value = 1
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = value
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Focus").clf()
            except:
                pass
            plt.figure("PSF Focus")
            plt.title("PSF Focus")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    elif mode == "Araignée":
        # on construit la PSF (Point Spread Function)
        PSF = np.zeros((20*math.ceil(rayon)+1, 20*math.ceil(rayon)+1))
        N = PSF.shape[0]
        somme = 0
        angle_araignee = 10 #° [entre 0 et 89.9°]
        rayon_sinc = 0.2
        araignee = np.sinc(np.linspace(0, 1, int((N-1)/2)+1))**2
        for i in range(int(-(N-1)/2), int((N-1)/2 + 1)):
            for j in range(int(-(N-1)/2), int((N-1)/2 + 1)):
                r = math.sqrt(i**2+j**2)/rayon*1.219670
                PSF[i+int((N-1)/2),j+int((N-1)/2)] = (2*scipy.special.jv(1,np.pi*r)/(np.pi*r))**2 if r else 1
                M = 10
                valeur = 0
                a = np.tan(angle_araignee*np.pi/180)
                b = rayon_sinc
                a2 = -a
                if abs(a*j-i) <= b+0.5+0.5*a  or abs(a2*i-j) <= b+0.5+0.5*a2:
                    for u in np.linspace(j-0.5, j+0.5,M):
                        for v in np.linspace(i-0.5, i +0.5, M):
                            if abs(a*u-v) <=rayon_sinc:
                                valeur += 1
                            if abs(a2*v-u) <= rayon_sinc:
                                valeur += 1
                    valeur /= 2*M**2
                    R = math.sqrt(i**2+j**2)
                    if R <= (N-1)/2:
                        PSF[i+int((N-1)/2),j+int((N-1)/2)] += valeur*araignee[round(R)]
                
                somme += PSF[i+int((N-1)/2),j+int((N-1)/2)]
        # Normalisation (conservation de l'énergie oblige)
        PSF = PSF/somme
        if plot:
            try:
                plt.figure("PSF Araignée").clf()
            except:
                pass
            plt.figure("PSF Araignée")
            plt.title("PSF Araignée")
            plt.imshow(PSF, cmap = "gist_gray")
            # plt.clim(0,1)
            plt.colorbar()
            plt.axis("equal")
            plt.show()
    
    else:
        print("Ce mode n'est pas pris en compte pour le moment")
        raise NameError(f"le mode {mode} n'est pas encore défini")
    
    # on réalise la convolution 2D avec la fonction de scipy.signal
    image_sortie = scipy.signal.convolve2d(image_entree,PSF,mode = "same", boundary="fill", fillvalue=fond_moyen)
    
    # on corrige l'exposition de l'image, en se disant que le max de l'image d'entrée doit correspondre au max de l'image de sortie
    maximum_image_entree = np.amax(image_entree)
    maximum_image_sortie = np.amax(image_sortie)
    image_sortie *= maximum_image_entree/maximum_image_sortie
    
    if plot:
        try:
            plt.figure(f"Image convoluée par la PSF {mode} en mieux").clf()
        except:
            pass
        plt.figure(f"Image convoluée par la PSF {mode} en mieux")
        plt.title(f"Image convoluée par la PSF {mode} en mieux")
        plt.imshow(image_sortie, cmap = "gist_gray")
        plt.clim(limites[0],limites[1])
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    
    print(f"Le temps pour convoluer l'iamge avec la PSF de manière optimisée est de {time.time()-t} secondes")
    return image_sortie

def comparer_images(image1, image2, plot = True):
    t = time.time()
    
    image_sortie = image1-image2
    if plot:
        try:
            plt.figure("différence").clf()
        except:
            pass
        plt.figure("différence")
        plt.title("Différence")
        plt.imshow(image_sortie, cmap = "gist_gray")
        plt.colorbar()
        plt.axis("equal")
        plt.show()
    
    print(f'Le temps pour comparer les images est de {time.time()-t} secondes')
    return image_sortie