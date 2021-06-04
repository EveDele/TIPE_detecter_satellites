# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:32:16 2021

@author: Utilisateur
"""
from fonctions_images_synthetiques_1 import generer_fond_moyen,ajouter_etoiles, ajouter_trainees, convoluer_PSF

def ciel(shape,denstar,mimanb,psf=0,kern=1,mimaint=(0.5,1),mimalen=(0.2,0.7),fond=0.01):
    hauteur = shape[0]
    largeur = shape[0] 
    minimum_image = 0 #correspond à un flux nul sur le pixel
    maximum_image = 1 #correspond à un flux maximal sur le pixel
    limites_images = (minimum_image, maximum_image)
    
    
    # paramètres de l'image
    fond_moyen = fond # un fond moyen de 1% de la dynamique du capteur. C'est le bon ordre de grandeur avec de la pollution lumineuse ou la lune dans le ciel.
    densite_etoiles = denstar # je dirais qu'il est compris entre 0.005 et 0.1 /!\si tu montes trop, le temps de calcul est assez élevé.
    intervalle_nbre_trainees = mimanb # le nombre de trainées dans une image est compris entre ces deux valeur
    intervalle_intensite_trainees = mimaint #l'intensité des trainées est comprise entre ces deux valeurs
    intervalle_longeur_trainees = mimalen #la longeur des trainées est comprise entre ces deux valeurs. 1 correspond à la diagonale de l'image
    
    #Gestion de l'aléatoire
    noyau_aleatoire = kern
    # Cela permet de générer exactement les mêmes images, si on garde la même valeur, et le même ordinateur
    # Si il n'est pas passé comme argument aux fonctions qui utilisent de l'aléatoire, le noyau qui est pris se base
    # sur la date et l'heure de l'ordinateur, ce qui garantit d'avoir un résultat différent à chaque fois
    
    
    
    # Génération du fond moyen
    I_fond_moyen = generer_fond_moyen(largeur, hauteur, fond_moyen)
    
    # Ajouter les étoiles
    I_etoiles = ajouter_etoiles(I_fond_moyen, densite_etoiles, fond_moyen, maximum_image, noyau_aleatoire = noyau_aleatoire)
    
        # Ajouter les trainées
    I_trainees = ajouter_trainees(I_etoiles,
                              intervalle_nbre_trainees,
                              intervalle_longeur_trainees,
                              intervalle_intensite_trainees,
                              noyau_aleatoire = noyau_aleatoire)
    if psf:
        I_trainees = convoluer_PSF(I_trainees, 2.1, 'Airy', fond_moyen)
    return I_trainees