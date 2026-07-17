# Navigation Autonome d'un Robot Auto-Équilibré par Vision Artificielle (UGOT)

Projet de Fin d'Année — HESTIM, Filière IIIA (S4)
Réalisé par **MESMEY Aime Ange** — Encadrant académique : **M. TULA Sridath**

## 🎯 Objectif

Faire naviguer de façon **entièrement autonome** un robot auto-équilibré **UGOT**, du point A vers un point B puis retour au point A, dans une zone de patrouille rectangulaire de **120 × 180 cm**, **sans aucun capteur de distance** (pas de LiDAR, pas d'ultrasons, pas de ToF) — uniquement à partir de la caméra embarquée et de la vision artificielle.

## 🧩 Contraintes du projet

- Aucun capteur de distance additionnel disponible/exploitable
- Localisation uniquement par odométrie (dead-reckoning) + gyroscope
- Détection, décision et pilotage moteur en temps réel, embarqués sur le robot
- Patrouille en boucle fermée avec retour précis au point de départ

## 🛠️ Matériel & environnement logiciel

**Matériel**
- Robot UGOT auto-équilibré (IP Wi-Fi : `172.16.0.68`)
- Caméra embarquée intégrée
- Gyroscope (estimation d'orientation)

**Logiciel**
- Python + SDK UGOT officiel
- OpenCV — traitement d'image & homographie
- YOLOv8 nano (`yolov8n.pt`, Ultralytics) — détection d'objets/humains
- NumPy — calcul matriciel

## 🧠 Architecture

Le contrôleur est structuré en classes, chacune avec une responsabilité isolée :

| Classe | Rôle |
|---|---|
| `Config` | Paramètres globaux (seuils, dimensions de la zone, IP robot…) |
| `Odometry` | Estimation de position par dead-reckoning + gyroscope |
| `Map2D` | Cartographie 2D en temps réel, trajectoire parcourue |
| `ObstacleDetector` | Détection d'obstacles par vision (seuillage adaptatif, Canny, filtres morphologiques) |
| `HumanDetector` | Détection humaine (YOLOv8), priorité absolue dans la machine d'états |
| `SirenLED` | Signalisation lumineuse/sonore du robot |
| `MissionController` | Machine d'états : orchestre patrouille, évitement, détection humaine et recalibration |

Pipeline global : **Vision (caméra) → Détection (obstacles/humains) → Décision (machine d'états) → Commande (moteurs)**, avec une hiérarchie de priorités continue : *détection humaine > obstacle critique > obstacle normal > poursuite de la mission*.

## 👁️ Perception

- **Seuillage adaptatif** : le seuil de luminosité s'ajuste à l'histogramme local de l'image pour rester fiable malgré les variations d'éclairage.
- **Contours (Canny)** + **filtres morphologiques** (dilatation/érosion) pour nettoyer les images binaires avant décision.
- **Matrice de détection 2×3** : le champ de vision est simplifié en Humain/Objet × Gauche/Centre/Droite ; une case s'active dès qu'un objet est repéré à moins de 50 cm.
- **Homographie** : conversion pixels → centimètres via 4 points sol calibrés (`cv2.findHomography`), pour estimer une distance réelle fiable (précision ±15 cm latéral, 15–40 cm en profondeur).
- **Vidage du buffer caméra** après chaque manœuvre, pour décider sur une image du présent plutôt que sur une scène dépassée.

## 📍 Localisation & cartographie

- Position estimée par **dead-reckoning** (vitesse × temps écoulé), recalibrée à chaque waypoint en fusionnant avec des repères visuels, pour compenser la dérive sur les trajets longs.
- **Cap (heading)** estimé via le gyroscope du robot plutôt que par simple intégration d'angle.
- **Carte 2D** mise à jour en continu (mini-carte de debug en temps réel).

## 🚧 Le défi central — pourquoi c'était difficile

La première version du contrôleur regroupait odométrie, vision, évitement et pilotage moteur dans **une seule boucle** :
- Aucune brique ne pouvait compenser l'erreur d'une autre
- La détection avait une **priorité rigide** : elle interrompait la marche planifiée sans jamais réintégrer proprement le trajet
- Sans itinéraire prédéfini, le robot **sortait du rectangle de patrouille** après plusieurs évitements consécutifs
- Il lui arrivait de **tourner sur lui-même** faute de notion fiable de position
- **Aucune récupération** : en l'absence de passage libre, le robot restait bloqué ou répétait indéfiniment le même mouvement

## ✅ Solution retenue

- **Architecture orientée classes** (voir ci-dessus) : logique claire, responsabilités isolées, boucle qui referme exactement le trajet sur le point A
- **Logique d'évitement stricte** : déviation uniquement si un obstacle est à la fois dans le couloir direct (±15 cm) et à moins de 7 cm, avec **déviation biaisée vers l'intérieur** de la zone et **récupération automatique anti-blocage**
- **Sécurité de zone** basée sur l'odométrie brute pour détecter et corriger les sorties de zone
- **Recalibration aux waypoints** combinant dead-reckoning et repères visuels

## 📅 Planification

Projet mené sur 5 semaines + rédaction du rapport, 100 % livré dans les délais :

1. Mise en route & équilibrage
2. Détection d'obstacles
3. Navigation autonome
4. Cartographie 2D
5. Navigation A → B → A

## 🔭 Perspectives

- **Court terme** : calibration automatique des seuils de détection selon la luminosité ambiante
- **Moyen terme** : ajout d'un capteur de profondeur pour fiabiliser les distances estimées
- **Long terme** : cartographie SLAM pour explorer des zones plus grandes

## 🏁 Conclusion

Ce projet a permis de concevoir, développer et valider un système de navigation autonome complet reposant exclusivement sur la vision artificielle — de la détection d'obstacles à la navigation point à point avec retour au point de départ.

---

*HESTIM — Filière IIIA — S4 — Année universitaire 2025–2026*
