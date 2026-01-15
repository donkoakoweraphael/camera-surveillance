# Projet : Système de Surveillance Multi-Caméra Intelligent

Ce projet implémente un système de surveillance MVP (Minimum Viable Product) capable de traiter simultanément plusieurs flux vidéo pour la détection d'objets, le suivi de trajectoire et la gestion de zones d'alerte.

## Fonctionnalités Clés
-   **Détection Multi-Objets** : Personnes, Sacs à dos, Sacs à main, Bouteilles, Téléphones.
-   **Suivi (Tracking)** : Algorithme SORT amélioré, gestion des trajectoires.
-   **Zones d'Alerte** : Définition de zones polygonales, alerte visuelle (rouge) et logs.
-   **Synchronisation Temporelle** : Alignement parfait des vidéos basé sur les métadonnées ou la configuration.
-   **Moniteur Haute Résolution** : Affichage grille sans perte de qualité.

---

# 🚀 Guide de Déploiement et d'Utilisation

Ce guide détaille les étapes pour installer et lancer le système sur Linux, Windows ou macOS.

## 1. Prérequis
-   **Python 3.8+** installé.
-   **Git** installé.
-   Un terminal (Bash, PowerShell, ou CMD).

## 2. Installation

1.  **Cloner le dépôt** :
    ```bash
    git clone https://github.com/donkoakoweraphael/camera-surveillance.git
    cd camera-surveillance
    ```

2.  **Mise en place de l'environnement virtuel** :
    *   **Linux / macOS** :
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows** :
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate
        ```

3.  **Installation des dépendances** :
    ```bash
    pip install -r requirements.txt
    ```
    *Note : Le projet utilise `ultralytics` pour YOLO et `opencv-python`.*

## 3. Configuration des Données

1.  **Vidéos** : Placez vos fichiers vidéo dans le dossier `videos-camera/`.
2.  **Configuration** : Le fichier `cams.json` contrôle tout.
    *   Assurez-vous que les chemins (`file`) correspondent à vos vidéos.
    *   Vous pouvez définir manuellement le `start_time` (Format ISO `YYYY-MM-DDTHH:MM:SS`) pour la synchronisation.
    *   Champs importants : `rotate` (90, -90), `zones` (points du polygone).

## 4. Outils de Configuration (Optionnel)

Pour définir les zones d'alerte graphiquement :
```bash
python zone_selector.py
```
*   Cliquez pour dessiner un polygone.
*   Appuyez sur `s` pour sauvegarder.
*   Appuyez sur `q` pour passer à la caméra suivante.

## 5. Exécution du Traitement (Pipeline)

Le cœur du système. Il lit les vidéos, détecte, track, et génère les vidéos annotées dans `output/`.

```bash
python pipeline.py
```
*   Attendre la fin du traitement (Barre de progression dans le terminal).
*   **Performance** : Configuré à 2 FPS (Frames Par Seconde) pour optimiser le temps de traitement CPU.

## 6. Visionnage (Moniteur)

Pour visualiser le résultat synchronisé dans une grille haute résolution :

```bash
python monitor.py
```
*   **Contrôles** :
    *   `Espace` : Pause / Lecture.
    *   `Flèche Gauche` : Reculer.
    *   `Flèche Droite` : Avancer.
    *   `q` : Quitter.

---

## Structure du Projet

```
.
├── cams.json           # Configuration des caméras et zones
├── pipeline.py         # Script principal de traitement
├── monitor.py          # Interface de visionnage
├── zone_selector.py    # Outil de dessin de zones
├── sort.py             # Algorithme de tracking
├── requirements.txt    # Dépendances Python
├── output/             # Vidéos générées et logs (alerts.log)
└── videos-camera/      # (Non inclus dans git) Dossier des sources
```

## Auteur
Akowé Raphaël DONKO