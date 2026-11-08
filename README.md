# Détection d'appareils inconnus en NILM par autoencodeurs (UK-DALE)

Ce dépôt compare trois autoencodeurs — **USAD**, **VAE** et **GRU-AE** — sur une tâche de **détection de nouveauté** en désagrégation de charge (NILM) : un modèle entraîné uniquement sur des appareils *connus* peut-il repérer, via son erreur de reconstruction, l'apparition d'un **appareil inconnu** dans le signal agrégé ?

Les trois notebooks suivent **exactement le même protocole**. Seul le modèle (étape 4) change, ce qui rend les résultats directement comparables.

## Contenu du dépôt

| Notebook | Modèle | Particularité |
|---|---|---|
| `usad_nilm_novelty_detection.ipynb` | **USAD** (Audibert et al., KDD 2020) | Encodeur partagé, deux décodeurs, entraînement adversarial en deux phases |
| `vae_nilm_novelty_detection.ipynb` | **VAE** | Goulot d'étranglement probabiliste (`mu` / `logvar`, reparamétrisation, terme KL), décodeur unique |
| `gruae_nilm_novelty_detection.ipynb` | **GRU-AE** | Encodeur et décodeur récurrents (GRU), décodeur unique, perte MSE simple |

La variable de comparaison est donc la **famille d'architecture** : dense à double décodeur adversarial (USAD), dense probabiliste (VAE), récurrent (GRU-AE).

## Protocole

1. **Données** : UK-DALE, maison 1, lues avec [NILMTK](https://github.com/nilmtk/nilmtk).
2. **Agrégat synthétique** : somme de la puissance *réelle* (sous-compteurs) des appareils connus uniquement — jamais le compteur général de la maison.
3. **Fenêtrage** : fenêtres glissantes (`WINDOW_SIZE = 200` échantillons, `STEP = 5`, pas d'échantillonnage de 6 s), normalisation z-score ajustée sur les fenêtres d'entraînement.
4. **Entraînement** : autoencodeur sur fenêtres « appareils connus » uniquement (50 époques, batch de 1024, espace latent de dimension 14).
5. **Score d'anomalie** : erreur de reconstruction par fenêtre.
   - USAD : `0,3 · MSE(w1) + 0,7 · MSE(w2)`
   - VAE : MSE de reconstruction décodée de façon déterministe depuis `mu` (le terme KL est exclu du score)
   - GRU-AE : MSE de reconstruction
6. **Test progressif** : injection d'un appareil nouveau à la fois dans une période de test disjointe de l'entraînement, puis calcul du ROC-AUC, du PR-AUC et de deux seuils opérationnels.

### Configuration par défaut

| Paramètre | Valeur |
|---|---|
| Appareils connus | `fridge freezer`, `washing machine`, `dish washer` |
| Nouvel appareil « faible puissance » | `microwave` (micro-ondes) |
| Nouvel appareil « forte puissance » | `kettle` (bouilloire) |
| Période d'entraînement | 2013-04-01 → 2014-05-31 |
| Période de test | 2014-07-01 → 2014-08-31 |
| Fraction de validation | 15 % |
| Graine aléatoire | 42 |

Les périodes d'entraînement et de test sont volontairement **disjointes** pour éviter toute fuite de données.

### Seuils de décision

- **Seuil 1 (ROC / indice de Youden)** : point de la courbe ROC maximisant `TPR − FPR`, comme dans l'article USAD.
- **Seuil 2 (optimal pour le F1)** : point de la courbe précision-rappel maximisant le F1-score.

Les deux sont calculés sur les fenêtres de test étiquetées, tous niveaux d'appareils confondus.

### Test séquentiel par période

La période de test est découpée en trois sous-périodes égales : (1) agrégat connu seul, (2) + appareil faible puissance, (3) + appareil forte puissance. Les notebooks tracent le signal en watts avec les zones détectées comme anormales, ainsi que des diagnostics détaillés (score, état de l'appareil injecté, entrée vs reconstruction).

## Résultats (run de référence)

**ROC-AUC / PR-AUC par appareil nouveau** (environ 6 % de fenêtres positives) :

| Modèle | Micro-ondes (faible) ROC-AUC | Micro-ondes PR-AUC | Bouilloire (forte) ROC-AUC | Bouilloire PR-AUC | ROC-AUC poolé |
|---|---|---|---|---|---|
| USAD | 0,964 | 0,457 | 0,977 | 0,509 | 0,970 |
| VAE | 0,972 | 0,573 | **0,989** | **0,815** | **0,981** |
| GRU-AE | **0,978** | **0,659** | 0,985 | 0,763 | **0,981** |

**Points de fonctionnement sur les données de test poolées :**

| Modèle | Seuil 1 : FPR | Seuil 1 : TPR | Seuil 2 : FPR | Seuil 2 : TPR | Seuil 2 : F1 |
|---|---|---|---|---|---|
| USAD | 0,054 | 0,986 | 0,036 | 0,924 | 0,743 |
| VAE | 0,079 | 0,989 | 0,038 | 0,876 | 0,711 |
| GRU-AE | 0,098 | 0,995 | 0,026 | 0,808 | 0,730 |

Les trois modèles détectent bien l'appareil nouveau (ROC-AUC > 0,96). Le cas « faible puissance » est le plus difficile, comme attendu. Le PR-AUC reste modeste car seulement ~6 % des fenêtres sont positives : il doit être lu par rapport à ce taux de base.

> Un nouveau run peut donner des valeurs légèrement différentes (non-déterminisme GPU).

## Installation

### Prérequis

- Python 3.9+
- UK-DALE converti au format HDF5 pour NILMTK (`ukdale.h5`)
- PyTorch, NumPy, pandas, scikit-learn, matplotlib
- `nilmtk` et `nilm_metadata`, installés depuis GitHub (sans leurs dépendances figées) :

```bash
pip install -q --no-deps git+https://github.com/nilmtk/nilm_metadata.git
pip install -q --no-deps git+https://github.com/nilmtk/nilmtk.git
```

Un GPU accélère l'entraînement mais reste optionnel.

### Variables d'environnement

| Variable | Rôle | Défaut |
|---|---|---|
| `UKDALE_H5_PATH` | Chemin vers `ukdale.h5` | chemin Kaggle si présent, sinon `data/ukdale.h5` |
| `OUTPUT_DIR` | Dossier de sortie (modèles, figures) | `/kaggle/working` sur Kaggle, sinon `outputs/` |

### Exécution

```bash
export UKDALE_H5_PATH=/chemin/vers/ukdale.h5
jupyter notebook
```

Ouvrir puis exécuter l'un des trois notebooks de haut en bas. Les poids entraînés sont sauvegardés dans `OUTPUT_DIR` (`usad_nilm_model.pth`, `vae_nilm_model.pth`, `gruae_nilm_model.pth`).

**Sur Kaggle** : attacher le dataset UK-DALE, activer Internet (pour installer `nilmtk`) et, si souhaité, un GPU.

## Limites à garder en tête

- **Seuils choisis sur des données de test étiquetées.** ROC-AUC et PR-AUC ne dépendent pas d'un seuil, mais FPR, TPR, F1 et accuracy aux seuils 1 et 2 sont **optimistes**. Un protocole plus rigoureux sélectionnerait le seuil sur une période de validation étiquetée séparée.
- **Validation non indépendante.** Le découpage train/validation est une permutation aléatoire de fenêtres glissantes très chevauchantes (200 échantillons, pas de 5) : la perte de validation n'estime donc pas fidèlement la généralisation.
- **Cadre étroit.** Une seule maison, un agrégat synthétique de trois appareils connus, seulement deux appareils nouveaux, et des seuils ON choisis à la main (frigo-congélateur 50 W, lave-linge 20 W, lave-vaisselle 20 W, bouilloire 2000 W, micro-ondes 200 W).
- **Un seul run par modèle.** Une graine, un entraînement : les écarts d'environ 0,01 de ROC-AUC entre modèles peuvent ne pas être significatifs.
- **Particularité du VAE.** Dans `reparameterize`, l'écart-type est calculé avec `exp(0.05 · logvar)` au lieu du `exp(0.5 · logvar)` standard supposé par le terme KL. Les résultats ci-dessus ont été obtenus avec 0,05 (`KL_WEIGHT = 0.01`) ; passer à 0,5 donne un VAE standard et nécessite de relancer l'expérience.

## Pistes d'amélioration

- Sélectionner les seuils sur une période de validation dédiée
- Répéter les expériences sur plusieurs graines et plusieurs maisons UK-DALE
- Ajouter d'autres appareils nouveaux et d'autres combinaisons d'appareils connus
- Découper train/validation par blocs temporels plutôt que par permutation aléatoire

## Références

- Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. A. **USAD: UnSupervised Anomaly Detection on Multivariate Time Series**. KDD 2020.
- Kelly, J., Knottenbelt, W. **The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes**. Scientific Data, 2015.
- Batra, N. et al. **NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring**. e-Energy 2014.

## Licence

À compléter (par exemple MIT). Le jeu de données UK-DALE est soumis à sa propre licence : consulter les conditions d'utilisation du dataset.
