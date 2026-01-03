# CSC8607 Deep Learning Project — AG_NEWS × TextCNN

Classification de texte sur le dataset AG_NEWS avec un modèle TextCNN.

##  Description du projet

- **Dataset** : AG_NEWS (4 classes : World, Sports, Business, Science/Technology)
- **Modèle** : TextCNN avec embeddings GloVe pré-entraînés
- **Performance** : ~91.5% accuracy sur le test set

##  Installation rapide

### Windows (PowerShell)
```powershell
# Exécuter le script d'installation
.\setup_env.bat
```

### Linux/macOS
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

##  Structure du projet

```
csc8607_projects/
├── configs/
│   └── config.yaml          # Configuration principale
├── src/
│   ├── __init__.py          # Package init
│   ├── data_loading.py      # Chargement des données
│   ├── model.py             # Architecture TextCNN
│   ├── train.py             # Script d'entraînement
│   ├── evaluate.py          # Évaluation du modèle
│   ├── lr_finder.py         # Learning rate finder
│   ├── grid_search.py       # Recherche d'hyperparamètres
│   ├── preprocessing.py     # Prétraitements
│   ├── augmentation.py      # Augmentation de données
│   └── utils.py             # Utilitaires
├── artifacts/
│   └── best.ckpt            # Meilleur checkpoint
├── runs/                    # Logs TensorBoard
├── screenshots/             # Captures d'écran TensorBoard
├── report.md                # Rapport du projet
├── requirements.txt         # Dépendances Python
└── README.md                # Ce fichier
```

##  Commandes principales

### Entraînement
```bash
python -m src.train --config configs/config.yaml
```

### LR Finder (trouver le learning rate optimal)
```bash
python -m src.lr_finder --config configs/config.yaml
```

### Grid Search (recherche d'hyperparamètres)
```bash
python -m src.grid_search --config configs/config.yaml
```

### Évaluation
```bash
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
```

### Visualisation TensorBoard
```bash
tensorboard --logdir=runs
# Puis ouvrir http://localhost:6006 dans un navigateur
```

##  Hyperparamètres clés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `embed_dim` | 200 | Dimension des embeddings |
| `num_filters` | 100 | Nombre de filtres CNN par taille de kernel |
| `kernel_sizes` | [3, 4, 5] | Tailles des fenêtres de convolution |
| `dropout` | 0.5 | Taux de dropout |
| `learning_rate` | 0.001 | Taux d'apprentissage |
| `batch_size` | 64 | Taille des batches |

##  Résultats

| Métrique | Valeur |
|----------|--------|
| Test Accuracy | 91.54% |
| Macro F1-Score | 0.9154 |
| Best Val Accuracy | 90.23% |

##  Rapport

Le rapport complet est disponible dans [`report.md`](report.md).

## 🔗 Références

- [TextCNN (Kim, 2014)](https://arxiv.org/abs/1408.5882)
- [AG_NEWS Dataset](https://pytorch.org/text/stable/datasets.html#ag-news)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
