# Classificação Automática de Imagens de CPRE via Deep Learning

**Aprendizagem Profunda – Módulo 2 | Grupo 4 | MECD 2025/2026**  
Universidade do Minho

**Grupo:**
- Filipa Oliveira da Silva (PG60004)
- Francisco Ricardo Teixeira Silva (PG60099)
- Lara Cristiana Sousa Antunes (PG58750)
- Marco Scianna (e12961)

---

## Contexto

Trabalho prático para classificação automática de imagens fluoroscópicas de CPRE em 4 classes: **Biliary_Leaks**, **Lithiasis**, **Normal** e **Stricture**, usando o dataset [MIQR-CC](https://doi.org/10.6084/m9.figshare.31079236).

Baseline de referência: **F1-macro = 0.738**  
Melhor resultado obtido: **F1-macro = 0.6757** (Ensemble)

---

## Estrutura

```
AP-M2_G4_MECD/
├── notebooks/                        # Notebooks finais (executar por ordem)
│   ├── 01_preprocessing_improved.ipynb
│   ├── 02_train_efficientnet_improved.ipynb
│   ├── 03_autoencoder_xgboost.ipynb
│   ├── 04_ensemble_gradcam.ipynb
│   ├── class_weights.json            # Gerado pelo notebook 01
│   ├── final_results.json            # Gerado pelo notebook 04
│   ├── test_labels_best2.npy         # Gerado pelo notebook 02
│   ├── test_probs_efficientnet_best2.npy
│   ├── test_probs_xgboost_best2.npy
│   └── test_probs_svm_best2.npy
├── notebooks_history/                # Versões intermédias do desenvolvimento
├── results/
│   ├── best/                         # Resultados finais (versão best2)
│   └── (versões anteriores para referência)
├── README.md
└── requirements.txt
```

> As pastas `models/`, `dataset_processed/` e `dataset_PM/` não estão no repositório por serem demasiado grandes. São geradas localmente ao correr os notebooks.

---

## Instalação

```bash
pip install -r requirements.txt
```

Python 3.10 | GPU recomendada (mínimo 8GB VRAM para o EfficientNet-B7)

---

## Como Executar

O dataset deve estar disponível localmente. Alterar a variável `DATASET_DIR` no início de cada notebook para apontar para a pasta correta:

```
dataset/
├── train/
│   ├── Biliary_Leaks/
│   ├── Lithiasis/
│   ├── Normal/
│   └── Stricture/
├── val/
└── test/
```

Executar os notebooks **pela ordem indicada** — cada um gera ficheiros usados pelo seguinte:

| # | Notebook | O que faz | Output principal |
|---|---|---|---|
| 01 | `01_preprocessing_improved.ipynb` | Segmentação + CLAHE + resize 384×384 | `dataset_processed/`, `class_weights.json` |
| 02 | `02_train_efficientnet_improved.ipynb` | Treino EfficientNet-B7 | `models/efficientnet_b7_best.pth`, `test_probs_efficientnet.npy` |
| 03 | `03_autoencoder_xgboost.ipynb` | Autoencoder + XGBoost + SVM | `models/autoencoder.pth`, `test_probs_xgboost.npy`, `test_probs_svm.npy` |
| 04 | `04_ensemble_gradcam.ipynb` | Ensemble + Grad-CAM | `final_results.json`, imagens Grad-CAM |

---

## Resultados

| Modelo | F1-macro |
|---|---|
| Baseline (artigo) | 0.7380 |
| ResNet50 | 0.5392 |
| XGBoost (features autoencoder) | 0.2391 |
| SVM (features autoencoder) | 0.3009 |
| EfficientNet-B7 | 0.6688 |
| **Ensemble (B7 + XGBoost + SVM)** | **0.6757** |

---

## Referências

- Dataset: https://doi.org/10.6084/m9.figshare.31079236
- Código baseline: https://github.com/monicaccmartins/MIQR-CC-Dataset
