# Classificação Automática de Imagens Fluoroscópicas de CPRE via Deep Learning

**Aprendizagem Profunda – Módulo 2 | Grupo 4 | MECD 2025/2026**  
Universidade do Minho

| Elemento | Nº |
|---|---|
| Filipa Oliveira da Silva | PG60004 |
| Francisco Ricardo Teixeira Silva | PG60099 |
| Lara Cristiana Sousa Antunes | PG58750 |
| Marco Scianna | e12961 |

---

## Descrição

Solução de Deep Learning para classificação automática de imagens fluoroscópicas de CPRE (Colangiopancreatografia Retrógrada Endoscópica) em 4 classes patológicas:

- **Biliary_Leaks** — fugas de bílis
- **Lithiasis** — cálculos biliares
- **Normal** — sem patologia
- **Stricture** — estenose dos ductos

Dataset utilizado: [MIQR-CC (Figshare)](https://doi.org/10.6084/m9.figshare.31079236)  
Baseline de referência: F1-macro = **0.738**  
Melhor resultado obtido: F1-macro = **0.6757** (Ensemble)

---

## Estrutura do Repositório

```
AP-M2_G4_MECD/
├── notebooks/
│   ├── 01_preprocessing_improved.ipynb   # Pré-processamento (CLAHE + segmentação)
│   ├── 02_train_efficientnet_improved.ipynb  # Treino EfficientNet-B7
│   ├── 03_autoencoder_xgboost.ipynb      # Autoencoder + XGBoost/SVM
│   └── 04_ensemble_gradcam.ipynb         # Ensemble + Grad-CAM
├── results/                              # Figuras geradas pelos notebooks
│   ├── class_distribution.png
│   ├── confusion_matrix_efficientnet_best2.png
│   ├── confusion_matrix_ensemble_best2.png
│   ├── confusion_matrix_xgboost_best2.png
│   ├── gradcam_examples_best2.png
│   ├── gradcam_errors_best2.png
│   ├── autoencoder_reconstructions_best2.png
│   └── autoencoder_loss_best2.png
├── README.md
└── requirements.txt
```

> **Nota:** As pastas `models/` e `dataset_processed/` não estão incluídas no repositório por limitações de tamanho. Os modelos treinados e o dataset processado são gerados localmente ao executar os notebooks pela ordem indicada.

---

## Requisitos

### Hardware
- GPU com pelo menos **8GB VRAM** recomendada (os modelos foram treinados numa máquina com GPU NVIDIA)
- Para o EfficientNet-B7 com batch_size=4 e imagens 384×384, são necessários ~6GB VRAM

### Software

```bash
pip install -r requirements.txt
```

Versão de Python testada: **3.10**

---

## Configuração do Dataset

1. Descarregar o dataset MIQR-CC:
   ```bash
   wget https://figshare.com/ndownloader/files/61063177 -O dataset.zip
   unzip dataset.zip
   ```

2. O dataset deve estar disponível no servidor com a estrutura:
   ```
   /mounts/monica/ERCP_image_classification/
   ├── dataset/
   │   ├── train/
   │   │   ├── Biliary_Leaks/
   │   │   ├── Lithiasis/
   │   │   ├── Normal/
   │   │   └── Stricture/
   │   ├── val/   (mesma estrutura)
   │   └── test/  (mesma estrutura)
   └── metadata.csv
   ```

3. Definir a variável de ambiente com o caminho do dataset:
   ```bash
   export DATASET_DIR=/mounts/monica/ERCP_image_classification/dataset
   ```
   Se o dataset estiver noutro caminho, alterar a variável `DATASET_DIR` no início de cada notebook.

---

## Ordem de Execução

Os notebooks devem ser executados **pela ordem indicada**. Cada notebook gera ficheiros (modelos `.pth`, probabilidades `.npy`) que são consumidos pelo notebook seguinte.

### Notebook 01 — Pré-processamento

```
notebooks/01_preprocessing_improved.ipynb
```

**O que faz:**
- Aplica segmentação fluoroscópica (Canny + Hough Transform) para remover bordas do equipamento
- Aplica CLAHE leve (clip_limit=1.5) para realçar contraste sem destruir detalhes finos
- Redimensiona para 384×384
- Calcula e guarda class weights balanceados

**Output gerado:**
- `dataset_processed/` — imagens processadas organizadas em train/val/test por classe
- `class_weights.json` — pesos de cada classe para o treino

**Tempo estimado:** 15–30 minutos (dependendo do hardware)

---

### Notebook 02 — Treino EfficientNet-B7

```
notebooks/02_train_efficientnet_improved.ipynb
```

**O que faz:**
- Carrega o dataset processado pelo notebook 01
- Treina EfficientNet-B7 com transfer learning (ImageNet)
- Aplica FocalLoss com class weights + WeightedRandomSampler
- Data augmentation conservador (rotação, flip, zoom, contraste, ruído)
- Early stopping com paciência 10 épocas
- Avalia no conjunto de teste e gera matrizes de confusão e curvas ROC

**Ficheiros necessários:** `dataset_processed/`, `class_weights.json`

**Output gerado:**
- `models/efficientnet_b7_best.pth` — melhor modelo salvo
- `test_probs_efficientnet.npy` — probabilidades no teste (para ensemble)
- `test_labels.npy` — labels do conjunto de teste
- `confusion_matrix_efficientnet_best2.png`
- `training_curves.png`
- `roc_curves_efficientnet.png`

**Hiperparâmetros principais:**

| Parâmetro | Valor |
|---|---|
| Optimizador | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 4 |
| Épocas máx. | 60 |
| Early Stopping | paciência 10 |
| Gradient Clip | 1.0 |

**Tempo estimado:** 2–4 horas (60 épocas com GPU)

**Resultado obtido:** F1-macro teste = **0.6688**

---

### Notebook 03 — Autoencoder + XGBoost/SVM

```
notebooks/03_autoencoder_xgboost.ipynb
```

**O que faz:**
- Treina autoencoder convolucional em todas as imagens do dataset (labeled + unlabelled, ~3138 imagens)
- Extrai features de 512 dimensões pelo encoder
- Aplica PCA (256 componentes, 99.9% da variância)
- Treina XGBoost e SVM nas features extraídas

**Ficheiros necessários:** `dataset_processed/`, `class_weights.json`

**Output gerado:**
- `models/autoencoder.pth`
- `models/xgboost.pkl`
- `models/svm.pkl`
- `models/scaler.pkl`, `models/pca.pkl`
- `test_probs_xgboost.npy`
- `test_probs_svm.npy`
- `autoencoder_reconstructions_best2.png`
- `autoencoder_loss_best2.png`
- `confusion_matrix_xgboost_best2.png`

**Hiperparâmetros principais:**

| Parâmetro | Autoencoder | XGBoost | SVM |
|---|---|---|---|
| Input size | 128×128 | — | — |
| Latent dim | 512 | — | — |
| LR / C | 1e-3 | 0.05 | 10 |
| Épocas / estimadores | 30 | 300 | — |
| Kernel | — | hist | RBF |

**Tempo estimado:** 1–2 horas

**Resultado obtido:** XGBoost F1-macro = **0.2391** | SVM F1-macro = **0.3009**

---

### Notebook 04 — Ensemble + Grad-CAM

```
notebooks/04_ensemble_gradcam.ipynb
```

**O que faz:**
- Carrega as probabilidades dos notebooks anteriores
- Faz grid search de pesos para o ensemble (B7 + XGBoost + SVM)
- Gera mapas Grad-CAM para imagens correctamente classificadas e erros
- Produz sumário final com todos os resultados

**Ficheiros necessários:**
- `test_probs_efficientnet.npy`
- `test_probs_xgboost.npy`
- `test_probs_svm.npy`
- `test_labels.npy`
- `models/efficientnet_b7_best.pth`
- `class_weights.json`

**Output gerado:**
- `confusion_matrix_ensemble_best2.png`
- `gradcam_examples_best2.png`
- `gradcam_errors_best2.png`
- `final_results.json`

**Pesos óptimos encontrados:** B7 = 0.5, XGBoost = 0.4, SVM = 0.1

**Resultado obtido:** Ensemble F1-macro = **0.6757**

---

## Resultados Finais

| Modelo | F1-macro (teste) | Δ vs Baseline |
|---|---|---|
| Baseline (artigo) | 0.7380 | — |
| ResNet50 | 0.5392 | −0.1988 |
| XGBoost (AE features) | 0.2391 | −0.4989 |
| SVM (AE features) | 0.3009 | −0.4371 |
| EfficientNet-B7 | 0.6688 | −0.0692 |
| **Ensemble (B7+XGB+SVM)** | **0.6757** | **−0.0623** |

### Resultados por Classe — Ensemble

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Biliary_Leaks | 0.7778 | 0.4118 | 0.5385 | 17 |
| Lithiasis | 0.7234 | 0.8293 | 0.7727 | 123 |
| Normal | 0.5410 | 0.7674 | 0.6346 | 43 |
| Stricture | 0.9464 | 0.6310 | 0.7571 | 84 |
| **Macro avg** | **0.7471** | **0.6599** | **0.6757** | 267 |

---

## Notas de Reprodutibilidade

- A seed está fixada a `42` em todos os notebooks (`random`, `numpy`, `torch`)
- Os resultados podem variar ligeiramente entre execuções devido a não-determinismo em operações CUDA
- O grid search do ensemble é determinístico dado os ficheiros `.npy` de probabilidades
- O `BEST2_v5_02_train_efficientnet_improved.ipynb` (arquivo) corresponde ao notebook `02_train_efficientnet_improved.ipynb` — é a versão final usada para os resultados reportados

---

## Referências

- Dataset MIQR-CC: https://doi.org/10.6084/m9.figshare.31079236
- Repositório baseline: https://github.com/monicaccmartins/MIQR-CC-Dataset
- EfficientNet: Tan & Le, ICML 2019
- FocalLoss: Lin et al., ICCV 2017
- Grad-CAM: Selvaraju et al., ICCV 2017
