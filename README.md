# TER NLI — Évaluation, robustesse et interprétabilité des modèles de Natural Language Inference

## Présentation du projet
Ce dépôt contient l’ensemble des expérimentations réalisées dans le cadre du Travail Encadré de Recherche (TER) portant sur l’évaluation, la robustesse et l’interprétabilité des modèles de Natural Language Inference (NLI).

L’objectif principal de cette étude est d’analyser dans quelle mesure les architectures Transformers modernes sont capables de réaliser une véritable inférence logique entre deux phrases, et non simplement d’exploiter des biais statistiques présents dans les jeux de données.

Le projet s’articule autour de deux axes expérimentaux principaux :
* **Axe 1 — Impact de l’architecture :** Comparaison de plusieurs architectures pré-entraînées (BERT, RoBERTa, DeBERTa) dans un environnement d’apprentissage identique afin d’évaluer leur capacité de généralisation et leur robustesse face aux biais linguistiques.
* **Axe 2 — Impact des données d’entraînement :** Étude de l’influence du corpus d’apprentissage sur le comportement du modèle en comparant plusieurs variantes de BERT entraînées sur différents jeux de données (SNLI, MNLI et ANLI).

---

## Natural Language Inference (NLI)
La tâche de NLI consiste à déterminer la relation logique entre une prémisse (*premise*) et une hypothèse (*hypothesis*). Trois classes sont possibles :

| Classe | Description |
| :--- | :--- |
| **Entailment** | La prémisse implique logiquement l’hypothèse. |
| **Contradiction** | L’hypothèse est incompatible avec la prémisse. |
| **Neutral** | La prémisse ne permet pas de conclure. |

**Exemple :**
* **Premise :** *"A man is playing football."*
* **Hypothesis :** *"A person is practicing a sport."*
* **Label →** `Entailment`

---

## Objectifs scientifiques
Ce travail cherche principalement à répondre aux questions suivantes :
1. Les architectures modernes apprennent-elles réellement des relations logiques ?
2. Les performances élevées sur SNLI reflètent-elles une véritable compréhension sémantique ?
3. Les modèles exploitent-ils des raccourcis statistiques et heuristiques ?
4. Quel est l’impact de l’architecture sur la robustesse ?
5. Quel est l’impact du corpus d’entraînement sur la généralisation ?
6. Les méthodes d’interprétabilité permettent-elles d’identifyer les biais décisionnels ?

---

## Jeux de données utilisés
* **SNLI (Stanford Natural Language Inference) :** Dataset historique du NLI contenant environ 570k paires. Utilisé principalement pour l'entraînement principal des modèles et l'étude de l'impact de l'architecture.
* **MNLI (Multi-Genre Natural Language Inference) :** Version plus diversifiée couvrant plusieurs genres textuels (fiction, conversations, presse, etc.). Utilisé pour l’évaluation de la généralisation hors distribution.
* **HANS (Heuristic Analysis for NLI Systems) :** Benchmark diagnostique conçu pour détecter les heuristiques superficielles (*lexical overlap*, *subsequence*, *constituent*). Utilisé pour mesurer la robustesse structurelle.
* **ANLI (Adversarial Natural Language Inference) :** Corpus adversarial construit selon une boucle homme-machine où les annotateurs cherchent volontairement à tromper les modèles. Utilisé pour évaluer la robustesse adversariale maximale.

---

## Architectures étudiées
* **BERT :** Architecture Transformer bidirectionnelle introduite par Google en 2018. Référence principale de notre étude.
* **RoBERTa :** Version optimisée de BERT (entraînement plus long, masquage dynamique, plus grand volume de données). Étudié pour mesurer l’impact du pré-entraînement sur la robustesse.
* **DeBERTa :** Architecture reposant sur le mécanisme de *Disentangled Attention* (découplage du contenu et de la position). Étudiée pour analyser l’impact des raffinements attentionnels modernes.

---

## Structure du dépôt
```text
TER-NLI/
│
├── notebooks/
│   ├── BERT_SNLI_Training_Optuna.ipynb
│   ├── RoBERTa_SNLI_Training_Optuna.ipynb
│   ├── DeBERTa_SNLI_Training_Optuna.ipynb
│   ├── BERT_MNLI_Training.ipynb
│   ├── BERT_ANLI_Training.ipynb
│   ├── Evaluation_Cross_Benchmark.ipynb
│   ├── Interpretability_BERT_SNLI.ipynb
│   └── BERT_SNLI_Weight_Decay_Sensitivity.ipynb
│
├── models/
├── figures/
├── results/
├── requirements.txt
└── README.md
```

## Protocole expérimental

### 1. Prétraitement
Les paires prémisse–hypothèse sont tokenisées, tronquées et converties en tenseurs compatibles avec les bibliothèques HuggingFace `transformers` et `datasets`.

### 2. Optimisation des hyperparamètres
Les notebooks `BERT_SNLI` et `RoBERTa_SNLI` utilisent une optimisation bayésienne via **Optuna** (*Tree-structured Parzen Estimator*). Les hyperparamètres étudiés sont : `learning rate`, `weight decay`, `batch size` et `epochs`. Les recherches sont réalisées sur un sous-ensemble représentatif afin de limiter le coût calculatoire.

### 3. Fine-tuning & Évaluation Cross-Benchmark
Les modèles finetunés (sur SNLI, MNLI ou ANLI) sont ensuite croisés et évalués sur l'ensemble des quatre benchmarks afin d'isoler finement les capacités de généralisation et la sensibilité aux biais.

### Métriques utilisées
* Accuracy, Precision, Recall, F1-score / F1-macro.
* Matrices de confusion.

---

## Interprétabilité
Le projet inclut également une analyse du processus de décision des modèles via deux approches :
* **Attention Maps :** Visualisation des poids d’attention des différentes couches des Transformers.
* **Integrated Gradients (via Captum) :** Méthode d’attribution post-hoc permettant d’identifier l’importance de chaque token dans la prédiction finale et de détecter les biais lexicaux.

---

## Environnement technique & Installation

### Bibliothèques principales
* Python | PyTorch
* HuggingFace (Transformers & Datasets)
* Scikit-learn | Optuna
* Matplotlib | Seaborn | Captum


### Installation
```bash
pip install -r requirements.txt
```
---

## Auteurs
* **Aïcha Ben Salah**
* **Mazin Mohammed Ahmed**

**Encadrement :** Docteur David Rousseau  
*Université d’Angers — Master Data Science — Année Universitaire 2025/2026*