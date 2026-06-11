# Mini GPT (caractère par caractère) — from scratch

Petit projet pédagogique : un Transformer **decoder-only** de style GPT, entraîné **caractère par caractère** sur un fichier texte, puis utilisé pour générer du texte en échantillonnant le prochain caractère pas à pas.

L’objectif est de comprendre et recoder les briques essentielles :
- embeddings de tokens + embeddings de position
- self-attention **causale** (pas d’accès au futur)
- connexions résiduelles + LayerNorm
- apprentissage en prédiction du token suivant (cross-entropy)
- génération par sampling

Ce projet n’a pas vocation à être un LLM de production. C’est volontairement minimal et fait pour apprendre.

---

## Contenu du dépôt

- `mini_gpt_tp.py`  
  Script unique qui :
  - charge `input.txt`
  - construit un vocabulaire de caractères
  - entraîne un mini GPT
  - génère un échantillon de texte

- `input.txt`  
  Ton corpus d’entraînement (texte UTF-8).

---

## Prérequis

- Python 3.10+
- PyTorch

Installation :

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install torch