# 🎙️ Chatterbox TTS FR

Interface Gradio locale pour piloter le modèle **Chatterbox TTS** avec le fine-tune français `Thomcles/Chatterbox-TTS-French`.

## Présentation
Chatterbox TTS FR est une interface de production audio locale pensée pour la
voix off : génération rapide, pré-écoute immédiate, presets réutilisables et
gestion fine des pauses. La génération est isolée dans un process séparé pour
permettre un arrêt immédiat (STOP) sans corrompre les sorties. Un traitement
audio optionnel (fade + zero-cross + détection de silences) réduit les pops
aux coupes.

Pensée pour les créatifs audiovisuels :
- sélection d’une **référence voix**
- saisie de **texte multi-ligne**
- **ajustement optionnel** à une durée cible
- sliders simples pour les paramètres expressifs
- bouton **STOP** pour annuler une génération en cours (aucun fichier conservé)
- réglages de **coupes audio** (fade, zero-cross, détection de silences)
- export **WAV horodaté** + pré-écoute dans l’UI

---

## Quickstart (60 secondes)

```bash
cd /Users/bricesodini/01_ai-stack/Chatterbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

➡️ Ouvrez ensuite http://127.0.0.1:7860

Au premier lancement, les poids Hugging Face sont téléchargés et mis en cache (internet requis une seule fois).

---

## 1. Prérequis

- macOS (Apple Silicon recommandé, backend MPS pris en charge)
- Python 3.11
- Accès à internet uniquement lors du premier lancement (téléchargement des poids Hugging Face)

### Dépendances Python

- chatterbox-tts
- torch (build compatible MPS recommandé)
- gradio
- librosa
- soundfile
- huggingface_hub
- safetensors
- numpy
- pytest (tests)

### Installation type

```bash
cd /Users/bricesodini/01_ai-stack/Chatterbox
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

💡 Si MPS n’est pas détecté : vérifiez votre installation PyTorch (certaines roues n’activent pas MPS selon la version/macOS). Consultez la doc officielle PyTorch pour macOS/Apple Silicon et installez une version compatible (`torch`, `torchvision`, `torchaudio` si besoin).

---

## 2. Structure projet

```
Chatterbox/
├── app.py            # UI Gradio (entrée principale)
├── refs.py           # gestion des fichiers Ref_audio/
├── text_tools.py     # outils texte + estimation/ajustement durée
├── tts_engine.py     # wrapper Chatterbox + fine-tune Thomcles
├── output_paths.py   # nommage + gestion preview/user
├── state_manager.py  # persistence state + presets
├── Ref_audio/        # références vocales (source unique de vérité)
├── output/           # WAV générés + preview Gradio
├── .state/state.json # état auto (dernier out dir, sliders…)
├── presets/          # presets créatifs (JSON)
└── tests/            # tests pytest
```

- `Ref_audio/` contient vos références (`.wav`, `.mp3`, `.m4a`, `.aiff`, `.flac`).
- `output/` contient aussi la copie “preview” servie à Gradio (Gradio-safe).
- `.state/` et `presets/` sont portables (commitables si besoin).

### Variables d’environnement (facultatif)

- `CHATTERBOX_REF_DIR` : changer le dossier de références.
- `CHATTERBOX_OUT_DIR` : changer le dossier de sortie par défaut.
- `GRADIO_SERVER_PORT` : changer le port Gradio (par défaut 7860).

---

## 3. Lancer l’application

```bash
cd /Users/bricesodini/01_ai-stack/Chatterbox
source .venv/bin/activate
python app.py
```

Gradio démarre sur http://127.0.0.1:7860. Tout tourne localement (pas de cloud requis).

---

## 4. Parcours utilisateur

### 4.1 Références vocales
- La liste déroulante affiche le contenu de `Ref_audio/`.
- Bouton **Refresh** : re-scan du dossier.
- Upload : copie dans `Ref_audio/` avec suffixes anti-collision (`_01_YYYYMMDD_HHMMSS`).
- Extensions autorisées : `.wav`, `.mp3`, `.m4a`, `.aiff`, `.flac` (les autres sont refusées proprement avec log explicite).

### 4.2 Zone texte & durée cible
- Champ multiligne (pas de SSML requis).
- Optionnel : renseignez une durée cible (secondes) puis cliquez sur **Ajuster le texte**.
- La suggestion apparaît en lecture seule ; **Utiliser la suggestion** remplace votre texte.
- Avertissement si l’algorithme a dû couper/allonger de façon importante (±20 %).
- Textbox « Texte interprété » : affiche le script réellement envoyé au TTS (anciens tokens ignorés).
- **Pré-chunking** : découpage déterministe AVANT la synthèse (utile > 40s).
- Slider **Mots minimum par chunk** : évite les fragments trop courts (ex: ligne d’un mot).
- Slider **Max mots sans terminator** : seuil de fallback si aucune fin de phrase n’est détectée.
- Slider **Durée max/chunk (s)** : garde-fou anti-dérive (10s recommandé).
- Bouton **Appliquer** : calcule le pré-chunking et affiche le résultat.
- Si un réglage change après **Appliquer** : état **non appliqué**. Au clic **Générer**, auto-apply avec log `auto_apply_before_generate`.
- Normalisation légère avant synthèse (ex: correction “II” → “Il”, espaces/retours ligne homogénéisés).
- Accordion « Aperçu des chunks » : liste numérotée (words, est_seconds, cut_reason, warnings).
- Le toggle **Logs détaillés** contrôle aussi la verbosité du terminal (tqdm + logs internes).
- Preset prêt à l’emploi : **stable-long-form** (pauses plus courtes + min mots + max durée).
 - Le preset **stable-long-form** est recommandé pour les textes longs ou littéraires afin d’éviter toute dérive vocale.
 - Preset **default** : profil grand public (robuste).

#### Pauses automatiques (ponctuation)
- Virgule `,` : pause réglable (par défaut 250ms).
- Point / fin de phrase `. ! ? …` : pause réglable (par défaut 400ms).
- Point-virgule `;`, deux-points `:`, tiret `—`/`-` : pauses réglables.
- Retour ligne `\n` : pause réglable (par défaut 300ms).
- La ponctuation est conservée dans le texte envoyé au modèle ; les silences sont ajoutés au montage audio.
- Les anciens tokens `{pause:ms}`, `{breath}`, `{beat}` sont ignorés si présents.

### Pré-chunking — ordre de priorité (résumé)
Le découpage suit une logique déterministe :

1. Retour ligne `\n` (prioritaire) si `min_words_per_chunk` atteint.
2. Fin de phrase `. ! ?` si le chunk dépasse `max_est_seconds`.
3. Fallback si phrase trop longue sans terminator : `:` puis `;` puis tiret `—`/`-` puis `,`.
4. Hard split par budget mots si aucun candidat.

Note : la ponctuation sert au rythme et n’est jamais un déclencheur de chunking hors fallback/limites.

### Anti-charabia (recommandé)
Si vous entendez des mots incompréhensibles en long-form :
- utilisez le preset **stable-long-form**
- activez **Stabiliser ponctuation**
- réduisez la **Température** (ex: 0.3–0.4) et augmentez légèrement le **CFG** (ex: 0.7–0.9)
- évitez les références vocales trop traitées (denoise agressif, compression) ou avec de longs silences “zéro”

Option future souhaitable :
- Bouton “Optimiser pour stabilité TTS” → applique le preset **stable-long-form**.

### 4.3 Paramètres créatifs
- Exagération (0–1.5) : expressivité globale.
- CFG : stabilité / tenue de la voix.
- Température : stabilité vs variation.
- Repetition Penalty : limite les répétitions.

Valeurs par défaut : 0.5 · 0.6 · 0.5 · 1.35 conformément au cahier des charges.

### 4.4 Traitement audio (coupes)
- Fade (ms) : fondu appliqué aux coupes.
- Zero-cross radius (ms) : fenêtre autour des coupes pour trouver le passage à zéro.
- Silence threshold : amplitude max pour considérer un silence.
- Silence min (ms) : durée min d’un silence pour appliquer le fade.

### 4.5 Sortie
- Champ « Dossier de sortie » (par défaut `output/`) personnalisable.
- Bouton **Choisir…** : ouvre le sélecteur natif macOS (Finder) et remplit automatiquement le champ avec le dossier choisi.
- Champ « Nom de fichier (optionnel) » : impose un nom (nettoyé), sinon fallback slug texte/ref.
- Toggle « Ajouter timestamp » (ON par défaut) : appose `_YYYY-MM-DD_HH-MM-SS`; si OFF et collision → suffixes `_01`, `_02`, etc.
- Prévisualisation audio : Gradio joue toujours la version `./output/...` (safe) puis l’outil copie le même fichier dans le dossier utilisateur choisi.
- **Générer** : 1 texte → 1 WAV.
- **STOP** : annule immédiatement la génération en cours, supprime le fichier temporaire.
- Nom final robuste, aucune écrasement silencieux.

### 4.6 Logs
Chaque action ajoute une ligne horodatée : import réussi/refusé, estimation + ajustement durée, lancement TTS, chemin de sortie, erreurs éventuelles.

### 4.7 Presets & état
- L’état courant (dernier dossier, sliders, nom de fichier, toggle timestamp…) est sauvegardé dans `./.state/state.json` à chaque génération ou changement d’output. Au redémarrage, l’UI se pré-remplit automatiquement.
- Section **Presets** :
  - Dropdown des presets présents dans `./presets/*.json`
  - Boutons `Charger`, `Sauver`, `Supprimer`
  - Les presets incluent : ref sélectionnée, dossier, nom de fichier, toggle timestamp, sliders.
  - Format JSON portable → partage facile.

---

## 5. Notes techniques

- `TTSEngine` charge `ResembleAI/chatterbox`, puis remplace uniquement `t3` par `t3_cfg.safetensors` depuis `Thomcles/Chatterbox-TTS-French`.
- Exécution forcée sur `mps` si disponible, sinon CPU.
- Les WAV sont écrits au sample rate natif du modèle.
- Les références absentes lèvent une erreur claire côté UI sans crasher l’app.
- La gestion du rythme est basée sur la ponctuation (virgules/points/retours ligne/tirets/points-virgules/deux-points). Les anciens tokens `{pause:ms}`, `{breath}`, `{beat}` sont ignorés.
- Prévisualisation Gradio : le fichier est toujours généré dans `./output/...`, puis copié dans le dossier utilisateur (aucun conflit avec les restrictions Gradio).
- Pré-chunking : découpage déterministe avant synthèse, avec fallback ponctuation puis hard split si nécessaire.

### Cache Hugging Face
- Les poids sont mis en cache automatiquement par `huggingface_hub` (par défaut sous `~/.cache/huggingface/`).
- Après le premier téléchargement, l’outil fonctionne hors-ligne tant que ce cache est présent.

---

## 6. Dépannage

- Blocage au premier lancement : connexion internet requise (download Hugging Face).
- MPS non détecté : l’app bascule sur CPU (plus lent).
- Import refusé : extension non listée (`.wav`, `.mp3`, `.m4a`, `.aiff`, `.flac`).
- Durée loin de la cible : l’ajustement est volontairement conservateur ; ajustez le texte manuellement si l’écart > ±20 %.
- Répertoires personnalisés : exportez `CHATTERBOX_REF_DIR` / `CHATTERBOX_OUT_DIR` avant `python app.py`.
- Gradio refuse un wav externe ? Le fichier preview reste toujours dans `./output/` : vérifier que ce dossier est accessible / non supprimé.

---

## 7. Tests

- Tests unitaires (ponctuation, naming, refs) via `pytest` :

```bash
source .venv/bin/activate
pytest -q
```

- Validation rapide : `python -m py_compile app.py text_tools.py tts_engine.py refs.py`

---

## 8. Logs terminal (tqdm / internes)

- Par défaut, le terminal est “clean” (pas de barres tqdm ni logs internes).
- Activer **Logs détaillés** pour réactiver le verbose terminal.

---

## 9. Prochaines itérations possibles

- LLM local pour reformulations plus intelligentes à durée cible.
- Bouton « Ouvrir dossier de sortie » depuis l’UI.
- Mesure automatique de la durée générée pour closing loop.
- Presets partagés en satu.

---

## 10. FAQ

**Q : J’obtiens des warnings Transformers (cache/attention). Dois-je m’inquiéter ?**  
R : Non, ils sont courants avec Chatterbox et n’impactent pas la génération. Vous pouvez réduire leur verbosité via `os.environ["TRANSFORMERS_VERBOSITY"] = "error"` si besoin.
