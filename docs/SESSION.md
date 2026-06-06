# Session 2026-06-06 — Unification zone texte

## Objectif
Résoudre le bug "You need to add some text for me to talk" + simplifier l'UI.

## Fait
- ✅ Diagnostic : le message vient de `chatterbox/tts.py:28 punc_norm()` (code dur),
  déclenché par `len(text) == 0`. Mémoire hippo "hallucination 3-mots" invalidée.
- ✅ Pass 1 UX Engine émis : `docs/runs/2026-06-06_text-zone-unification/pass-1-output.md`
- ✅ Refactor frontend :
  - Suppression tabs (Texte/Ajusté/Interprété)
  - Suppression bouton "Préparer"
  - Suppression bouton "Aperçu chunks"
  - Une seule zone texte éditable (snapshot_text = source de vérité)
  - Aperçu normalisé (lexique) intégré à la section Texte
  - Section "Aperçu des chunks" auto
  - Boutons séparateur agissent sur la zone unique au curseur
  - Fonctions mortes supprimées (handlePrepare, handleSnapshot, handlePreview, etc.)
- ✅ Backend : `_pad_short_text()` branché dans `run_tts_pipeline`
- ✅ TypeScript compile, build OK
- ✅ Commit + push

## À tester (côté user)
1. Taper "test de voix pour voir si ça fonctionne" → Générer
   - Attendu : audio du texte (PAS "you need to add some text")
2. Taper un texte très court (1-2 mots) → vérifier le padding (audio plus long)
3. Cliquer "Insérer séparateur" au milieu du texte → vérifier aperçu chunks
4. Activer le glossaire → vérifier que l'aperçu normalisé apparaît

## Décisions
- **Zone unique** : `snapshot_text` est la source de vérité (ce qui est envoyé au backend)
- **`text_raw` synchronisé** : pour ne pas casser la route backend
- **Padding 6 mots** : hard-coded dans `tts_pipeline.py` (constante `MIN_WORDS_FOR_SYNTHESIS`)

## Risques restants
- Faible : pas de trim audio compensatoire après padding (audio un peu plus long que nécessaire)
- Faible : `text_interpreted` toujours dans le state mais plus affiché (route `/v1/prep/interpret` est no-op)

## Prochaine étape
Test utilisateur. Si OK, on peut éventuellement :
- Brancher `_trim_audio_to_expected_duration()` pour le trim compensatoire
- Supprimer l'état `text_interpreted` et la route `/v1/prep/interpret` (no-op)
