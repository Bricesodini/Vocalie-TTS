import type { UIState } from "@/lib/types";

export const POLL_INTERVAL_MS = 700;

export const LANGUAGE_OPTIONS = [
  { value: "fr-FR", label: "Francais (fr-FR)" },
  { value: "en-US", label: "English (en-US)" },
  { value: "es-ES", label: "Espanol (es-ES)" },
  { value: "de-DE", label: "Deutsch (de-DE)" },
  { value: "it-IT", label: "Italiano (it-IT)" },
  { value: "pt-PT", label: "Portugues (pt-PT)" },
  { value: "nl-NL", label: "Nederlands (nl-NL)" },
  { value: "ja-JP", label: "Japanese (ja-JP)" },
  { value: "ko-KR", label: "Korean (ko-KR)" },
  { value: "zh-CN", label: "Chinese (zh-CN)" },
  { value: "zh-TW", label: "Chinese (zh-TW)" },
  { value: "ru-RU", label: "Russian (ru-RU)" },
];

export const VOICE_DESIGN_PRESETS = [
  {
    id: "fr_news_m",
    label: "Journal TV FR (masculin)",
    instruct:
      "Voix masculine adulte, timbre clair, pitch moyen-bas, debit soutenu, volume fort, accent francais neutre. Ton autoritaire, confiant et informatif.",
  },
  {
    id: "fr_story_f",
    label: "Narration douce FR (feminin)",
    instruct:
      "Voix feminine adulte, timbre doux, pitch moyen, debit moyen, volume modere, accent francais neutre. Ton chaleureux et rassurant.",
  },
  {
    id: "fr_angry_m",
    label: "Colere FR (masculin)",
    instruct:
      "Voix masculine adulte, timbre rauque, pitch moyen-bas, debit rapide, volume fort. Emotion colerique, ton tranchant et percutant.",
  },
  {
    id: "fr_young_f",
    label: "Jeune adulte FR (feminin)",
    instruct:
      "Voix feminine jeune adulte, pitch moyen-haut, debit rapide, volume normal, accent francais neutre. Ton enjoue et expressif.",
  },
  {
    id: "fr_senior_m",
    label: "Senior FR (masculin)",
    instruct:
      "Voix masculine senior, pitch bas, debit lent, volume modere, accent francais neutre. Ton grave et pose.",
  },
];

export const VOICE_DESIGN_OPTIONS = {
  gender: [
    { value: "none", label: "Neutre" },
    { value: "masculine", label: "Masculin" },
    { value: "feminine", label: "Feminin" },
  ],
  age: [
    { value: "none", label: "Neutre" },
    { value: "teen", label: "Ado" },
    { value: "young_adult", label: "Jeune adulte" },
    { value: "adult", label: "Adulte" },
    { value: "senior", label: "Senior" },
  ],
  pitch: [
    { value: "none", label: "Neutre" },
    { value: "low", label: "Bas" },
    { value: "mid", label: "Moyen" },
    { value: "high", label: "Haut" },
  ],
  speed: [
    { value: "none", label: "Neutre" },
    { value: "slow", label: "Lent" },
    { value: "medium", label: "Normal" },
    { value: "fast", label: "Rapide" },
  ],
  volume: [
    { value: "none", label: "Neutre" },
    { value: "soft", label: "Faible" },
    { value: "normal", label: "Normal" },
    { value: "loud", label: "Fort" },
  ],
  accent: [
    { value: "none", label: "Neutre" },
    { value: "fr_neutral", label: "Francais neutre" },
    { value: "fr_paris", label: "Francais parisien" },
    { value: "fr_quebec", label: "Francais quebecois" },
    { value: "fr_belgium", label: "Francais belge" },
    { value: "fr_swiss", label: "Francais suisse" },
  ],
  emotion: [
    { value: "none", label: "Neutre" },
    { value: "happy", label: "Joyeux" },
    { value: "sad", label: "Triste" },
    { value: "angry", label: "Colere" },
    { value: "excited", label: "Excite" },
    { value: "calm", label: "Calme" },
  ],
  texture: [
    { value: "none", label: "Neutre" },
    { value: "clear", label: "Claire" },
    { value: "warm", label: "Chaleureuse" },
    { value: "raspy", label: "Rauque" },
    { value: "nasal", label: "Nasale" },
  ],
  style: [
    { value: "none", label: "Neutre" },
    { value: "conversational", label: "Conversationnel" },
    { value: "narrative", label: "Narratif" },
    { value: "authoritative", label: "Autoritaire" },
    { value: "dramatic", label: "Dramatique" },
  ],
};

export const EMPTY_STATE: UIState = {
  preparation: {
    text_raw: "",
    text_adjusted: "",
    text_interpreted: "",
    glossary_enabled: false,
    glossary_profile: null,
    glossary_options: {},
  },
  direction: {
    snapshot_text: "",
    chunk_markers: [],
    chunk_ranges: [],
    chunks_preview: [],
  },
  engine: {
    engine_id: "",
    voice_id: null,
    language: "fr-FR",
    params: {},
    chunk_gap_ms: 0,
  },
  post: {
    edit_enabled: false,
    trim_enabled: false,
    normalize_enabled: false,
    target_dbfs: -1,
  },
};

