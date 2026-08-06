export type SourceLanguage = "auto" | "zh-CN" | "en" | "ja" | "ko" | "es" | "fr" | "de";
export type TargetLanguage = Exclude<SourceLanguage, "auto">;
export type SenseVoiceLanguage = "auto" | "zh" | "yue" | "en" | "ja" | "ko";

export const senseVoiceLanguages: ReadonlyArray<{ value: SenseVoiceLanguage; label: string }> = [
  { value: "auto", label: "自动检测" },
  { value: "zh", label: "普通话" },
  { value: "yue", label: "粤语" },
  { value: "en", label: "英语" },
  { value: "ja", label: "日语" },
  { value: "ko", label: "韩语" },
];

export const sourceLanguages: ReadonlyArray<{ value: SourceLanguage; label: string }> = [
  { value: "auto", label: "自动检测" },
  { value: "zh-CN", label: "中文（简体）" },
  { value: "en", label: "英语" },
  { value: "ja", label: "日语" },
  { value: "ko", label: "韩语" },
  { value: "es", label: "西班牙语" },
  { value: "fr", label: "法语" },
  { value: "de", label: "德语" },
];

export const targetLanguages = sourceLanguages.filter(
  (language): language is { value: TargetLanguage; label: string } => language.value !== "auto",
);

export function getLanguageLabel(value: SourceLanguage | TargetLanguage) {
  return sourceLanguages.find((language) => language.value === value)?.label || value;
}

export function getTranslationPreview(language: TargetLanguage) {
  const previews: Record<TargetLanguage, string> = {
    "zh-CN": "翻译会显示在这里",
    en: "Translation appears here",
    ja: "翻訳はここに表示されます",
    ko: "번역이 여기에 표시됩니다",
    es: "La traducción aparece aquí",
    fr: "La traduction s’affiche ici",
    de: "Die Übersetzung erscheint hier",
  };
  return previews[language];
}
