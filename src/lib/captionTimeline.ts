import type { CaptionResult } from "../hooks/useSpeechRecognition";

export const MIN_CAPTION_FONT_SIZE = 12;

export type TimedCaptionSegment = {
  text: string;
  isFinal: boolean;
  receivedAt: number;
  speaker?: number;
};

export type CaptionLineRun = {
  text: string;
  speaker?: number;
};

type IndexedCaptionRun = CaptionLineRun & {
  start: number;
  end: number;
};

const SENTENCE_END = /[\u3002\uff01\uff1f!?\uff1b;]$/;
const LEADING_PUNCTUATION = /^[\u3002\uff01\uff1f!?\uff1b;\uff0c,\u3001\uff1a:]/;
const ASCII_WORD_END = /[A-Za-z0-9]$/;
const ASCII_WORD_START = /^[A-Za-z0-9]/;

export function applyCaptionResult(
  segments: TimedCaptionSegment[],
  result: CaptionResult,
  speaker?: number,
): TimedCaptionSegment[] {
  const next: TimedCaptionSegment = {
    text: result.text.trim(),
    isFinal: result.isFinal,
    receivedAt: result.receivedAt,
    speaker,
  };

  if (!next.text) return segments;

  const last = segments[segments.length - 1];
  if (last && !last.isFinal) {
    return [...segments.slice(0, -1), next];
  }

  return [...segments, next];
}

function separatorBetween(previous: string, next: string) {
  if (!previous || !next || SENTENCE_END.test(previous) || LEADING_PUNCTUATION.test(next)) {
    return "";
  }
  return ASCII_WORD_END.test(previous) && ASCII_WORD_START.test(next) ? " " : "";
}

function characterWeight(character: string) {
  if (/\s/.test(character)) return 0.35;
  if (/^[\u0000-\u00ff]$/.test(character)) return 0.56;
  return 1;
}

export function captionTextWeight(text: string) {
  return Array.from(text).reduce((total, character) => total + characterWeight(character), 0);
}

export function composeCaptionText(segments: TimedCaptionSegment[]) {
  let text = "";
  const segmentBoundaries: number[] = [];
  const runs: IndexedCaptionRun[] = [];

  for (const segment of segments) {
    const normalized = segment.text.replace(/\s+/g, " ").trim();
    if (!normalized) continue;

    const start = text.length;
    text += separatorBetween(text, normalized);
    if (text) segmentBoundaries.push(text.length);
    text += normalized;
    runs.push({
      text: text.slice(start),
      speaker: segment.speaker,
      start,
      end: text.length,
    });
  }

  return { text, segmentBoundaries, runs };
}

export function captionSegmentsToText(segments: TimedCaptionSegment[]) {
  return composeCaptionText(segments).text.trim();
}

function chooseLineBreak(text: string, segmentBoundaries: number[]) {
  const candidates = new Map<number, number>();
  const addCandidate = (index: number, priority: number) => {
    if (index <= 0 || index >= text.length) return;
    candidates.set(index, Math.min(priority, candidates.get(index) ?? Number.POSITIVE_INFINITY));
  };

  for (const boundary of segmentBoundaries) addCandidate(boundary, 1.2);

  Array.from(text).forEach((character, index) => {
    const position = index + 1;
    if (/[\u3002\uff01\uff1f!?\uff1b;]/.test(character)) addCandidate(position, 0);
    else if (/[\uff0c,\u3001\uff1a:]/.test(character)) addCandidate(position, 0.35);
    else if (/\s/.test(character)) addCandidate(position, 0.7);
    else addCandidate(position, 2.4);
  });

  const totalWeight = captionTextWeight(text);
  const targetWeight = totalWeight / 2;
  const minimumSideWeight = Math.min(4, totalWeight * 0.18);
  let bestIndex = Math.floor(text.length / 2);
  let bestScore = Number.POSITIVE_INFINITY;

  for (const [index, priority] of candidates) {
    const leftWeight = captionTextWeight(text.slice(0, index));
    const rightWeight = totalWeight - leftWeight;
    if (leftWeight < minimumSideWeight || rightWeight < minimumSideWeight) continue;

    const score = Math.abs(leftWeight - targetWeight) + priority * Math.max(1, totalWeight * 0.035);
    if (score < bestScore) {
      bestIndex = index;
      bestScore = score;
    }
  }

  return bestIndex;
}

function layoutCaptionLineRanges(
  segments: TimedCaptionSegment[],
  singleLineCapacity: number,
): Array<{ text: string; start: number; end: number }> {
  const composed = composeCaptionText(segments);
  if (!composed.text) return [];

  const maxVisibleWeight = Math.max(singleLineCapacity * 2, 1);
  const totalWeight = captionTextWeight(composed.text);
  let text = composed.text;
  let segmentBoundaries = composed.segmentBoundaries;
  let startOffset = 0;

  if (totalWeight > maxVisibleWeight) {
    let start = text.length - 1;
    let suffixWeight = characterWeight(text[start] ?? "");
    while (start > 0 && suffixWeight < maxVisibleWeight) {
      start -= 1;
      suffixWeight += characterWeight(text[start] ?? "");
    }

    const boundary = [
      ...segmentBoundaries,
      ...Array.from(text).map((character, index) => {
        const position = index + 1;
        return /[\u3002\uff01\uff1f!?\uff1b;]/.test(character) ? position : -1;
      }).filter((position) => position > 0),
    ]
      .filter((position) => position >= start && position < text.length)
      .sort((left, right) => Math.abs(left - start) - Math.abs(right - start))[0];

    if (boundary !== undefined) start = boundary;
    const sliced = text.slice(start);
    const trimmed = sliced.trim();
    startOffset = start + sliced.indexOf(trimmed);
    text = trimmed;
    segmentBoundaries = segmentBoundaries
      .filter((position) => position > startOffset)
      .map((position) => position - startOffset);
  }

  if (captionTextWeight(text) <= singleLineCapacity) {
    return [{ text, start: startOffset, end: startOffset + text.length }];
  }

  const breakAt = chooseLineBreak(text, segmentBoundaries);
  const rawFirst = text.slice(0, breakAt);
  const rawSecond = text.slice(breakAt);
  const first = rawFirst.trim();
  const second = rawSecond.trim();
  const firstStart = startOffset + rawFirst.indexOf(first);
  const secondStart = startOffset + breakAt + rawSecond.indexOf(second);
  return [
    first ? { text: first, start: firstStart, end: firstStart + first.length } : null,
    second ? { text: second, start: secondStart, end: secondStart + second.length } : null,
  ].filter((line): line is { text: string; start: number; end: number } => line !== null);
}

export function layoutCaptionLines(
  segments: TimedCaptionSegment[],
  singleLineCapacity: number,
): string[] {
  return layoutCaptionLineRanges(segments, singleLineCapacity).map((line) => line.text);
}

export function layoutCaptionLineRuns(
  segments: TimedCaptionSegment[],
  singleLineCapacity: number,
): CaptionLineRun[][] {
  const composed = composeCaptionText(segments);
  const lines = layoutCaptionLineRanges(segments, singleLineCapacity);
  return lines.map((line) => {
    const runs = composed.runs
      .filter((run) => run.end > line.start && run.start < line.end)
      .map((run) => {
        const start = Math.max(run.start, line.start);
        const end = Math.min(run.end, line.end);
        return {
          text: composed.text.slice(start, end),
          speaker: run.speaker,
        };
      })
      .filter((run) => run.text.length > 0);

    return runs.reduce<CaptionLineRun[]>((merged, run) => {
      const previous = merged[merged.length - 1];
      if (previous?.speaker === run.speaker) {
        previous.text += run.text;
      } else {
        merged.push(run);
      }
      return merged;
    }, []);
  });
}

export function fitCaptionFontSize(
  lines: string[],
  preferredFontSize: number,
  availableWidth: number,
) {
  const widestLine = Math.max(1, ...lines.map(captionTextWeight));
  const fittedSize = Math.floor((availableWidth * 0.96) / widestLine);
  return Math.max(MIN_CAPTION_FONT_SIZE, Math.min(preferredFontSize, fittedSize));
}
