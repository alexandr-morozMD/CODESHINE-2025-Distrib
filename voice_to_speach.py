import whisper

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment

from pydub import AudioSegment

import re
from collections import Counter
from typing import Iterable, List, Tuple

# def record_voice(filename="audio.wav", duration=10, samplerate=16000):
#     print(f"🎤 Запись {duration} секунд... Говорите!")
#     audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
#     sd.wait()  # ждём завершения записи
#     wav.write(filename, samplerate, (audio * 32767).astype(np.int16))  # конвертация в int16
#     print(f"✅ Запись сохранена в {filename}")


def transcribe_voice(filename="audio.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename, language="en")
    print("Recognized text:")
    print(result["text"])


def convert_mp3_to_wav(mp3_filename, wav_filename="audio.wav"):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")
    print(f"Conversion of {mp3_filename} to {wav_filename} completed.")

# def split_text_by_context(text):
#     max_characters = 200
#     splitter = CharacterTextSplitter(chunk_size=max_characters, chunk_overlap=0)
#     chunks = splitter.split_text(text)
#     return chunks

def transcribe_audio(filebytes):
    with open("temp_audio.wav", "wb") as f:
        f.write(filebytes)
        f.close()
    model = whisper.load_model("base")
    result = model.transcribe("temp_audio.wav", language="en")
    # print("📝 Распознанный текст:")
    return result["text"]

MAX_PARAGRAPH_CHARS = 900
MIN_PARAGRAPH_CHARS = 200

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "we",
    "i",
    "you",
    "they",
    "this",
    "those",
    "these",
    "their",
    "our",
    "or",
    "but",
    "so",
    "if",
    "not",
    "can",
    "could",
    "should",
    "would",
    "about",
    "into",
    "also",
    "there",
}


def transcription_keys_model(text):
    paragraphs = _segment_transcript(text)
    paragraph_summaries = [
        (paragraph, _extract_key_points(paragraph)) for paragraph in paragraphs
    ]
    tasks = _extract_tasks(text)
    return _format_analysis(paragraph_summaries, tasks)
    

def process_meeting_audio(file):
    print(file)
    # convert_mp3_to_wav("audio.mp3", "audio.wav")
    transcription = transcribe_audio(file.getvalue())
    # print(transcription)
    analyse_results = transcription_keys_model(transcription)
    return analyse_results


def text_modify(text):
    parts = text.split("Tasks:", 1)
    if len(parts) == 2:
        before_tasks = parts[0].strip()
        after_tasks = parts[1].strip()
        return before_tasks, after_tasks
    else:
        return text.strip(), ""

def paragraph_modify(text):
    regex = re.compile(r"Paragraph [0-9]:")
    modified_text = regex.sub(lambda match: f"#### {match.group()}\n\n", text)
    # print(regex.search(starting_text).group())
    return modified_text

def key_points(text: str) -> list[str]:
    # Разбиваем по блокам задач

    regex = re.compile(r"Key Points:")
    modified_text = regex.sub(lambda match: f"\n{match.group()}", text)
    # print(regex.search(starting_text).group())
    return modified_text

    
# def task_to_list(text: str) -> list[str]:
#     text = text.replace("\n", "\n\n")
#     pattern = re.compile(r"Task \d+:")
#     # return re.findall(pattern, text)
#     result = pattern.split(text)
#     print(f"{result=}")
#     return result
#     # return text.split("Task")
    
def task_to_list(text: str) -> list[str]:
    # Разбиваем по блокам задач
    tasks = re.split(r"- Task \d+:", text)
    results = []

    for t in tasks:
        t = t.strip()
        if not t:
            continue

        assigned_match = re.search(r"Assigned to:\s*(.*)", t)
        task_match = re.search(r"Task:\s*(.*)", t)
        details_match = re.search(r"Details:\s*(.*)", t, re.DOTALL)

        assigned = assigned_match.group(1).strip() if assigned_match else "Unassigned"
        task = task_match.group(1).strip() if task_match else "No task description"
        details = details_match.group(1).strip() if details_match else "No details"

        results.append(
            f"**Assigned to:** {assigned}\n\n"
            f"**Task:** {task}\n\n"
            f"**Details:** {details}"
        )

    return results


def _segment_transcript(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    # First try to split by blank lines to preserve natural paragraphs
    candidates = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not candidates:
        candidates = [cleaned]

    paragraphs: List[str] = []
    for block in candidates:
        if len(block) <= MAX_PARAGRAPH_CHARS:
            paragraphs.append(block)
            continue

        start = 0
        while start < len(block):
            end = min(start + MAX_PARAGRAPH_CHARS, len(block))
            split_at = _find_split_point(block, start, end)
            paragraphs.append(block[start:split_at].strip())
            start = split_at

    merged: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) < MIN_PARAGRAPH_CHARS and buffer:
            buffer = f"{buffer} {paragraph}".strip()
            merged.append(buffer)
            buffer = ""
        elif len(paragraph) < MIN_PARAGRAPH_CHARS:
            buffer = paragraph
        else:
            if buffer:
                merged.append(f"{buffer} {paragraph}".strip())
                buffer = ""
            else:
                merged.append(paragraph)
    if buffer:
        merged.append(buffer)

    return merged or [cleaned]


def _find_split_point(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return len(text)

    window = text[start:end]
    for delimiter in [". ", "! ", "? ", "\n"]:
        relative = window.rfind(delimiter)
        if relative != -1:
            return start + relative + len(delimiter)
    return end


def _extract_key_points(paragraph: str) -> List[str]:
    sentences = _split_into_sentences(paragraph)
    if not sentences:
        return []

    if len(sentences) <= 3:
        return [sentence.strip().rstrip(".,") for sentence in sentences]

    frequencies = _collect_word_frequencies(paragraph)
    scored: List[Tuple[str, float]] = []

    for sentence in sentences:
        words = _normalize_words(sentence)
        if not words:
            continue
        score = sum(frequencies.get(word, 0) for word in words) / len(words)
        scored.append((sentence, score))

    if not scored:
        return [sentence.strip().rstrip(".,") for sentence in sentences[:3]]

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = [sentence.strip().rstrip(".,") for sentence, _ in scored[:3]]
    return selected


def _collect_word_frequencies(text: str) -> Counter:
    words = _normalize_words(text)
    return Counter(words)


def _normalize_words(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in re.split(r"[^A-Za-zА-Яа-яЁё0-9']+", text.lower()):
        candidate = raw.strip("' ")
        if candidate and candidate not in STOPWORDS:
            tokens.append(candidate)
    return tokens


def _extract_tasks(text: str) -> List[Tuple[str, str, str]]:
    indicators = [
        "need to",
        "needs to",
        "has to",
        "should",
        "must",
        "will",
        "action item",
        "follow up",
        "todo",
        "task",
    ]

    sentences = _split_into_sentences(text)
    tasks: List[Tuple[str, str, str]] = []

    for sentence in sentences:
        lowered = sentence.lower()
        if not any(indicator in lowered for indicator in indicators):
            continue

        assigned = _detect_assignee(sentence)
        task_desc = _detect_task_description(sentence)
        details = _detect_details(sentence)
        tasks.append((assigned, task_desc, details))

    return tasks


def _detect_assignee(sentence: str) -> str:
    patterns = [
        re.compile(r"\b([A-ZА-ЯЁ][\w'-]*(?:\s+[A-ZА-ЯЁ][\w'-]*)*)\s+(?:needs to|need to|has to|should|must|will|shall|is going to)", re.IGNORECASE),
        re.compile(r"^(Team|Everyone|All|We)\b", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(sentence)
        if match:
            candidate = match.group(1)
            if candidate.lower() in {"we", "all"}:
                return "Team"
            return candidate.title()
    return "Unassigned"


def _detect_task_description(sentence: str) -> str:
    patterns = [
        re.compile(r"(?:needs to|need to|has to|should|must|will|shall|is going to)\s+(.*)", re.IGNORECASE),
        re.compile(r"(?:Let'?s|Let us)\s+(.*)", re.IGNORECASE),
        re.compile(r"action item[:\-]\s*(.*)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip().rstrip(". ")
    return sentence.strip().rstrip(". ")


def _detect_details(sentence: str) -> str:
    deadline_pattern = re.compile(r"\b(by|before|on|at|due|next)\s+([^,.;]+)", re.IGNORECASE)
    matches = deadline_pattern.findall(sentence)
    if matches:
        details = [" ".join(match).strip() for match in matches]
        return "; ".join(details)
    return ""


def _format_analysis(
    paragraph_summaries: Iterable[Tuple[str, List[str]]],
    tasks: List[Tuple[str, str, str]],
) -> str:
    lines: List[str] = []
    for index, (paragraph, key_points) in enumerate(paragraph_summaries, start=1):
        if not paragraph.strip():
            continue
        lines.append(f"Paragraph {index}:")
        lines.append(paragraph.strip())
        lines.append("Key Points:")
        if key_points:
            for point in key_points:
                lines.append(f"- {point}")
        else:
            lines.append("- No key points detected")
        lines.append("")

    lines.append("Tasks:")
    if tasks:
        for idx, (assigned, task, details) in enumerate(tasks, start=1):
            lines.append(f"- Task {idx}:")
            lines.append(f"  Assigned to: {assigned or 'Unassigned'}")
            lines.append(f"  Task: {task or 'No task description'}")
            lines.append(f"  Details: {details or 'No additional details'}")
    else:
        lines.append("- Task 1:")
        lines.append("  Assigned to: Unassigned")
        lines.append("  Task: Action items were not detected automatically")
        lines.append("  Details: Review the transcript manually")

    return "\n".join(lines).strip()


def _split_into_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text.strip())
    cleaned = []
    for sentence in sentences:
        candidate = sentence.strip()
        if candidate:
            cleaned.append(candidate)
    return cleaned
