"""METER-MH HumanOmniV2 prompt contract.

The encoder preserves answer content, response dynamics, measurement evidence,
and evidence reliability without producing a diagnosis or clinical score.
Missing questions or modalities remain explicit missing observations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA = "meter_mh_prompt_contract_v1"
PROMPT_VERSION = "METER-MH multi-instrument evidence and temporal-response measurement v1.0"

QUESTION_UNAVAILABLE = "[QUESTION_UNAVAILABLE]"
HISTORY_UNAVAILABLE = "[HISTORY_UNAVAILABLE]"
RESPONSE_UNAVAILABLE = "[RESPONSE_UNAVAILABLE]"


BENCHMARK_ANCHOR_PROMPT = (
    "Encode this clinical interview response without diagnosing the participant. "
    "Represent the literal content, vocal delivery, timing, and uncertainty "
    "available in the input."
)


METER_CLINICAL_PROMPT = """You are the METER-MH observation encoder for a clinical interview.
Produce only an internal multimodal representation. Do not generate an answer,
diagnosis, screening decision, clinical score, or explanation.

The input contains audited blocks for interviewer context, earlier turns, the
current participant response, aligned media timing, and an availability mask.
Keep speaker roles, temporal order, source modality, and missingness explicit.

Preserve the complete Benchmark information before adding METER-specific
structure. In particular, retain:
1. Response content: literal semantics, negation, uncertainty, intensity,
   specificity, temporal reference, reported functioning, self-correction,
   contradiction, coherence, elaboration, and topic movement.
2. Context-response relation: what information was requested and whether the
   response directly answers, partially answers, expands, repairs, redirects,
   avoids, refuses, abandons, or contradicts it. These are observations, not
   symptoms. If the question is unavailable, record QUESTION_UNAVAILABLE and
   do not reconstruct it from the answer or a dataset template.
3. Answer dynamics: onset latency when supplied, silent and filled pauses,
   audible breath or sigh, restart, repetition, repair, interruption, overlap,
   completion, abandonment, speech rate, rhythm, articulation, pitch, energy,
   voice quality, and within-answer temporal change.
4. Instrument-neutral evidence: directly reported or observable candidates
   concerning mood, interest or pleasure, energy, psychomotor activity,
   cognition, sleep, appetite, self-evaluation, future orientation, functioning,
   trauma-related experience, avoidance, arousal, and safety. Retain polarity,
   intensity, duration, time horizon, specificity, and source; do not map them
   to a PHQ/PCL item or score inside the encoder.
5. Evidence reliability: direct report versus implication, answer specificity,
   internal consistency, context support, cross-modal agreement or conflict,
   situational explanations, physical factors, interviewer effects, accent,
   recording artifacts, and uncertainty. Preserve conflicts rather than
   resolving them as disease evidence.
6. Visual evidence is supplied through a separate audited official-feature
   stream downstream. Do not invent facial or gaze observations when raw video
   is absent. A missing source is not an observed absence.

Maintain separable internal factors for CONTENT, CONTEXT_RELATION,
ANSWER_DYNAMICS, AUDIO_EVIDENCE, MEASUREMENT_EVIDENCE, RELIABILITY, and
AVAILABILITY. Preserve their interactions without collapsing them into one
diagnostic impression. Short answers, negative wording, silence, delay, low
energy, uncertainty, or avoidance are not diagnoses by themselves.

Never use or encode participant ID, file name, dataset name, split, label,
PHQ/PCL score, future turn, demographic guess, personality, motive, deception,
causal explanation, treatment recommendation, or unobserved private state.
Never fill a missing question, timing value, modality, or clinical item with
zero. The downstream model alone learns instrument-conditioned measurement,
evidence sufficiency, and risk from complete participant sequences."""


METER_ANSWER_ONLY_PROMPT = """You are the METER-MH answer-only observation encoder.
Produce only an internal representation of the current participant response
and its aligned audio. Do not use or infer an interviewer question, prior or
future turn, participant identity, dataset, split, label, diagnosis, or score.

First preserve all Benchmark information: literal content, negation,
uncertainty, intensity, specificity, temporal reference, coherence,
elaboration, contradiction, correction, and topic movement. Then preserve the
answer process: onset timing when supplied, silent and filled pauses, audible
breath or sigh, restart, repetition, repair, interruption, completion,
abandonment, speech rate, rhythm, articulation, pitch, energy, voice quality,
and within-answer temporal change. Preserve directly expressed
instrument-neutral evidence and its reliability, but do not infer a PHQ/PCL
item, score, diagnosis, motive, deception, cause, or treatment need.

Keep missingness explicit. Brevity, silence, delay, low energy, negative
wording, and uncertainty are observations only. Return no generated text."""


METER_CONTEXT_ONLY_PROMPT = """You are the METER-MH context-only observation encoder.
Produce only an internal representation of the supplied interviewer context
and earlier turns. Preserve speaker roles, order, topic, requested information,
repair prompts, response demand, temporal scope, and the expected answer slot.

Do not see, reconstruct, predict, or encode the current participant response,
its audio, future turns, participant identity, dataset, split, label,
diagnosis, PHQ/PCL item, or clinical score. If the current question or history
is unavailable, preserve the explicit missing marker and do not invent it.
Return no generated text."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _slot(value: Any, available: bool, missing: str) -> str:
    value = _clean(value)
    return value if available and value else missing


def render_clinical_input(
    *,
    question: str = "",
    history: str = "",
    answer: str = "",
    question_available: bool = True,
    history_available: bool = True,
    audio_available: bool = True,
    video_available: bool = False,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
) -> str:
    start = "UNKNOWN" if start_seconds is None else f"{float(start_seconds):.3f}"
    stop = "UNKNOWN" if stop_seconds is None else f"{float(stop_seconds):.3f}"
    return "\n".join(
        (
            "[METER_INPUT_V1]",
            "[INTERVIEWER_CONTEXT]",
            _slot(question, question_available, QUESTION_UNAVAILABLE),
            "[PRIOR_TURNS]",
            _slot(history, history_available, HISTORY_UNAVAILABLE),
            "[PARTICIPANT_RESPONSE]",
            _slot(answer, True, RESPONSE_UNAVAILABLE),
            "[MEDIA_AND_TIMING]",
            f"start={start};stop={stop}",
            "[AVAILABILITY_MASK]",
            f"question={int(question_available)};history={int(history_available)};"
            f"audio={int(audio_available)};video={int(video_available)}",
        )
    )


def build_messages(
    *,
    view: str,
    question: str = "",
    history: str = "",
    answer: str = "",
    question_available: bool = True,
    history_available: bool = True,
    audio_path: str | None = None,
    video_payload: Any | None = None,
    start_seconds: float | None = None,
    stop_seconds: float | None = None,
) -> list[dict[str, Any]]:
    view = str(view).lower()
    if view == "clinical":
        instruction = METER_CLINICAL_PROMPT
        text = render_clinical_input(
            question=question,
            history=history,
            answer=answer,
            question_available=question_available,
            history_available=history_available,
            audio_available=audio_path is not None,
            video_available=video_payload is not None,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
    elif view == "answer":
        instruction = METER_ANSWER_ONLY_PROMPT
        text = render_clinical_input(
            answer=answer,
            question_available=False,
            history_available=False,
            audio_available=audio_path is not None,
            video_available=video_payload is not None,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
    elif view == "context":
        instruction = METER_CONTEXT_ONLY_PROMPT
        text = render_clinical_input(
            question=question,
            history=history,
            question_available=question_available,
            history_available=history_available,
            audio_available=False,
            video_available=False,
        )
    else:
        raise ValueError(f"unknown METER-MH view: {view}")

    content: list[dict[str, Any]] = []
    if audio_path is not None:
        content.append({"type": "audio", "audio": str(audio_path)})
    if video_payload is not None:
        content.append({"type": "video", "video": video_payload})
    content.append({"type": "text", "text": f"{instruction}\n\n{text}"})
    return [{"role": "user", "content": content}]


def contract() -> dict[str, Any]:
    prompt_map = {
        "benchmark_anchor": BENCHMARK_ANCHOR_PROMPT,
        "clinical": METER_CLINICAL_PROMPT,
        "answer": METER_ANSWER_ONLY_PROMPT,
        "context": METER_CONTEXT_ONLY_PROMPT,
    }
    return {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": {name: sha256_text(value) for name, value in prompt_map.items()},
        "benchmark_information_retained": True,
        "raw_labels_in_prompt": False,
        "diagnostic_instruction": False,
        "clinical_score_prediction_in_prompt": False,
        "instrument_mapping_deferred_downstream": True,
        "missing_question_policy": "explicit QUESTION_UNAVAILABLE; never reconstruct",
        "missing_history_policy": "explicit HISTORY_UNAVAILABLE; never impute",
        "missing_modality_policy": "availability mask; missing is not observed absence",
        "timing_policy": "only supplied aligned boundaries; unknown is never zero-filled",
        "visual_policy": "official visual feature stream is stored separately downstream",
        "allowed_views": ["clinical", "answer", "context"],
    }


CONTRACT = contract()


if __name__ == "__main__":
    print(json.dumps(CONTRACT, ensure_ascii=False, indent=2))
