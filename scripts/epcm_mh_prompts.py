"""EPCM-MH HumanOmniV2 prompt contract.

The prompt is observation-first: it asks HumanOmniV2 to preserve measurable
content and response process information, while leaving clinical measurement
to participant-level out-of-fold heads.  The old Benchmark representation is
kept as an immutable anchor and is never overwritten by this view.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA = "epcm_mh_prompt_contract_v1"
PROMPT_VERSION = "EPCM-MH observation-first clinical measurement v1.0"

QUESTION_UNAVAILABLE = "[QUESTION_UNAVAILABLE]"
HISTORY_UNAVAILABLE = "[HISTORY_UNAVAILABLE]"
RESPONSE_UNAVAILABLE = "[RESPONSE_UNAVAILABLE]"
AUDIO_UNAVAILABLE = "[AUDIO_UNAVAILABLE]"
VIDEO_UNAVAILABLE = "[VIDEO_UNAVAILABLE]"


# This text is intentionally retained as the Benchmark anchor.  The EPCM
# cache is an additional view; it is not allowed to replace this representation.
BENCHMARK_ANCHOR_PROMPT = (
    "Encode this clinical interview response without diagnosing the participant. "
    "Represent the literal content, vocal delivery, timing, and uncertainty "
    "available in the input."
)


EPCM_CLINICAL_PROMPT = """You are the EPCM-MH observation encoder for a clinical interview.
Encode the supplied observation faithfully. You are not a clinician, you do not
diagnose the participant, and you do not output a screening decision. Return no
explanation or generated answer; produce only the internal multimodal
representation.

The input is divided into audited blocks. Keep the speaker roles, time order,
and availability mask explicit:
[INTERVIEWER_CONTEXT] is the question or preceding interviewer speech.
[PRIOR_TURNS] contains only earlier supplied turns.
[PARTICIPANT_RESPONSE] is the current participant response.
[MEDIA_AND_TIMING] describes the aligned audio/video window and its boundaries.
[AVAILABILITY_MASK] states which sources are actually present.

Preserve all directly observable information, including:
1. Literal semantics: negation, uncertainty, modality, specificity, temporal
   reference, self-correction, repair, topic change, coherence, elaboration,
   contradiction, and whether the requested content was answered.
2. Context-response relation: direct answer, partial answer, expansion, repair
   request, redirection, contradiction, unresolved response, or explicit
   refusal. Do not treat any relation as pathological by itself.
3. Response process: onset latency, silent and filled pauses, breath or sigh
   only when audible, continuation, interruption, restart, repair, completion,
   abandonment, turn overlap, speech rate, rhythm, and within-answer change.
4. Audio observations: pitch movement, energy, articulation, voice quality,
   rhythm, and temporal variation when present in the supplied audio.
5. Visual observations: gaze direction, head orientation, facial action,
   visible motion, and their timing only when a usable video stream is supplied.
6. Clinical-measurement evidence candidates, without clinical inference:
   reported mood, interest or pleasure, energy, cognition, self-evaluation,
   future orientation, sleep or appetite, functioning, and safety-related
   content. Record each only when directly expressed or clearly observable in
   the input, retaining polarity, intensity, specificity, time horizon, and
   source modality.
7. Nuisance and uncertainty: explicit situational events, physical symptoms,
   interviewer effects, accent or recording artifacts, ambiguity, missing
   context, and evidence conflicts. Preserve the event and uncertainty; do not
   decide that it has a disease cause.

Use separate internal slots for CONTENT, CONTEXT_RELATION, RESPONSE_PROCESS,
AUDIO, VISUAL, CLINICAL_EVIDENCE, and UNCERTAINTY. Do not collapse a short
answer, silence, low energy, negative wording, gaze aversion, delayed onset,
or uncertainty into a diagnosis. These are observations whose meaning must be
learned and tested downstream.

Never infer or encode a label, PHQ/PCL/MADRS score, diagnosis, identity,
demographic attribute, personality, motive, deception, causal explanation,
treatment need, or private mental state. Never use participant ID, file name,
dataset, split, future turns, or any supplied clinical label. Never guess a
missing question, missing history, unobserved facial detail, or off-window
event. An unavailable modality is not an observed absence.

The representation must support two separate downstream uses: estimating a
continuous clinical measurement from a complete participant sequence, and
auditing which observable evidence was sufficient for that estimate. Do not
perform either task in the prompt and do not replace the protected Benchmark
anchor representation."""


EPCM_ANSWER_ONLY_PROMPT = """You are the protected EPCM-MH answer-only observation encoder.
Encode only the supplied participant response and its aligned audio/video
window. Do not use an interviewer question, previous turn, future turn,
participant ID, dataset, split, label, or clinical score.

Preserve literal meaning, negation, uncertainty, specificity, temporal
reference, topic change, coherence, elaboration, correction, repair,
interruption, completion, response onset, silent and filled pauses, audible
breath or sigh, speech rate, rhythm, pitch movement, energy, articulation,
voice quality, visible motion, gaze, head orientation, and their timing when
actually present. Preserve the difference between a missing modality and an
observed absence.

Do not infer diagnosis, symptoms, severity, identity, demographics,
personality, motive, deception, cause, treatment need, or private mental state.
Brevity, silence, low energy, negative wording, uncertainty, delayed onset,
or reduced eye contact are observations only. Return no text; produce only the
internal representation."""


EPCM_CONTEXT_ONLY_PROMPT = """You are the protected EPCM-MH context-only observation encoder.
Encode only the supplied interviewer question, interviewer speech, and prior
turns. Preserve speaker roles, order, requested information, topic, repair
prompts, and the expected response slot. Do not encode the current participant
response, its audio, its video, or any future turn.

Do not infer or predict diagnosis, symptoms, severity, identity, demographics,
personality, motive, deception, cause, treatment need, or private mental state.
Do not invent a missing question or reconstruct it from a template. Return no
text; produce only the internal representation."""


EPCM_SOURCE_DIALOGUE_PROMPT = """You are a source-domain conversational response-process encoder.
The data may be acted or non-clinical dialogue. Learn only general observable
conversation dynamics: turn transitions, response onset, pauses, fillers,
repairs, overlap, completion, topic continuation, affective delivery, and
speaker-context relations. Do not treat emotion labels as depression or PTSD
labels, and do not infer clinical states. Preserve the availability mask and
the difference between an unobserved event and an observed absence. Return no
text; produce only the internal representation."""


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
    """Render an auditable input without fabricating missing information."""

    start = "UNKNOWN" if start_seconds is None else f"{float(start_seconds):.3f}"
    stop = "UNKNOWN" if stop_seconds is None else f"{float(stop_seconds):.3f}"
    return "\n".join(
        (
            "[EPCM_INPUT_V1]",
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
    """Build HumanOmniV2 messages for a versioned EPCM view."""

    view = str(view).lower()
    if view == "clinical":
        instruction = EPCM_CLINICAL_PROMPT
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
        instruction = EPCM_ANSWER_ONLY_PROMPT
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
        instruction = EPCM_CONTEXT_ONLY_PROMPT
        text = render_clinical_input(
            question=question,
            history=history,
            question_available=question_available,
            history_available=history_available,
            audio_available=False,
            video_available=False,
        )
    elif view == "source_dialogue":
        instruction = EPCM_SOURCE_DIALOGUE_PROMPT
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
    else:
        raise ValueError(f"unknown EPCM view: {view}")

    content: list[dict[str, Any]] = []
    if audio_path is not None:
        content.append({"type": "audio", "audio": str(audio_path)})
    if video_payload is not None:
        content.append({"type": "video", "video": video_payload})
    content.append({"type": "text", "text": f"{instruction}\n\n{text}"})
    return [{"role": "user", "content": content}]


def contract() -> dict[str, Any]:
    prompts = {
        "benchmark_anchor": BENCHMARK_ANCHOR_PROMPT,
        "clinical": EPCM_CLINICAL_PROMPT,
        "answer": EPCM_ANSWER_ONLY_PROMPT,
        "context": EPCM_CONTEXT_ONLY_PROMPT,
        "source_dialogue": EPCM_SOURCE_DIALOGUE_PROMPT,
    }
    return {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": {name: sha256_text(value) for name, value in prompts.items()},
        "raw_labels_in_prompt": False,
        "diagnostic_instruction": False,
        "clinical_measurement_in_prompt": False,
        "missing_question_policy": "explicit sentinel; never impute",
        "missing_history_policy": "explicit sentinel; never impute",
        "missing_modality_policy": "availability mask; missing is not absence",
        "timing_policy": "only supplied aligned window; timestamps stored separately",
        "protected_anchor": "Benchmark cache remains immutable and is concatenated downstream",
        "allowed_views": ["clinical", "answer", "context", "source_dialogue"],
        "source_label_policy": "MELD/emotion labels are never clinical supervision",
    }


CONTRACT = contract()


if __name__ == "__main__":
    print(json.dumps(CONTRACT, ensure_ascii=False, indent=2))
