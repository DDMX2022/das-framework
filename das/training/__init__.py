"""Training and growth utilities for DAS forests.

The core ``das`` package proves isolated expert lifecycle operations. This
subpackage adds the "growing child" loop: teacher-generated lessons, candidate
expert training, regression checks, and audited accept/reject decisions.
"""

from .evaluator import (
    ExpertEvalSet,
    accuracy,
    clone_leaf,
    evaluate_candidate_suite,
    evaluate_leaf,
    evaluate_suite,
    router_accuracy,
    train_leaf,
)
from .growth import GrowthCycle, GrowthManager, GrowthPolicy, GrowthResult
from .teachers import (
    EndpointLLMTeacher,
    HashingTextEncoder,
    LLMTeacherError,
    LessonBatch,
    TeacherError,
    VectorTeacher,
    teacher_from_config,
)

__all__ = [
    "EndpointLLMTeacher",
    "ExpertEvalSet",
    "GrowthManager",
    "GrowthCycle",
    "GrowthPolicy",
    "GrowthResult",
    "HashingTextEncoder",
    "LLMTeacherError",
    "LessonBatch",
    "TeacherError",
    "VectorTeacher",
    "accuracy",
    "clone_leaf",
    "evaluate_candidate_suite",
    "evaluate_leaf",
    "evaluate_suite",
    "router_accuracy",
    "teacher_from_config",
    "train_leaf",
]
