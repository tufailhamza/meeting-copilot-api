"""Live WebSocket message contract.

Public surface for the inbound (client -> server) and outbound (server -> client) messages,
plus helpers for parsing/serializing and emitting JSON Schema for documentation.
"""

from src.live.schemas_live.common import (
    FactCheckVerdict,
    LiveModel,
    QuestionStatus,
    QuestionView,
    SessionUser,
    UserQuestions,
    new_id,
)
from src.live.schemas_live.inbound import (
    ChatMessageIn,
    InboundMessage,
    TranscriptSegment,
    TranscriptSegmentMessage,
    TranscriptSessionEndMessage,
    TranscriptSessionStartMessage,
    UserConnectedOnLiveScreen,
    inbound_json_schema,
    parse_inbound,
)
from src.live.schemas_live.outbound import (
    Answer,
    ChatMessageOut,
    ChatToken,
    CompanyDetectedOut,
    ErrorMessage,
    FactCheck,
    OutboundMessage,
    QuestionAdded,
    MarkQuestion,
    Risk,
    SessionReady,
    Suggestion,
    dump_outbound,
    outbound_json_schema,
)

__all__ = [
    # common
    "FactCheckVerdict",
    "LiveModel",
    "QuestionStatus",
    "QuestionView",
    "SessionUser",
    "UserQuestions",
    "new_id",
    # inbound
    "ChatMessageIn",
    "InboundMessage",
    "TranscriptSegment",
    "TranscriptSegmentMessage",
    "TranscriptSessionEndMessage",
    "TranscriptSessionStartMessage",
    "UserConnectedOnLiveScreen",
    "inbound_json_schema",
    "parse_inbound",
    # outbound
    "Answer",
    "ChatMessageOut",
    "ChatToken",
    "CompanyDetectedOut",
    "ErrorMessage",
    "FactCheck",
    "OutboundMessage",
    "QuestionAdded",
    "MarkQuestion",
    "Risk",
    "SessionReady",
    "Suggestion",
    "dump_outbound",
    "outbound_json_schema",
]
