"""
Compliance monitoring for call-center conversations.

Analyzes conversation in real-time to detect:
- Agent pushing credit card offers
- Customer expressing disinterest
- Agent continuing to push despite customer disinterest

Uses Gemini to identify speakers and check for compliance violations.
"""

import asyncio
import json
import time
from typing import Optional, List, Dict, Any

from loguru import logger
from google import genai
from google.genai import types


# --------------------------------------------------------------------------- #
#  Compliance analysis prompt                                                  #
# --------------------------------------------------------------------------- #
COMPLIANCE_PROMPT = """\
You are a call-center compliance monitor. Analyze the following conversation \
transcript between a call-center agent and a customer.

Your tasks:
1. **Speaker Identification**: Identify which speaker is the AGENT and which \
is the CUSTOMER based on the conversation content.
   - The AGENT typically: introduces themselves, represents the company, \
offers products/services, follows scripts, asks verification questions
   - The CUSTOMER typically: calls for help, asks questions, responds to \
offers, may express interest or disinterest

2. **Compliance Check**: Check ONLY for credit card offer related violations:
   - Flag if the AGENT mentions or pushes credit card offers/products
   - Flag if the CUSTOMER expresses disinterest in credit card offers \
(e.g., "no thanks", "not interested", "I don't want", "no", etc.)
   - **CRITICAL**: Flag a HIGH severity alert if the AGENT continues to push \
credit card offers AFTER the CUSTOMER has already expressed disinterest

Return your analysis as a JSON object with this exact structure:
{
  "segments": [
    {"index": 0, "role": "agent", "text": "original text"},
    {"index": 1, "role": "customer", "text": "original text"}
  ],
  "alerts": [
    {
      "severity": "high",
      "title": "Short alert title",
      "description": "Detailed description of the violation",
      "segment_indices": [2, 4]
    }
  ]
}

SEVERITY RULES:
- "high": Agent continues pushing credit card offers after customer expressed \
disinterest. This is a serious compliance violation.
- "medium": Agent pushing credit card offers aggressively or repeatedly
- "low": Agent mentions a credit card offer (informational, not necessarily \
a violation)

IMPORTANT:
- Only return alerts that are actually detected; return empty array if no issues
- ALWAYS return valid JSON only, no extra text or markdown
- Be conservative: only flag clear violations, not ambiguous cases
- The "segments" array must include ALL segments with their identified roles

Conversation transcript (segments numbered):
"""


# --------------------------------------------------------------------------- #
#  ComplianceMonitor class                                                     #
# --------------------------------------------------------------------------- #
class ComplianceMonitor:
    """Monitors call-center conversations for compliance violations."""

    def __init__(
        self,
        client: genai.Client,
        llm_model: str,
        min_analysis_interval: float = 3.0,
    ):
        self.client = client
        self.llm_model = llm_model
        self.segments: List[str] = []
        self.speaker_roles: Dict[int, str] = {}
        self.alerts: List[Dict[str, Any]] = []
        self._alert_id_counter = 0
        self._analysis_lock = asyncio.Lock()
        self._last_analysis_time = 0.0
        self._min_analysis_interval = min_analysis_interval
        self._last_analyzed_count = 0

    def add_segment(self, text: str) -> int:
        """Add a new transcript segment. Returns the segment index."""
        self.segments.append(text.strip())
        return len(self.segments) - 1

    async def should_analyze(self) -> bool:
        """Check if we should run analysis now."""
        if len(self.segments) < 2:
            return False
        if len(self.segments) == self._last_analyzed_count:
            return False
        elapsed = time.time() - self._last_analysis_time
        if elapsed < self._min_analysis_interval:
            return False
        return True

    async def analyze(self) -> Optional[Dict[str, Any]]:
        """Run compliance analysis on the accumulated conversation.

        Returns a dict with ``speaker_roles``, ``new_alerts``, and
        ``all_alerts`` keys, or ``None`` if analysis was skipped.
        """
        async with self._analysis_lock:
            if not await self.should_analyze():
                return None

            self._last_analysis_time = time.time()
            self._last_analyzed_count = len(self.segments)

            transcript_text = "\n".join(
                f"[{i}] {seg}" for i, seg in enumerate(self.segments)
            )
            prompt = COMPLIANCE_PROMPT + transcript_text

            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.llm_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )

                result_text = (response.text or "").strip()
                result = json.loads(result_text)

                # Update speaker roles
                if "segments" in result:
                    for seg in result["segments"]:
                        idx = seg.get("index", -1)
                        role = seg.get("role", "unknown")
                        if 0 <= idx < len(self.segments):
                            self.speaker_roles[idx] = role

                # Track new alerts (deduplicate by title + description)
                existing_keys = {
                    (a["title"], a["description"]) for a in self.alerts
                }
                new_alerts = []
                for alert in result.get("alerts", []):
                    key = (alert.get("title", ""), alert.get("description", ""))
                    if key not in existing_keys:
                        self._alert_id_counter += 1
                        alert["id"] = f"alert_{self._alert_id_counter}"
                        alert["timestamp"] = time.strftime("%H:%M:%S")
                        self.alerts.append(alert)
                        new_alerts.append(alert)
                        existing_keys.add(key)

                logger.info(
                    f"Compliance analysis: {len(self.speaker_roles)} roles, "
                    f"{len(new_alerts)} new alerts"
                )

                return {
                    "speaker_roles": {
                        str(k): v for k, v in self.speaker_roles.items()
                    },
                    "new_alerts": new_alerts,
                    "all_alerts": self.alerts,
                }

            except json.JSONDecodeError as e:
                logger.error(f"Compliance JSON parse error: {e}")
                return None
            except Exception as e:
                logger.error(f"Compliance analysis error: {e}")
                return None
