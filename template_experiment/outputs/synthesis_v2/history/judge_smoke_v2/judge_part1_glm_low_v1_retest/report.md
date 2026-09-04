# GLM paired low-effort smoke retest

Same patient, evidence, prompt, presentation order, temperature, and 2,048-token cap. Reasoning changed from provider default to low; original responses are preserved.

```json
{
  "audit_id": "J-3001-P-1-c81915",
  "status": "complete",
  "error": "",
  "finish_reason": "stop",
  "request_id": "gen-1788513571-r59LJQs3ZHUaFKvqA1D6",
  "usage": {
    "completion_tokens": 268,
    "prompt_tokens": 1553,
    "total_tokens": 1821,
    "completion_tokens_details": {
      "audio_tokens": 0,
      "reasoning_tokens": 72
    },
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cache_write_tokens": 0,
      "cached_tokens": 0,
      "video_tokens": 0
    }
  },
  "reported_cost_usd": null,
  "attempts": [
    {
      "started_utc": "2026-09-04T09:19:30.794687+00:00",
      "status": "response_received",
      "finished_utc": "2026-09-04T09:19:50.965659+00:00",
      "elapsed_seconds": 20.17
    }
  ]
}
```

A valid result verifies this one request only. No full panel ran. Do not merge with the original collection as if both used identical runtime settings.
