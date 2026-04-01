# ACERVO v0.4.0 Diagnostic

## Component Health

| Component | Score |
|-----------|-------|
| s1_intent | 78% |
| s2_activation | 56% |
| s3_budget | 32% |
| s3_quality | 81% |

## S1 Failures: 9
- Turn 2: expected=overview, got=specific
- Turn 6: expected=chat, got=overview
- Turn 19: expected=chat, got=specific
- Turn 1: expected=overview, got=specific
- Turn 8: expected=chat, got=specific
- Turn 14: expected=overview, got=specific
- Turn 1: expected=overview, got=specific
- Turn 2: expected=overview, got=specific
- Turn 9: expected=chat, got=overview

## Cross-Matrix (Category x Component)

| Category | S1 Intent | S2 Activation | S3 Budget | S3 Quality | Score |
|----------|-----------|---------------|-----------|------------|-------|
| RESOLVE | 73% | 50% | 0% | 100% | 100% |
| GROUND | 80% | 50% | 0% | 77% | 92% |
| RECALL | n/a | n/a | n/a | 33% | 67% |
| FOCUS | 73% | 38% | 40% | 89% | 100% |
| ADAPT | 89% | 100% | 100% | 78% | 100% |