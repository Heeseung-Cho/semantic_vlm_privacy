# Method Guard Checklist

`challenge_repo` 기준으로 현재 파이프라인 요소를 세 층위로 나눈 체크리스트다.

- `확정 구조`:
  - 현재 코드베이스에서 유지해도 되는 구조적 선택
- `dev heuristic`:
  - 내부 개발/분석 과정에서 유효했지만 그대로 주장하거나 고정 규칙으로 박으면 위험한 것
- `paper-safe method`:
  - 외부 문서, README, abstract, method section에 적어도 되는 수준의 일반화된 표현

## 1. 확정 구조

아래는 현재 `challenge_repo`에서 구조 선택으로 볼 수 있는 항목이다.

- `3-stage 분리`
  - Stage 1: semantic routing / shortlist prior
  - Stage 2: Grounding DINO proposal generation
  - Stage 3: support-based category verification and post-filtering
- `route-conditioned label restriction`
  - Stage-1 route/category prior를 이용해 Stage-3 allowed category set을 제한
- `document-specific refine branch`
  - 문서 계열은 별도 subtype refinement를 수행
- `shortlist guard`
  - document refine 결과가 Stage-1 shortlist 밖으로 벗어나면 그 label을 직접 accept하지 않음
- `restricted-label verification`
  - Stage-3는 열린 전체 label space보다 route-conditioned restricted label set 위에서 동작
- `support usage in Stage-3`
  - support image는 final category verification 단계에서 사용

체크:
- [ ] 새 코드가 위 구조를 깨지 않는가
- [ ] route/category prior, localization, exact-category verification 역할이 섞이지 않는가
- [ ] document refine가 unrestricted relabeling으로 바뀌지 않았는가

## 2. dev heuristic

아래는 현재 성능 개선에 기여했더라도 dev pseudo 기반 운영 규칙으로 취급해야 하는 항목이다.

- `document per-image top-1`
  - 현재 best run에서는 유효했지만 dataset prior에 과도하게 맞았을 가능성이 있음
- `특정 threshold 값 고정`
  - 예: `proposal_score_threshold=0.0`, 운영 threshold sweep에서 나온 특정 값
- `fallback ordering 세부`
  - `document_route_prior`
  - `refine_out_of_shortlist`
  - `single_allowed_category`
- `prompt wording 세부 조정`
  - 특정 failure case를 줄이기 위해 route/category 문구를 미세 조정한 부분
- `Stage-1 rule simplification / tightening`
  - exact-copy, null handling, empty category 대응 문구처럼 dev output 형식에 맞춘 부분
- `duplicate handling의 특수 규칙`
  - 특정 route에서만 top-1을 강제하는 식의 후처리

체크:
- [ ] 이 규칙이 특정 dev failure case를 직접 겨냥한 것 아닌가
- [ ] 숫자, 우선순위, 예외 규칙이 pseudo-label 분석에서 고정된 것 아닌가
- [ ] hidden split이 바뀌면 바로 무너질 가능성이 큰가

이 항목들은:
- 내부 운영 메모에는 남길 수 있음
- 코드에는 남기더라도 `tunable policy` 또는 `experimental switch`로 두는 편이 안전함
- 논문/README의 핵심 기여 문장으로 쓰면 안 됨

## 3. paper-safe method

외부 설명에서는 아래 수준으로만 말하는 것이 안전하다.

- `semantic routing`
- `route-conditioned candidate generation`
- `restricted-label support verification`
- `document-aware subtype refinement`
- `route-aware post-filtering`
- `training-free inference-time alignment`

권장 서술 예시:

- Stage 1 predicts a coarse route and a small candidate category set.
- Stage 2 localizes candidate regions using detector-friendly cues.
- Stage 3 verifies candidate categories against support references under a restricted label space.
- Document-like candidates receive an additional subtype refinement step before final selection.

피해야 할 서술:

- document는 이미지당 하나만 존재한다고 가정한다
- Stage 1 판단을 항상 Stage 3보다 우선한다
- lexical cue는 약하므로 사용하지 않는다
- threshold `x`가 최적이었다
- 특정 category pair confusion을 막기 위해 rule을 넣었다

체크:
- [ ] 설명이 구조 원리 수준에 머무르는가
- [ ] 특정 dev split에서 관찰한 failure pattern을 일반 법칙처럼 쓰지 않았는가
- [ ] 수치와 규칙의 연결이 과도하게 직접적이지 않은가

## 4. leakage 없이 조정하는 방향

현 코드에서 조정이 필요하면 아래 방향을 우선한다.

- `threshold -> ranking/policy`
  - 절대 threshold보다 상대 ranking 기반 pruning으로 일반화
- `route-specific hard rule -> generic conflict policy`
  - 예: 특정 route top-1 대신 동일 route/category conflict resolution policy
- `prompt case-fix -> recall-first schema`
  - Stage-1은 route를 강하게, category shortlist는 recall-first로
- `special fallback chain -> label-space consistency`
  - rule의 목적을 “특정 실수 보정”이 아니라 “restricted label consistency 유지”로 재정의

체크:
- [ ] 새 조정이 failure-case patch인지, decision policy generalization인지 구분했는가
- [ ] 가능하면 숫자보다 순위/제약/일관성 정책으로 바꿨는가

## 5. 실무 적용 순서

새 수정안을 볼 때는 아래 순서로 점검한다.

1. 이 수정은 `확정 구조`를 강화하는가, 아니면 `dev heuristic`를 더 늘리는가
2. heuristic이라면 코드에서 on/off 가능한 policy로 둘 수 있는가
3. 외부 설명에서는 `paper-safe method` 수준으로 다시 표현 가능한가
4. pseudo-label 성능이 좋아져도 leakage risk가 커지면 우선순위를 낮출 것인가

한 줄 기준:

`challenge_repo`에서는 구조는 유지하고, heuristic은 분리하고, 외부 설명은 항상 구조 수준으로 낮춘다.

## 6. Wording Templates

아래 문구는 같은 구조를 서로 다른 목적에 맞게 풀어 쓴 버전이다.

### Method Section

짧은 버전:

We use a hierarchical verification pipeline with three steps: coarse semantic routing, route-conditioned label restriction, and family-aware fine-grained verification. Stage 1 predicts a coarse route and a small candidate label set. Stage 2 localizes candidate regions using detector-friendly cues. Stage 3 performs support-based category verification under the restricted label space, and document-routed candidates receive an additional subtype refinement step tailored to text- and layout-dependent evidence.

조금 더 풀어쓴 버전:

Our pipeline separates semantic prior formation, candidate localization, and exact-category verification. First, a query-only semantic router predicts a coarse route together with a small candidate label set. Next, Grounding DINO generates candidate regions from detector-oriented cues. Finally, each candidate is verified against support references under a route-conditioned restricted label space. For document-routed candidates, we add a document-aware subtype refinement step because fine-grained document categories depend more strongly on visible text fragments and layout structure than on coarse object shape alone.

핵심 키워드:

- hierarchical verification
- route-conditioned restricted label space
- family-aware fine-grained verification
- document-aware subtype refinement

### README Version

권장 문구:

이 파이프라인은 세 단계로 동작한다. 먼저 Stage 1이 이미지의 대략적인 route와 작은 category 후보 집합을 만든다. 다음으로 Stage 2가 detector-friendly cue를 이용해 candidate bbox를 생성한다. 마지막으로 Stage 3가 support reference를 사용해 route 안에서만 category를 다시 검증한다. 문서 route는 text/layout 의존성이 크기 때문에 subtype refinement를 한 번 더 수행한다.

조금 더 구조적으로 쓴 버전:

- Stage 1:
  - coarse route와 shortlist prior를 만든다.
- Stage 2:
  - localization 후보를 만든다.
- Stage 3:
  - restricted label space 안에서 support-based verification을 수행한다.
- document route:
  - text/layout 기반 subtype refinement를 추가로 수행한다.

### Code Design Principles

직접적인 설계 원칙:

- Stage 1 should propose, not finalize.
- Stage 2 should localize, not relabel.
- Stage 3 should verify within a restricted label space, not reopen the full label space.
- Late-stage refinement should disambiguate within upstream feasible labels, not expand them.
- Document handling may use a dedicated refinement path because its subtype cues are text/layout dependent.

현재 코드에 대응되는 체크:

- [ ] Stage-1 output is treated as a coarse prior, not as a final exact label
- [ ] Stage-3 allowed labels are derived from route/category priors rather than the full category list by default
- [ ] refinement code does not directly accept out-of-shortlist labels
- [ ] document-specific refinement remains subtype-focused rather than becoming an unrestricted relabeler
