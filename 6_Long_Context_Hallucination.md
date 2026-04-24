# 📙 6강. LLM Long Context & Hallucination Reduction

> **핵심 메시지**: "Longer context"와 "Retrieval augmentation"은 **경쟁 종교가 아니다**.
> 둘은 서로 다른 failure mode를 해결한다.
> 가장 믿을 만한 시스템은 **hybrid**: 모델 내부에서 long-position 사용을 안정화 + 모델 외부에서 증거를 제어.

---

## 🎯 학습 목표 (Learning Goals)

- Long-context 처리의 근본 문제 이해 (attention, positional encoding)
- **Framework A**: Long-context를 position-geometry 보존 문제로
- **Framework B**: Hallucination 감소를 evidence-grounded selective generation으로
- RAG, Self-RAG, CRAG, GraphRAG 등 최신 retrieval 계열 이해
- Long-context + hallucination 감소가 **pipeline 문제**임을 인식

---

## 1. 두 가지 큰 프레임워크 ⭐⭐⭐

### 1-1. Framework A: Position-Geometry Preservation
- **목표**: 입력 길이와 답변 위치가 커져도 long-context 성능이 무너지지 않게
- **대표 방법**: PI, YaRN, PoSE, LongLoRA, LongAlign, LongRoPE, LongRoPE2

### 1-2. Framework B: Evidence-Grounded Selective Generation
- **목표**: 모든 답변 claim이 evidence로 뒷받침되게 (hallucination 감소)
- **대표 방법**: Self-RAG, CRAG, GraphRAG, LongCite, SelfCite

### 💡 통합 관점
- Framework A = **모델 내부**에서 long position 사용 안정화
- Framework B = **모델 외부**에서 evidence pipeline 제어
- 최고 시스템은 **두 가지를 모두 사용**하는 hybrid

---

## 2. Core Concepts: 기초

### 2-1. Next-Token Prediction
자기회귀 LM 학습 목표:
```
L = -Σ_t log p_θ(w_t | w_<t)
```
- 전체 시퀀스의 log-likelihood를 chain-rule로 분해
- Cross-entropy는 이 분해의 음수 log

### 2-2. Teacher Forcing & Causal Mask
```
M_ij = 0 if j ≤ i else -∞
```
- **학습 중**: 모델이 true prefix `w_<t`를 봄 (teacher forcing)
- **생성 중**: 자기가 생성한 토큰에 의존
- 이 **train-test gap**이 fluent error chain 생성 원인

### 2-3. Attention 복습
```
Attn(Q, K, V) = softmax(QKᵀ/√d_k + M) V
```
- 비용: O(L² d_k) → long context의 첫 bottleneck

---

## 3. 위치 정보 (Positional Encoding) 계보 ⭐

### 3-1. 왜 위치가 필요한가
- "dog bites man" ≠ "man bites dog"
- Attention은 permutation-invariant (위치 정보 없으면 순서 모름)

### 3-2. 2가지 큰 설계 선택
1. **Additive PE**: `h_i = x_i + p_i` (원 Transformer)
2. **Relative-position scoring**: attention score가 거리/phase에 의존 (ALiBi, RoPE)

### 3-3. 주요 PE 계열
| 방법 | 핵심 |
|------|------|
| **Absolute (sinusoidal)** | 고정된 sin/cos 벡터 |
| **Learned PE** | 위치별 임베딩 학습 |
| **Relative PE** | 두 토큰 간 거리만 사용 |
| **ALiBi** | Linear bias로 거리에 페널티 |
| **RoPE (Rotary PE)** | 위치를 회전 행렬로 표현, 상대 거리 정보 자연스럽게 보존 |

### 3-4. 왜 RoPE가 중요한가
- 현대 LLM(LLaMA 등)의 사실상 표준
- Phase 기반이라 **내삽/외삽**이 수학적으로 정의됨
- 이 덕분에 대부분 long-context 확장 방법이 RoPE 위에서 동작

---

## 4. Context Window vs Long-Context Skill ⭐

### ⚠️ 반드시 구분
- **Nominal context length**: 엔지니어링 속성. "몇 토큰 받나"
- **Effective long-context skill**: 행동적 속성. "실제로 잘 쓰나"

> 모델이 128k 토큰을 받아도 잘 못 쓰면 의미 없음!

### Long-context 작업의 이중 과제
1. Token capacity (얼마나 담을 수 있나)
2. 그 capacity 안에서 **position의 안정적 사용**

---

## 5. 역사적 계보: Memory Extension 방법들

### 5-1. Transformer (2017, Vaswani)
- Dense self-attention, parallel training
- Bottleneck: O(L²) in sequence length

### 5-2. Transformer-XL (2019)
```
H̃^(r-1)_τ = [mem^(r-1)_{τ-1} ; H^(r-1)_τ]
```
- **Segment recurrence**: 이전 segment의 hidden state를 cache로 재사용
- Attention context를 재계산 없이 확장

### 5-3. Compressive Transformer (2020)
```
C_t = [f_c(M_t) ; C^trim_{t-1}]
```
- 오래된 memory는 **압축** (버리지 않고)
- 최근 memory는 고해상도 유지
- Two-timescale memory (짧은 것 + 긴 압축)

### 5-4. ALiBi (2022)
- Attention score에 linear bias로 거리 페널티
- 외삽(extrapolation) 시에도 안정적

### 5-5. RoPE (2021, 2023)
- Rotary position embedding
- 현대 LLM의 표준

### 5-6. 외부 증거 시스템 계보
| 방법 | 연도 | 핵심 |
|------|------|------|
| **REALM** | 2020 | Masked LM + retrieval |
| **RAG** | 2020 | Retrieve + Generate |
| **FiD** | 2021 | Fusion-in-Decoder |
| **RETRO** | 2022 | Chunk 단위 retrieval |
| **Memorizing Transformer** | 2022 | 학습 가능한 memory bank |

---

## 6. Framework A 상세: Long-Context 확장 ⭐⭐⭐

### 6-1. 공통 템플릿: Generic Remapped Attention
```
ℓ_ij^(h)(ψ) = ⟨RoPE(q_i, g_ψ(m_i)), RoPE(k_j, g_ψ(m_j))⟩ / √d_k + b_ij^(h)
```
- `g_ψ`: remapping function
- `m_i, m_j`: raw positions
- PI, YaRN, LongRoPE, LongRoPE2가 **`g_ψ`를 어떻게 선택하느냐**가 주요 차이

### 6-2. Design Goal for g
```
g(m) ≈ m  for short contexts
g(m)이 큰 m을 안정 범위로 압축  for long contexts
```

### 6-3. Positional Interpolation (PI) — 기본 recipe ⭐
```
g_PI(m) = m · (L_train / L_test)
```
- 모든 상대 phase를 `L_train/L_test` 배로 축소
- **왜 효과적**: 큰 unseen offset → 익숙한 범위로 매핑
- Modest continued training으로 잘 작동 → 커뮤니티 표준 baseline

### 6-4. YaRN (Yet another RoPE extension)
- PI의 개선
- **주파수별로 다른 interpolation** (고주파는 유지, 저주파는 확장)
- 짧은 context 품질을 덜 해침

### 6-5. LongRoPE / LongRoPE2
- Non-uniform search로 최적 remapping 찾기
- **2M 토큰**까지 확장 가능

### 6-6. PoSE (Position Skip-wisE)
- `g(m)`을 추론 시 바꾸지 않고, **학습 분포**를 바꿈
- 학습 예제를 chunk로 나누고 chunk별 position offset 할당
- 짧은 학습 window 안에서 큰 positional gap 학습

### 6-7. LongLoRA
```
min_{Δθ}  L_LM(θ₀ + Δθ ; g(m), long data)
```
- `Δθ`: low-rank (LoRA) 업데이트
- 학습 중 **sparse (shifted sparse) attention** 사용
- 추론 시엔 dense 가능
- 제한된 하드웨어에서 long-context tuning 가능

### 6-8. LongAlign
- **Long instruction data**, packing, batching, loss balancing
- Long-context가 positional만의 문제가 아니라는 관점
- Instruction tuning에 긴 예제 없으면 extended model이 여전히 실패

### 6-9. Training-Time Support 계열 정리

| 방법 | 바꾸는 것 | 도움 |
|------|-----------|------|
| **PoSE** | 학습 position 분포 | 짧은 window에서 큰 gap 시뮬레이션 |
| **LongLoRA** | Adaptation 비용 | Sparse + LoRA로 저렴한 확장 |
| **LongAlign** | Long instruction tuning | Extended model도 instruction 따르게 |

### 6-10. Inference-Time Memory 계열
| 방법 | 핵심 |
|------|------|
| **StreamingLLM** | Anchor tokens + rolling local cache |
| **Infini-attention** | Local attention + compressed memory channel |
| **ReAttention** | Training-free top-k preselection |

---

## 7. Framework B 상세: Hallucination Reduction ⭐⭐⭐

### 7-1. Hallucination 정의와 유형
> **Hallucination**: 생성된 claim이 뒷받침되지 않거나, 모순되거나, 여러 출처에서 잘못 융합됨

**주요 grounded-QA 오류 유형**:
- **Unsupported insertion**: 근거 없는 삽입
- **Contradiction**: 증거와 모순
- **Conflation**: 여러 출처 잘못 합침
- **Citation error**: 잘못된 인용

### 7-2. 단순 Faithfulness Score ⭐
```
supp(c, E) = 1 if evidence E supports claim c else 0
F(y, E) = (1/m) Σ_{j=1}^m supp(c_j, E)
```
- `c_j`: 답변 y의 atomic claim
- **Claim-level scoring**이 whole-answer accuracy보다 정밀

### 7-3. Long Context vs Retrieval 비교 ⭐

| 구분 | Long Context Window | Retrieval |
|------|---------------------|-----------|
| 방식 | 더 많은 토큰을 직접 읽음 | 큰 corpus에서 검색, 작은 evidence set 유지 |
| 유리 | 증거가 이미 한 곳에 있을 때 | 몇 snippet만 중요할 때 |
| 실패 모드 | 올바른 위치 놓침 | 핵심 snippet이 retrieve 안 됨 |

### 7-4. Retriever Score 공식 ⭐
```
r(x, d) = u(x)ᵀ v(d)
z_{1:k} = TopK_{d∈D} r(x, d)
```
- `u, v`: 학습된 embedding 함수
- Dense retrieval (embedding 유사도) vs Lexical retrieval (BM25 등)

---

## 8. RAG 계열 역사와 주요 방법

### 8-1. RAG 기본 (Lewis et al., 2020)
```
p(y|x) = Σ_{z∈D} p_η(z|x) p_θ(y|x,z)
```
- Token-level variant:
```
p(y_t | x, y_<t) = Σ_{z∈D} p_η(z|x) p_θ(y_t | x, z, y_<t)
```
- **핵심 통찰**: RAG는 prompt engineering이 아님. Evidence를 explicit variable로 만들어 **conditional distribution 자체를 바꿈**

### 8-2. FiD (Fusion-in-Decoder, Izacard & Grave 2021)
- 여러 passage를 **독립적으로 encode**
- Decoder가 cross-attention으로 통합
- Document probability marginalization 대신 **architectural decision**

### 8-3. Self-RAG (Asai et al., 2024) ⭐
**핵심 아이디어**: Retrieval을 정적 pipeline이 아닌 **모델이 선택하는 action**으로

```
action a_t ∈ {retrieve, continue, critique}
```
- 모델이 **언제 retrieve**할지 결정
- **Reflection tokens**로 evidence와 answer 품질 신호 방출
- 암묵적 objective: policy-conditioned version of RAG (z를 쓸지, 얼마나 쓸지 결정)

### 8-4. CRAG (Corrective RAG, Yan et al., 2024) ⭐
**핵심 아이디어**: Retrieval 품질이 나쁘면 blindly 조건부 생성하면 해로움

```
query x → retrieve z_{1:k}
→ evaluate quality q(z_{1:k}|x)
→ correct ẑ_{1:k}
→ generate y
```
- Retrieval **reliability를 명시적으로 모델링**

### 8-5. Self-RAG vs CRAG 핵심 차이
- **Self-RAG**: retrieval을 generation **내부**에서 adaptive하게
- **CRAG**: retrieval을 generation **이전**에 robust하게 (품질 체크 + 수정)

### 8-6. GraphRAG (Edge et al., 2024)
**핵심 아이디어**: 일반 passage retrieval은 **global corpus question**에 약함

```
corpus → G = (V, E) → community 클러스터링
→ community별 요약
→ query 시 community-level partial answer 결합
```
- 문서-level 합성이 필요한 질문에 유리 ("이 컬렉션의 주요 테마는?")

### 8-7. LongRAG
- Long-context LLM과 retrieval의 결합
- Retrieval과 long context는 상호보완적

---

## 9. Verification & Revision 계열

### 9-1. RARR (2023)
- LM이 자기 답을 external evidence로 **research & revise**
- Post-hoc factual revision을 practical pipeline으로

### 9-2. Chain-of-Verification (CoVe, 2023/24)
- 검증을 명시적 체크 단계로 분해
- 구조화된 self-checking이 hallucination 감소

### 9-3. SelfCheckGPT (2023)
- **여러 샘플링 결과의 inconsistency**로 hallucination 탐지
- Retrieval label 없이 zero-resource black-box detection

---

## 10. Citation 계열: LongCite, SelfCite

### 10-1. LongBench-Cite / LongCite (2024/25)
- Long-context QA에서 **fine-grained citation** 평가
- 주장-span 수준에서 support 품질 측정
- Citation quality를 mainstream evaluation으로

### 10-2. 왜 Citation이 중요한가
- Auditable 답변 시 핵심
- "정답 맞음"을 넘어 "근거가 맞음"까지 평가

---

## 11. 평가 Benchmark 계열

### 11-1. Framework A (Long-Context) 평가
| 벤치마크 | 타겟 |
|---------|------|
| **Needle-in-Haystack** | 긴 문서에 숨긴 정보 찾기 |
| **RULER** | 실제 context 능력 측정 |
| **LongBench v2** | 실전 long-context reasoning |

### 11-2. Framework B (Hallucination) 평가
| 벤치마크 | 타겟 | 의미 |
|---------|------|------|
| **TruthfulQA** | 흔한 거짓 회피 | Truthfulness를 fluency와 분리 |
| **HaluEval** | Instruction following에서의 hallucination | 폭넓은 stress test |
| **FActScore** | Long-form 텍스트의 atomic claim 정확도 | Claim-level 평가 |
| **LongBench-Cite / LongCite** | Long-context QA의 fine-grained support span | Citation 품질 mainstream화 |

---

## 12. 6가지 핵심 교훈 (반드시 암기) ⭐⭐

1. **큰 context window ≠ long-context 능력**
2. 대부분 long-context 확장 = **position remapping + careful adaptation**
3. Retrieval과 long context는 **서로 다른 failure mode 해결**
4. Hallucination 감소는 **pipeline 문제**: retrieve → rerank → compress → generate → verify → cite
5. Citation은 auditable 답변이 필요할 때의 **modeling target**
6. 최고 시스템은 **안정된 position 처리 + 명시적 evidence 제어**의 조합

---

## 13. 여전히 어려운 문제들

- **True million-scale reasoning**: 읽는 것보다 올바르게 추론하기
- **Exception preservation under compression**: 작은 예외 조항이 압축 시 사라짐
- **Reliable claim decomposition**: 의미 있는 atomic claim으로 분해
- **Joint short & long context optimization**: 확장 방법이 짧은 context 품질을 해칠 수 있음
- **Robust evaluation**: 합성 테스트만으로 부족

> 프론티어는 더 이상 "몇 토큰을 fit시키는가?"가 아니라 **"답변이 faithful하고 verifiable한가?"**

---

## 14. 주요 수학 노테이션 정리

| 기호 | 의미 |
|------|------|
| `w_t` | position t의 token ID |
| `x_t` | `w_t`의 embedding vector |
| `H` | hidden state 행렬 (position별) |
| `Q, K, V` | query, key, value 행렬 |
| `M` | causal/sparse attention mask |
| `ℓ` | 입력 길이 (토큰 수) |
| `π` | 답변 증거의 위치 |
| `M(f; ℓ, π)` | 길이 ℓ, 증거 위치 π에 대한 task metric |
| `supp(c, E)` | claim c가 evidence E에 의해 뒷받침 indicator |
| `z` | latent evidence variable (passage, chunk, graph 요약 등) |
| `g(m)` | long-context 확장을 위한 position 재매핑 함수 |
| `B` | evidence packing을 위한 prompt-budget |

---

## 📝 시험 필수 암기 체크리스트

### 🔑 반드시 쓸 수 있어야 할 공식
- [ ] Causal mask 정의
- [ ] Attention 공식 (복습)
- [ ] RAG 시퀀스-level: `p(y|x) = Σ_z p_η(z|x) p_θ(y|x,z)`
- [ ] Retriever score: `r(x,d) = u(x)ᵀ v(d)`
- [ ] PI 공식: `g_PI(m) = m · L_train/L_test`
- [ ] Generic remapped logit 공식
- [ ] Faithfulness score: `F(y, E) = (1/m) Σ supp(c_j, E)`
- [ ] Self-RAG action space: `a_t ∈ {retrieve, continue, critique}`
- [ ] CRAG pipeline: query → retrieve → evaluate → correct → generate

### 🧠 개념 이해 (서술형)
- [ ] Framework A vs Framework B 설명
- [ ] Nominal context length vs Effective long-context skill 차이
- [ ] Teacher forcing이 왜 hallucination의 씨앗인가
- [ ] PI가 왜 단순한데 효과적인가
- [ ] Long context vs Retrieval 언제 무엇을 선택?
- [ ] Self-RAG와 CRAG의 핵심 차이
- [ ] GraphRAG가 일반 RAG보다 유리한 경우
- [ ] RAG가 prompt engineering이 **아닌** 이유
- [ ] 왜 hallucination 감소가 pipeline 문제인가?
- [ ] Citation을 modeling target으로 본다는 의미

### 🎯 비교표 암기
- [ ] Position encoding 5종 (Absolute/Learned/Relative/ALiBi/RoPE)
- [ ] Framework A 지원 방법 3종 (PoSE/LongLoRA/LongAlign)
- [ ] Framework B 계열 (Self-RAG/CRAG/GraphRAG)
- [ ] Verification 3종 (RARR/CoVe/SelfCheckGPT)
- [ ] 평가 benchmark 4종 (TruthfulQA/HaluEval/FActScore/LongCite)
- [ ] Memory extension 역사: Transformer → Transformer-XL → Compressive → ALiBi → RoPE
