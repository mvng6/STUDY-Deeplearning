# 📘 4강. Inference Optimization (추론 최적화)

> **핵심 메시지**: 추론 최적화는 모델 설계 문제이자 시스템 문제다.
> 최적화가 유의미하려면 **목표 하드웨어와 런타임 스택의 실제 병목**을 개선해야 한다.
> 파라미터 수 감소나 FLOPs 감소만으로는 wall-clock latency 감소를 보장하지 않는다.

---

## 🎯 학습 목표 (Learning Goals)

- Transformer 추론의 구조 이해: **prefill, decode, KV cache**
- 4가지 최적화 레버 구분: **Quantization, Pruning/Sparsity, Distillation, Low-rank Structure**
- FLOPs/파라미터가 아닌 **실제 배포 병목** 기준으로 판단
- 최적화 주장 읽을 때 올바른 질문 던지기 (어떤 병목이 어떤 하드웨어에서 개선되었나?)

---

## 1. 기본 개념: Inference가 계산하는 것

### 1-1. 신경망의 기본 연산
```
h = Wx + b,  u = σ(h)
```
- `x`: 입력 벡터, `W`: 가중치 행렬, `b`: bias
- `σ`: GELU, ReLU 등 비선형 함수
- **추론 대부분 = 선형대수(행렬곱) + 원소별 비선형 연산**

### 1-2. 모델 측 효율화의 3가지 대상
1. `W`의 **entries** (값 자체) → Pruning
2. `W`를 저장하는 **precision** (정밀도) → Quantization
3. 연산이 **하드웨어에서 실행되는 방식** → Runtime 최적화 (FlashAttention 등)

### 1-3. Training vs Inference
| 구분 | 내용 |
|------|------|
| **Training** | 파라미터 업데이트 (gradient + backprop). 비용 큼. |
| **Inference** | 고정된 모델로 새 입력 평가. 학습 없음. |
| 중요성 | 배포된 서비스는 추론을 수백만 번 반복 → 작은 절약도 큰 영향 |

---

## 2. Transformer Inference의 2단계 구조 ⭐

| 단계 | 특성 | 병목 |
|------|------|------|
| **Prefill** | 입력 프롬프트를 한 번에 처리 | Compute-bound |
| **Decode** | 토큰 하나씩 생성 (자기회귀) | **Memory-bound (KV cache 의존)** |

### KV Cache
- Decode 단계에서 과거 Key, Value를 저장하는 캐시
- **시퀀스 길이에 따라 커짐** → 긴 문맥에서 병목의 주범
- 디코딩 시 cache 크기: `cache ∝ t × N_KV × (d_k + d_v)`

---

## 3. 5가지 배포 지표 (반드시 구분)

1. **Latency** — 지연시간 (특히 p50, p95, p99 tail latency 중요)
2. **Throughput** — 처리량 (req/s, tok/s)
3. **Memory** — 메모리 풋프린트
4. **Energy** — 에너지 소비
5. **Accuracy** — 정확도

### ⚠️ 핵심 원리: FLOPs 감소 ≠ Latency 감소
- **Decode가 bandwidth-bound**이면 weight bytes 감소가 FLOPs 감소보다 중요
- **Unstructured sparsity**는 dense kernel이면 latency 개선 안 됨
- **Dense distilled student**가 heavily compressed large model을 이길 수 있음

### Goodput (최신 온라인 지표)
```
goodput = |{r : TTFT_r ≤ τ₁, TPOT_r ≤ τ₂}| / T_wall
```
- TTFT (Time To First Token), TPOT (Time Per Output Token)
- SLO(Service Level Objective) 만족 요청만 카운트

---

## 4. 4가지 최적화 레버 (Levers) ⭐⭐⭐

| 레버 | 무엇을 바꾸나 | 주요 이득 | 배포 리스크 |
|------|---------------|-----------|------------|
| **Quantization** | weights/activations/KV cache의 숫자 정밀도 | 모델 크기, 메모리 대역폭 | Outlier, clipping, 캘리브레이션 오류 |
| **Pruning/Sparsity** | 0이 아닌 파라미터 수/배치 | 하드웨어 친화적일 때만 속도↑ | FLOP↓ 해도 end-to-end 속도 그대로 |
| **Distillation** | 학습 목표 + 학생 모델 구조 | 최고의 품질-지연 프론티어 | Teacher bias, train-inference mismatch |
| **Low-rank structure** | 행렬 인수분해 / LoRA | 저렴한 적응, 병합 가능 | Rank 너무 작으면 병목 |

---

## 5. Quantization (양자화) — 심화

### 5-1. Affine Quantization 기본 공식 ⭐
```
q = round(x / scale) + zero_point
```
- **Scale (s)**와 **Zero-point (z)** 결정이 핵심
- Integer-only inference를 가능하게 함 (production 배포에 필수)

### 5-2. PTQ vs QAT 비교

| 항목 | PTQ (Post-Training Quantization) | QAT (Quantization-Aware Training) |
|------|----------------------------------|-----------------------------------|
| 시점 | 학습 후 | 학습/파인튜닝 중 |
| 최적화 신호 | Calibration statistics, local reconstruction | End-to-end task loss (fake quantization) |
| 비용 | 낮음 | 높음 |
| 강점 | 빠른 배포 (INT8, INT4 weight-only) | 공격적 bit-width에서 더 나은 복구 |
| 실패 모드 | Outlier, 캘리브레이션 오류, kernel mismatch | 최적화 불안정, 과적합 |

### 5-3. 2차 구조 (Second-order)가 계속 등장하는 이유
학습된 모델 근처에서 작은 perturbation δ의 효과:
```
ΔL ≈ (1/2) δᵀ H δ     (gradient ≈ 0 at optimum)
```
- **H (Hessian)**: 국소 곡률/민감도 행렬
- 큰 곡률 방향으로의 변화가 더 비쌈
- GPTQ, OBD/OBS, HAWQ 같은 방법들에 공통으로 등장

### 5-4. 주요 Quantization 방법 계보
| 방법 | 연도 | 핵심 기여 |
|------|------|-----------|
| **Jacob et al.** | 2018 | Affine scale/zero-point, integer-only inference |
| **PACT** | 2018 | Clipping range를 학습 |
| **LSQ** | 2019-20 | Step size를 surrogate gradient로 학습 |
| **HAWQ** | 2019-20 | Hessian 정보로 mixed precision |
| **AdaRound** | 2020 | Rounding을 최적화 문제로 |
| **BRECQ** | 2021 | Block-wise reconstruction |
| **GPTQ, LLM.int8()** | 2022 | LLM용 scalable rowwise PTQ + outlier handling |

---

## 6. Pruning & Sparsity

### 6-1. Pruning을 제약 최적화로 공식화 ⭐
```
min_{m, Wᶜ}  L(m ⊙ Wᶜ)     s.t.  ‖m‖₀ ≤ k
```
- `m ∈ {0,1}^dim(W)`: binary mask (어떤 가중치가 살아남나)
- `Wᶜ`: 보정된(compensated) 가중치 (pruning 후 남은 값이 이동 가능)
- `‖m‖₀`: 0-norm (0이 아닌 항목 수)

### 6-2. Pruning의 2가지 설계 축
1. **Pattern**: element, channel, head, block 등
2. **Recovery**: none, local compensation, brief fine-tuning, full retraining

### 6-3. OBS Compensation 유도 (Optimal Brain Surgeon) ⭐
좌표 `i`를 pruning할 때:
```
min_δ  (1/2) δᵀH δ    s.t.  eᵢᵀδ = -θᵢ
```
Lagrangian으로 풀면:
```
δ* = -[θᵢ / [H⁻¹]ᵢᵢ] H⁻¹_{:,i}
ΔL*ᵢ ≈ (1/2) θᵢ² / [H⁻¹]ᵢᵢ
```
- 한 좌표를 prune하고, 국소 2차 모델이 허용하는 가장 저렴한 방향으로 보상

### 6-4. Unstructured vs Structured vs Semi-structured
| 유형 | 특성 | 속도 개선 |
|------|------|-----------|
| **Unstructured** | 개별 weight 단위 | Dense kernel 쓰면 없음 |
| **Structured** | Channel/head 단위 | 실제 차원 축소 → 속도↑ |
| **Semi-structured** | 예: 2:4 sparsity (NVIDIA) | 전용 hardware 지원 시 속도↑ |

---

## 7. Knowledge Distillation (증류)

### 7-1. 표준 KD 목적함수 ⭐⭐ (시험 필수 공식)
```
L_KD = (1-α) · CE(y, pˢ) + α τ² · KL(p^T_τ ‖ pˢ_τ)
```
- `p^S = softmax(zˢ)`: student 확률
- `p^T_τ = softmax(z^T / τ)`: temperature 적용 teacher
- `α`: hard label vs teacher imitation trade-off
- `τ`: temperature (크면 분포가 부드러워져 relative class preference 보기 쉬움)
- `τ²` 항의 의미: gradient 크기 보정 (Part II 유도)

### 7-2. Soft Target이 유용한 이유
Teacher 분포가 class 2가 class 3보다 그럴듯하다는 정보를 줌 → one-hot보다 훨씬 풍부한 supervision

### 7-3. Distillation 변형들
| 종류 | 무엇을 match | 대표 방법 |
|------|--------------|-----------|
| **Classical KD** | Teacher output distribution | Hinton KD, DistilBERT |
| **Feature KD** | 중간 hidden state / attention | FitNets, TinyBERT, MiniLM |
| **Sequence-level KD** | Teacher-generated 전체 시퀀스 | Sequence KD |
| **On-policy generative KD** | Student-generated trajectory + reverse KL 등 | **MiniLLM, GKD** |

### 7-4. Generative LLM을 위한 핵심 질문
1. **무엇을** match할 것인가? (target)
2. **어떤 divergence**를 쓸 것인가? (KL, reverse KL)
3. **누구의 prefix/trajectory** 위에서 match 시행할 것인가?

---

## 8. Low-rank Structure & LoRA

### 8-1. LoRA 기본
```
W_final = W + AB    (A ∈ R^{d×r}, B ∈ R^{r×k}, r ≪ d)
```
- 원 가중치 `W`는 **동결**, 저차원 `A, B`만 학습
- Inference 시 `W + AB`로 병합 가능 → extra latency 없음

### 8-2. QLoRA
- Base model을 **4-bit NF4**(NormalFloat 4)로 저장
- Compute 시 dequantize, LoRA adapter만 학습
- 주로 **training-memory optimization**
- 제한된 하드웨어에서 큰 모델 adaptation 가능

### 8-3. LoftQ — Quantization과 LoRA 공동 최적화
```
min_{Ŵ∈Q, A, B}  ‖W - Ŵ - AB‖²_F
```
- `Ŵ`: quantized backbone
- `AB`: low-rank correction
- Quantized base와 adapter를 **처음부터 함께** 선택

### 8-4. LoRA 이후 변형들
| 변형 | 핵심 |
|------|------|
| **AdaLoRA** | 고정된 rank 예산을 layer별로 재할당 |
| **DoRA** | Weight update를 magnitude와 direction으로 분해 |

---

## 9. Runtime 최적화: FlashAttention ⭐

### 9-1. 문제: 표준 Attention의 병목
- 전체 score matrix `S = QKᵀ/√d_k` 생성
- Softmax 확률 저장
- 큰 텐서를 HBM(high-bandwidth memory)과 on-chip memory 사이 이동
- **주 비용 = memory traffic**, FLOPs 아님

### 9-2. 핵심 아이디어: Online Softmax in Tiles
각 query row에 대해 유지:
- **running maximum** `m`
- **running denominator** `ℓ`
- **running weighted-value accumulator** `o`

### 9-3. Block 단위 업데이트 공식
새 block 도착 시:
```
m_new = max(m_old, m_blk)
ℓ_new = e^(m_old - m_new) · ℓ_old + Σ_blk e^(s_j - m_new)
o_new = e^(m_old - m_new) · o_old + Σ_blk e^(s_j - m_new) · v_j
```

### 9-4. ⚠️ 중요한 구분
**FlashAttention은 attention을 근사하지 않음!**
같은 함수를 **더 효율적으로** 계산함 (exact).

---

## 10. Benchmarking (올바른 벤치마킹)

### 10-1. 신뢰할 수 있는 벤치마크 요소
- 하드웨어, 드라이버/런타임, 컴파일러, 서빙 프레임워크
- 정확한 모델 버전, 정밀도, 실제 실행된 kernel 경로
- Prompt/output 길이 분포, batch size, concurrency
- Warmup 정책, 동기화된 타이밍, p50/p95/p99
- 품질 체크

### 10-2. 자주 하는 실수와 해결
| 실수 | 해결 |
|------|------|
| Workload/decoding 설정 다름 | Workload 고정 후 명시 |
| Sparse/low-bit 주장, kernel 미확인 | 실제 kernel 경로 확인 |
| 평균 latency만 보고 | Tail latency + goodput 보고 |
| Offline throughput을 interactive 증거로 사용 | 각자 operating point에서 보고 |
| Perplexity만으로 품질 판단 | Downstream task quality 체크 |

### 10-3. LLM 특유의 주의사항
- **Prefill과 decode 분리 보고** (bottleneck이 다를 수 있음)
- Prompt 길이와 생성 길이 명시

---

## 11. Bottleneck 원칙과 의사결정 규칙 ⭐

### 11-1. Bottleneck Principle
> 최적화는 **target stack에서 실제 bottleneck 항**을 개선할 때만 배포할 가치가 있다.

### 11-2. 3가지 배포 테스트
후보 최적화는 다음을 모두 만족해야:
1. 관련 평가 분포에서 task 품질 유지
2. Target workload의 bottleneck 항 개선
3. 기준선과 **동일한 스택/workload/품질 목표** 아래 측정

---

## 📝 시험 필수 암기 체크리스트

### 🔑 반드시 쓸 수 있어야 할 공식
- [ ] Quantization affine: `q = round(x/s) + z`
- [ ] KD loss: `L = (1-α)CE + ατ²KL(p^T_τ ‖ pˢ_τ)`
- [ ] Pruning 제약: `min L(m⊙Wᶜ) s.t. ‖m‖₀ ≤ k`
- [ ] OBS 비용: `ΔL* = θᵢ²/(2[H⁻¹]ᵢᵢ)`
- [ ] LoftQ: `min ‖W - Ŵ - AB‖²_F`
- [ ] FlashAttention online softmax 업데이트 공식
- [ ] Goodput 공식

### 🧠 개념 이해 (서술형 자주 출제)
- [ ] FLOPs 감소가 latency 감소를 보장하지 않는 이유
- [ ] Prefill vs Decode 병목 차이
- [ ] PTQ vs QAT 언제 무엇을 선택?
- [ ] FlashAttention이 **근사가 아닌** 이유
- [ ] Unstructured sparsity가 자주 속도 개선 없이 끝나는 이유
- [ ] Bottleneck principle 설명
- [ ] KV cache가 decode 병목인 이유
- [ ] Soft target(KD)가 hard label보다 풍부한 이유

### 🎯 비교표 암기
- [ ] 4가지 최적화 레버 비교
- [ ] PTQ vs QAT 표
- [ ] Pruning 3가지 (Unstructured/Structured/Semi-structured)
- [ ] KD 변형 4가지 (Classical/Feature/Sequence/On-policy)
