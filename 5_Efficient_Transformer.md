# 📗 5강. Efficient Transformer (효율적 트랜스포머)

> **핵심 메시지**: 모든 sequence model은 하나의 질문에 답한다.
> **"모델은 이미 본 prefix x<t를 어떻게 기억할 것인가?"**
> 답은 몇 가지 되지 않는다: 모든 과거 토큰 보존(exact retrieval), 선택/압축 요약, recurrent state 압축, 학습된 convolutional filter, 또는 이들의 hybrid.

---

## 🎯 학습 목표 (Learning Goals)

- Transformer 기본 연산 이해: **attention, KV cache**
- 효율적 Transformer의 5가지 수정 방향 구분
- Efficient attention, recurrent model, state-space model, hybrid 비교
- 점근 복잡도 ≠ 실제 하드웨어 효율 (커널 구현이 중요)

---

## 1. The Whole Field in One Picture

### 1-1. 4가지 Memory 스타일
모든 모델을 "과거를 어떻게 표현/업데이트하는가"로 분류:

| 스타일 | 특성 |
|--------|------|
| **Exact retrieval** | 모든 과거 토큰 저장 (Transformer) |
| **Selective/compressed** | 일부 선택 또는 압축 요약 (Sparse, Low-rank) |
| **Recurrent state** | 고정 크기 상태로 압축 (RNN, SSM) |
| **Hybrid** | 여러 방식 결합 |

### 1-2. 핵심 질문 2개
1. **무엇이 저장되는가?** (모든 과거 / sparse subset / low-rank 요약 / recurrent state / memory bank)
2. **어떻게 읽히는가?** (similarity search / fixed filter / state readout / hybrid)

이 두 질문에 답하면 대부분의 구조는 이미 해명됨.

---

## 2. Core Ideas: Standard Transformer

### 2-1. 표준 Attention 공식 ⭐⭐ (시험 필수)
```
Attn(Q, K, V) = softmax(QKᵀ/√d_k + M) V
```
- `Q = H W_Q`, `K = H W_K`, `V = H W_V`
- `M`: causal 또는 sparse mask
- **복잡도: O(L² · d_k)** → 긴 시퀀스에서 bottleneck

### 2-2. Multi-Head Attention (MHA)
```
head^(m) = Attn(X W_Q^(m), X W_K^(m), X W_V^(m))
MHA(X)   = Concat(head^(1), ..., head^(N_h)) W_O
```
- 다른 head는 다른 관계에 특화: local syntax, long-range reference, instruction following 등

### 2-3. 하나의 Decoder Block
```
X̃  = X + MHA(LN(X))
X' = X̃ + MLP(LN(X̃))
```
- Attention: 위치 **간** 정보 혼합
- MLP: 각 위치 **내** 정보 변환
- Residual: earlier info 유지

### 2-4. Causal Mask (자기회귀 핵심)
```
M_ij = 0 if j ≤ i else -∞
```
- Teacher forcing 중에도 미래 토큰을 보지 못하게
- 학습은 병렬이어도 생성은 순차적인 이유

### 2-5. Exact Attention의 근본 비용
- `QKᵀ` 형성 → O(L²) 저장공간과 compute
- 긴 context에서 첫 주요 bottleneck

---

## 3. 왜 Exact Attention은 비쌀까?

### 3-1. Training 측면
- 전체 score matrix를 한 번에 만들어 backprop
- O(L² d_k) 메모리 사용

### 3-2. Inference 측면 (더 중요)
- **KV cache**가 시퀀스 길이에 비례해 커짐
- Decoding 시 cache 크기: `cache ∝ t × N_KV × (d_k + d_v)`

### 3-3. Transformer vs Finite-state Recurrence
| 구조 | 상태 크기 |
|------|-----------|
| Transformer decoding | **KV cache가 시퀀스 길이로 증가** |
| Recurrent (RNN, SSM) | **시간에 관계없이 고정** |

이것이 현대 long-context 설계의 핵심 구분.

---

## 4. 위치 정보 (Positional Information)

### 4-1. 왜 위치가 중요한가
- 위치 없으면 "dog bites man"과 "man bites dog" 구분 불가
- **Context window**: 모델이 한 번에 처리할 수 있는 토큰 수 (≠ long-context 능력)

### 4-2. 2가지 설계 선택
1. **Additive position encoding**: `h_i = x_i + p_i`
2. **Relative-position scoring**: attention score가 거리/phase에 의존 (ALiBi, RoPE)

### ⚠️ 중요한 구분
- **Nominal context length**: 엔지니어링 속성 (몇 토큰 받나)
- **Effective long-context skill**: 행동적 속성 (실제로 잘 사용하나)

---

## 5. Efficient Transformer 5가지 수정 방향 ⭐⭐⭐

| 접근 | 무엇을 바꾸나 | 대표 방법 |
|------|---------------|-----------|
| **1. Sparse attention** | attention 패턴 (위치 비교를 줄임) | Sparse Transformer, Longformer, BigBird, Reformer |
| **2. Low-rank attention** | 시퀀스 차원 압축 | Linformer, Nyströmformer |
| **3. Kernelized (Linear) attention** | softmax를 kernel inner product로 근사 | Performer, RWKV |
| **4. Implementation 변경** | kernel 스케줄링 (정확도 유지) | **FlashAttention** |
| **5. Cache 축소** | 저장되는 state | **MQA, GQA, MLA** |

---

## 6. Sparse Attention (희소 주의)

### 핵심: 각 토큰이 **일부 과거만** 보게 함
- 패턴 예시: local window, strided, global+local, random
- 복잡도를 O(L²)에서 O(L·k)로 낮춤

### 대표 방법
- **Longformer**: local window + 일부 global token
- **BigBird**: local + random + global
- **Reformer**: LSH로 비슷한 key끼리 묶어 sparse 만듦

---

## 7. Low-rank Attention

### Linformer 핵심 아이디어
- Attention matrix가 low-rank라는 가정
- Key/Value의 **시퀀스 차원**을 projection matrix E, F로 축소
- 예: 4096 토큰 → 256 basis vector로 요약

### Nyströmformer
- Landmark points로 full kernel matrix 근사

### ⚠️ 한계
- Attention pattern이 highly localized/irregular면 실패

---

## 8. Linear (Kernelized) Attention ⭐

### 8-1. Kernel Factorization
Softmax kernel을 feature map으로 근사:
```
exp(qᵀk / √d_k) ≈ φ(q)ᵀ φ(k)
```

### 8-2. 핵심 유도 (Causal)
```
y_t ≈ [Σ_{j≤t} φ(q_t)ᵀ φ(k_j) v_j] / [Σ_{j≤t} φ(q_t)ᵀ φ(k_j)]
    = [φ(q_t)ᵀ (Σ_{j≤t} φ(k_j) v_jᵀ)] / [φ(q_t)ᵀ (Σ_{j≤t} φ(k_j))]
```

### 8-3. Recurrent State로 재표현 ⭐
```
S_t = S_{t-1} + φ(k_t) v_tᵀ    (matrix-valued state)
z_t = z_{t-1} + φ(k_t)
y_t ≈ φ(q_t)ᵀ S_t / (φ(q_t)ᵀ z_t)
```

### 💡 핵심 통찰
> "Transformers are RNNs in the linear-attention setting"
> 전체 prefix가 **하나의 행렬 state (S_t, z_t)**로 요약됨.

### 8-4. Performer / FAVOR+
- Softmax kernel을 **positive random features**로 근사
- 유용한 항등식: `e^(qᵀk) = e^(‖q‖²/2) · e^(‖k‖²/2) · e^(-‖q-k‖²/2)`
- 마지막 항은 Gaussian-like → random feature 근사 가능

---

## 9. FlashAttention: Exact attention 가속 ⭐⭐

### 9-1. 핵심 차이
**FlashAttention은 근사 없이** attention을 정확하게 계산하되,
**kernel schedule**(I/O 관리)을 바꿈.

### 9-2. 작동 방식
각 query row에 대해 유지:
- Running maximum `m`
- Running denominator `ℓ`
- Running weighted-value accumulator `o`

Keys/Values를 block 단위로 처리, `(m, ℓ, o)`를 online 업데이트
→ **전체 attention matrix를 slow memory에 저장하지 않음**

### 9-3. Online Softmax 업데이트 (block 단위)
```
m_new = max(m_old, m_blk)
ℓ_new = e^(m_old - m_new) ℓ_old + Σ_blk e^(s_j - m_new)
o_new = e^(m_old - m_new) o_old + Σ_blk e^(s_j - m_new) v_j
최종: y = o_final / ℓ_final
```

### 9-4. 왜 중요한가
- Asymptotic complexity 개선이 속도의 유일한 길은 아님
- **더 나은 exact kernel**이 baseline을 극적으로 강화
- FlashAttention-2, FlashAttention-3로 계속 개선

---

## 10. Cache 축소: MQA, GQA, MLA ⭐⭐

### 10-1. 왜 중요한가
- Decoding 시 cache 크기 ∝ `t × N_KV × (d_k + d_v)`
- 많은 query head가 같은 과거를 공유/재구성하면 cache↓

### 10-2. 3가지 비교

| 방법 | 무엇을 cache | 주요 이점 | 주요 단점 |
|------|--------------|-----------|-----------|
| **Full MHA** | query head마다 하나의 KV set | 최대 head 유연성 | 가장 큰 KV cache |
| **MQA** (Multi-Query) | 모든 head가 **하나의** KV set 공유 | Cache 최소, 빠른 decoding | 너무 restrictive |
| **GQA** (Grouped-Query) | head 그룹마다 하나의 KV set | 품질-cache **절충** | Head specialization 일부 손실 |
| **MLA** (Multi-head Latent) | **latent cache**를 head별 KV로 재구성 | 매우 큰 cache 절약 | 구조적 복잡도↑ |

### 10-3. 구체적 예
32 query head일 때:
- MHA: 32개 KV set 저장
- GQA (4개씩 grouping): **8개** KV set만 저장

### 10-4. 왜 GQA/MLA가 historically important
- Training 복잡도뿐 아니라 **inference cache 비용**이 중심 화두가 됨
- 현대 decoder-only LLM의 실질적 표준

---

## 11. State-Space Models (SSM) 기본

### 11-1. 연속시간 시스템
```
ḣ(t) = A h(t) + B u(t)
y(t) = C h(t) + D u(t)
```

### 11-2. 이산화 (Zero-order hold)
```
h_t = Ā h_{t-1} + B̄ u_t
Ā = e^(ΔA),  B̄ = ∫₀^Δ e^((Δ-τ)A) B dτ
```

### 11-3. HiPPO / S4
- 과거를 원리적 basis로 요약
- Long-range dependency를 안정적으로 다루는 수학적 기반

### 11-4. 주요 SSM 계열
| 모델 | 핵심 |
|------|------|
| **S4** | Structured state space, 긴 시퀀스에 경쟁력 |
| **H3, Hyena** | Convolutional state-space와 attention의 hybrid |
| **RWKV** | Linear attention과 recurrence를 섞음 |
| **Mamba** | Token-dependent selection (input에 따라 상태 변화) |
| **Mamba-2** | Structured state space duality |

---

## 12. 공통 프레임워크: Finite-state models

### 12-1. 핵심 통찰
많은 현대 모델이 다음 두 가지만으로 구분됨:
1. **어떤 finite state를 유지하는가?**
2. **state를 어떻게 업데이트하는가?**

### 12-2. 비교표
| 방법 | Core update 아이디어 | 주요 추가 요소 |
|------|---------------------|----------------|
| Linear attention | Additive state | Feature-map kernelization |
| **RetNet** | Additive state + decay | 고정 forgetting |
| **GLA** (Gated Linear Attention) | Additive state + gate | Data-dependent forgetting |
| **DeltaNet** | Additive state + correction | Delta-rule residual write |
| **Gated DeltaNet** | Decay/gate + correction | Two memory controls |
| **Mamba** | State-space recurrence | Token-dependent selection |
| **xLSTM** | Gated recurrent memory | Matrix-valued cell state |

---

## 13. 2023~2026 프론티어의 4가지 변화 ⭐

### 13-1. Exact attention은 여전히 강한 baseline
- FlashAttention-2, -3가 exact softmax를 극적으로 하드웨어 효율적으로 만듦

### 13-2. Inference cache 효율이 중심 화두
- GQA와 MLA가 long-context decoding을 실용화

### 13-3. Finite-state 모델의 재등장
- RetNet, Mamba, DeltaNet, xLSTM이 고정 상태 memory로 경쟁력 확보

### 13-4. Hybrid 설계가 실용적 타협
- Exact retrieval과 compressed memory는 다른 문제 해결
- 많은 시스템이 둘을 결합

### 💡 프론티어가 결정하지 않은 것
- 만능 Transformer 대체재는 **없음**
- 점근적 linear가 항상 실전에서 승리하는 것도 **아님**
- Exact retrieval이 obsolete가 된 것도 **아님**

---

## 14. "Efficient"의 다차원 의미 ⭐

단일 숫자로 효율성을 판단하지 말 것. 구분해야 할 5가지:

1. **Sequence-length complexity during training** (점근적)
2. **실제 하드웨어 효율** (GPU/TPU에서)
3. **자기회귀 추론 KV cache / recurrent state 크기**
4. **Long context에서의 latency / throughput**
5. **모델 품질** (표준 LM, long-context retrieval, extrapolation)

> **예**: 점근 복잡도는 좋은데 kernel이 hardware 친화적이지 않아 실전에서 지는 경우 多
> → 이것이 FlashAttention이 여전히 중요한 이유

---

## 📝 시험 필수 암기 체크리스트

### 🔑 반드시 쓸 수 있어야 할 공식
- [ ] Attention: `Attn(Q,K,V) = softmax(QKᵀ/√d_k + M)V`
- [ ] Multi-head: `MHA = Concat(head_1,...,head_h) W_O`
- [ ] Decoder block: `X̃ = X + MHA(LN(X))`, `X' = X̃ + MLP(LN(X̃))`
- [ ] Causal mask 정의
- [ ] Linear attention state: `S_t = S_{t-1} + φ(k_t)v_tᵀ`
- [ ] Readout: `y_t ≈ φ(q_t)ᵀ S_t / (φ(q_t)ᵀ z_t)`
- [ ] Kernel factorization: `exp(qᵀk/√d_k) ≈ φ(q)ᵀφ(k)`
- [ ] KV cache 크기 공식: `cache ∝ t × N_KV × (d_k + d_v)`
- [ ] FlashAttention online softmax 업데이트

### 🧠 개념 이해 (서술형)
- [ ] Exact attention이 왜 O(L²)인가?
- [ ] Transformer 디코딩이 왜 순차적인가? (causal mask + teacher forcing)
- [ ] FlashAttention이 근사 아니라 exact인 이유
- [ ] MQA vs GQA vs MLA 차이와 trade-off
- [ ] Linear attention이 왜 "Transformer = RNN" 해석을 가능하게 하나?
- [ ] Context window vs Long-context skill 차이
- [ ] 왜 asymptotic complexity만으로는 효율성 판단 불가?
- [ ] SSM이 long-range 처리에 유리한 이유
- [ ] 5가지 efficient transformer 수정 방향

### 🎯 비교표 암기
- [ ] 5가지 efficient attention 접근 (sparse/low-rank/linear/FlashAttention/cache 축소)
- [ ] MHA/MQA/GQA/MLA 비교
- [ ] Finite-state 모델 7종 (Linear attn/RetNet/GLA/DeltaNet/Gated DeltaNet/Mamba/xLSTM)
- [ ] "Efficient"의 5가지 다차원 의미
