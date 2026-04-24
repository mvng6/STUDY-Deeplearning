# 📕 7강. Reliable Classification (신뢰할 수 있는 분류)

> **핵심 메시지**: 배포된 분류기는 단지 label generator가 아니다.
> **확률**을 생성하고, 그 확률로 **의사결정**을 한다.
> 신뢰도 연구는 전체 chain을 다룬다: **honest numbers + useful uncertainty + sensible action**.
> 0.98이라는 confidence는 그 숫자가 수치적으로 신뢰할 만할 때만 의미 있다.
> **Accuracy는 reliability의 일부일 뿐이다.**

---

## 🎯 학습 목표 (Learning Goals)

- Reliability pipeline 이해: `logits → softmax → calibration → action`
- Calibration, Proper scoring rules, ECE, Brier score, Log loss 구분
- Post-hoc calibration 방법 (Temperature scaling, Platt, Isotonic, Dirichlet)
- Uncertainty 추정 (Deep ensembles, MC dropout)
- Selective prediction (Risk-coverage curve, SelectiveNet)
- Distribution shift 하의 reliability 문제

---

## 1. Reliability는 Pipeline이다 ⭐

```
logits z(x) → softmax p(x) → calibration p̃(x) → action (predict / abstain)
```

전체 chain을 연구해야 함. Accuracy는 그중 하나일 뿐.

---

## 2. Clean Split Discipline ⭐

**반드시 기억할 데이터 분할**:
1. **Training set**: 분류기 학습
2. **Calibration (validation) set**: 모델 **동결 후** calibrator 학습
3. **Test set**: 최종 평가 (딱 한 번)
4. **Deployment data**: 미래 데이터

### ⚠️ 흔한 초보자 실수
Test set으로 calibrator를 학습하지 말 것! 정보 유출로 reliability가 과대평가됨.

---

## 3. What Calibration Means

### 3-1. 평이한 정의
**Calibrated**: 모델이 "90% 확신"이라고 말한 예시들 중 실제 90%가 맞음.

### 3-2. 3가지 수준
1. **Binary calibration**: 이진 확률
2. **Top-label calibration**: 최상위 class의 confidence
3. **Full multiclass calibration**: 전체 확률 벡터

### 3-3. Calibration vs Accuracy
- **Accuracy**: 얼마나 자주 맞추나
- **Calibration**: 확률이 얼마나 honest한가
- 이 둘은 **독립적**! 높은 accuracy에도 miscalibrated 가능

### 3-4. Calibration vs Sharpness (Resolution)
- **Calibration**: 숫자가 honest한가
- **Sharpness/Resolution**: 얼마나 자신 있게 class 분리하는가
- 둘 다 중요: 항상 0.5만 예측하는 모델은 calibrated지만 쓸모없음

---

## 4. Reliability Diagram ⭐

### 만드는 법
1. 예측을 confidence 기준으로 bin에 나눔
2. 각 bin에서 **mean confidence vs empirical accuracy** plot

### 해석
- **대각선 위**: Perfect calibration
- **대각선 아래**: **Overconfidence** (confidence가 accuracy보다 큼)
- **대각선 위 (대각선보다 높은 점)**: **Underconfidence**

---

## 5. Proper Scoring Rules ⭐⭐

### 5-1. Brier Score (이진)
```
BS = E[(C - Y)²]
```
- `C ∈ [0,1]`: 예측 확률, `Y ∈ {0,1}`: 레이블

### 5-2. Multiclass Brier Score
```
BS_multi = (1/n) Σᵢ Σₖ (pᵢ,ₖ - eᵧᵢ,ₖ)²
```
- `e_y`: one-hot label vector

### 5-3. Log Loss (NLL, Cross-Entropy)
```
NLL = -(1/n) Σᵢ log p_{i, yᵢ}
```

### 5-4. "Strictly Proper"란?
**Strictly proper scoring rule**: 진짜 분포 `q`가 minimum을 유일하게 달성
→ 평균적으로 truthful reporting이 최적

### 5-5. Log Loss의 Proper함 유도 (핵심)
```
R_log(p; q) = -Σ q_k log p_k
           = H(q) + KL(q ‖ p)
```
- `H(q)`: entropy (p와 무관)
- `KL(q‖p) ≥ 0`, 등호는 p=q일 때만
- 따라서 `R_log(p; q) ≥ R_log(q; q)` → Log loss는 strictly proper

### 💡 왜 중요?
Deep learning의 cross-entropy loss는 단순히 편의상 differentiable surrogate가 아니라,
population level에서 **honest probability estimation의 canonical loss**.

---

## 6. Expected Calibration Error (ECE) ⭐

### 6-1. 정의
예측을 M bin으로 나누고:
```
ECE = Σₘ (|Bₘ|/n) · |acc(Bₘ) - conf(Bₘ)|
```

### 6-2. ECE의 6가지 함정 ⭐⭐
| # | 함정 | 설명 |
|---|------|------|
| 1 | Binning scheme 의존 | bin 개수/경계에 따라 결과 변함 |
| 2 | Biased estimator | 유한 샘플에서 편향 |
| 3 | Top-label만 봄 | 예제당 한 숫자만 |
| 4 | Classwise/subgroup failure 숨김 | 평균이 세부 실패 숨김 |
| 5 | Proper scoring rule 아님 | 최적화 목표로 부적합 |
| 6 | **Trivial blandness를 보상** | 항상 평균 예측하면 ECE 낮아 보임 |

### 6-3. Example C 교훈 (매우 중요)
Model A (accuracy 5/8, sharp) vs Model B (accuracy 5/8, bland)
- Coarse 2-bin ECE: Model B = 0 (완벽해 보임)
- Brier, NLL: Model A가 더 좋음

**Lesson**: ECE가 **trivial blandness를 보상**할 수 있음. Proper score와 함께 봐야 함.

---

## 7. Post-hoc Calibration

### 7-1. Generic 문제 구조
Base model 동결 후 map 학습:
```
g : old outputs → new probabilities
arg min_{g∈G} Σᵢ ℓ(g(sᵢ), yᵢ)
```
- Calibration set에서 학습

### 7-2. Temperature Scaling ⭐⭐⭐ (가장 중요)
```
p^(T)_i = softmax(z_i / T)
```

**학습**: Calibration set NLL 최소화
```
L(T) = -Σᵢ log p^(T)_{i,yᵢ}
     = Σᵢ [-z_{i,yᵢ}/T + log Σₖ e^(z_{i,k}/T)]
```

**핵심 성질**:
- **Top class를 보존** (monotone transform이라서)
- Accuracy는 그대로, confidence만 보정
- 하나의 스칼라 T만 학습 → 매우 단순

### 7-3. 다른 Post-hoc 방법들
| 방법 | 특성 |
|------|------|
| **Platt Scaling** | Logistic 보정 (SVM 원조) |
| **Isotonic Regression** | 비파라메트릭 단조 보정 |
| **Beta Calibration** (Kull) | 이진용 더 원리적인 파라메트릭 |
| **Dirichlet Calibration** | Multiclass 전체 simplex 보정 |

---

## 8. Practical Reliability Toolbox ⭐

### 8-1. 3가지 주요 Uncertainty 기법 비교
| 방법 | 학습 비용 | Test 비용 | 강점 | 한계 |
|------|-----------|-----------|------|------|
| **Temperature scaling** | 작음 | 작음 | 쉬움, 좋은 baseline | Ranking 개선 안 함 |
| **Deep ensembles** | 큼 | 중-큼 | 강한 baseline, shift 하에서 우수 | 비쌈 |
| **MC dropout** | 작-중 | 중 | Retrofit 쉬움 | 보통 ensemble보다 약함 |

### 8-2. Deep Ensembles
M개 독립 모델을 독립적 초기화로 학습 후 평균:
```
p̄(x) = (1/M) Σ_m p^(m)(x)
```

**왜 잘 되는가**: Convex proper scores(Brier, log loss)에서 Jensen 부등식
```
S(p̄(x), y) ≤ (1/M) Σ_m S(p^(m)(x), y)
```
→ Ensemble은 개별 평균보다 같거나 좋다

### 8-3. MC Dropout
- Test time에도 dropout 활성화, T번 forward:
```
p̄(x) = (1/T) Σ_t p^(t)(x)
```
- Gal & Ghahramani (2016): **Bernoulli variational family**로 해석 → 근사 Bayesian inference
- ELBO 체인:
```
Bayesian predictive → variational approx with Bernoulli mask
→ dropout training objective → MC averaging at test time
```

### 8-4. Predictive Uncertainty 측도
Given `p^(1), ..., p^(T)`:
```
p̄ = (1/T) Σ p^(t)
H(p̄) = -Σ p̄_k log p̄_k              (Predictive entropy)
Cov(x) = (1/T) Σ (p^(t) - p̄)(p^(t) - p̄)ᵀ
MI(x) = H(p̄) - (1/T) Σ H(p^(t))     (Mutual information)
```
- MI는 **epistemic uncertainty**의 proxy (모델 간 disagreement)

---

## 9. Selective Prediction (기권) ⭐

### 9-1. 기본 결정 이론 유도
비용:
- 오분류 비용 = 1
- Abstain 비용 = λ (0 < λ < 1)

예상 비용:
- Answer: `1 - max_k p_k(x)`
- Abstain: `λ`

**Answer하면 유리한 조건**:
```
max_k p_k(x) ≥ 1 - λ
```
→ Reject threshold = `1 - λ`

### 9-2. Coverage & Selective Risk
- **Coverage**: 답변한 예제 비율
- **Selective risk**: 답변한 것 중 error율

### 9-3. Risk-Coverage Curve ⭐
- X축: coverage, Y축: selective risk
- Confidence로 sort하고 threshold로 몇 개를 accept할지 조절
- **AURC (Area Under Risk-Coverage)**: scalar 요약

### 9-4. Calibration vs Selective Prediction 구분
- **Calibration quality**: 숫자가 honest한가
- **Ranking quality**: 어려운 경우가 낮은 score를 받나
- **Monotone recalibration은 ranking을 바꾸지 않음**
- → Temperature scaling으로 risk-coverage curve가 거의 안 바뀔 수 있음

### 9-5. End-to-End Reject Methods

#### SelectiveNet (Geifman & El-Yaniv, 2019)
3개 head: prediction `f_θ`, selection `g_θ ∈ [0,1]`, auxiliary
- **Coverage-constrained objective**:
```
min R_sel(θ)  s.t.  φ(θ) ≥ c
```
- Penalty relaxation:
```
L(θ) = R_sel(θ) + λ max(0, c - φ(θ))² + α L_aux
```

#### Deep Gamblers
- **Reservation output**을 label space에 추가
- Probability mass를 answer classes + reservation에 할당
- Abstention이 probability simplex 자체에 녹아 있음

### 9-6. 4가지 기권 방법 계열
| 계열 | 핵심 | 강점/한계 |
|------|------|-----------|
| **Post-hoc thresholding** | Score 임계값 정함 | 쉬움; score 품질에 의존 |
| **SelectiveNet** | End-to-end 학습 | Ranking 직접 개선; retraining 필요 |
| **Deep Gamblers** | Reservation output | 우아한 end-to-end; loss 설계 민감 |
| **OOD-aware / plugin** | Correctness + ID 확률 공동 추정 | Open-world 유용; 복잡 |

---

## 10. Why Calibration Breaks After Deployment ⭐⭐

### 10-1. 3가지 Shift Story

#### Story 1: Covariate Shift
- `P(X)` 바뀜, label 의미 동일
- 예: 낮→밤 이미지, 병원 A→B 스캐너

#### Story 2: Label Shift
- Class prior `P(Y)` 바뀜
- 예: 독감 유행 시즌별 변동

#### Story 3: Out-of-Distribution (OOD)
- 학습에 없던 종류의 입력
- 모델이 known class 중 선택하도록 강제되면 **자신 있게 틀림**

### 10-2. 왜 Reliability가 Shift에 취약
Calibration은 예측-결과의 joint 분포에 대한 **conditional statement**
→ joint 분포가 바뀌면 모델 가중치가 고정되어도 calibration 깨짐

**핵심 실증 논문**:
- **Ovadia et al. (NeurIPS 2019)**: uncertainty under shift의 landmark benchmark
- **Minderer et al. (NeurIPS 2021)**: 최신 신경망에서 fragility 재확인

### 10-3. Label-Shift 보정 공식 ⭐⭐ (시험 필수)
```
P_t(Y=y | X=x) = w_y · P_s(Y=y | X=x) / Σ_j w_j · P_s(Y=j | X=x)
w_y = P_t(Y=y) / P_s(Y=y)
```
- Source calibrated classifier도 target prior 바뀌면 잘못됨

---

## 11. Murphy Decomposition (Brier 분해) ⭐⭐⭐

### 11-1. 공식 (시험 필수)
```
BS = ȳ(1-ȳ)      - Var(Q)       + E[(Q-C)²]
     └────────┘    └────────┘      └────────┘
     uncertainty   resolution      reliability
```
- `C`: 예측, `Q = E[Y|C]`, `ȳ = E[Y]`

### 11-2. 유도 핵심
Start: `BS = E_C[E_{Y|C}[(C-Y)² | C]]`

Step 1 (조건부 기댓값):
```
E[(C-Y)² | C] = Q(1-C)² + (1-Q)C² = (C-Q)² + Q(1-Q)
⇒ BS = E_C[(C-Q)²] + E_C[Q(1-Q)]
```

Step 2 (두 번째 항 정리):
```
E[Q(1-Q)] = ȳ - Var(Q) - ȳ² = ȳ(1-ȳ) - Var(Q)
```

Final:
```
BS = ȳ(1-ȳ) - Var(Q) + E[(Q-C)²]
```

### 11-3. 해석
| 성분 | 의미 |
|------|------|
| **Uncertainty** = `ȳ(1-ȳ)` | 데이터셋 본질적 난이도 (base rate만으로 결정) |
| **Resolution** = `Var(Q)` | Forecast group 간 유용한 분리 (크면 좋음 → 빼기) |
| **Reliability** = `E[(Q-C)²]` | Calibration error (0이 완벽) |

### 💡 핵심 교훈
모델이 모든 예측을 mean 쪽으로 collapse시키면 reliability가 좋아지지만 resolution이 무너짐.
→ **Calibration과 proper score를 함께 보고해야 하는 이유**

### 11-4. What Strict Propriety Does NOT Guarantee
올바른 population target만 알려줌. 유한 샘플 학습에서 calibration을 자동 보장 안 함.
Miscalibration 원인:
- 유한 샘플 오차
- Model misspecification
- Optimization dynamics
- Regularization
- Distribution shift

→ **Post-hoc calibration, uncertainty estimation, deployment auditing이 계속 필요**

---

## 12. Beyond Global Calibration ⭐

### 12-1. 왜 Global Average가 문제인가
Group A: mean conf 0.90, acc 0.70 (overconfident +0.20)
Group B: mean conf 0.50, acc 0.70 (underconfident -0.20)

**Global**: mean conf 0.70 = mean acc 0.70 → **perfect 처럼 보임**!
실제론 두 그룹 모두 miscalibrated.

### 12-2. Stronger Notions

| 개념 | 요구사항 |
|------|----------|
| **Local calibration** | Score/feature space의 local 영역에서 calibration |
| **Multicalibration** | 많은 identifiable subgroup에서 **동시에** calibration |
| **Strong calibration testing** | 잔여 miscalibration 큰 subgroup 존재 여부 감사 |

**대표 작업**: Hébert-Johnson et al. (2018), Luo et al. (2022), Feng et al. (2024)

---

## 13. 역사적 계보 (Timeline) ⭐

### 4 Branch로 정리
1. **Forecast-evaluation**: Brier → Murphy → Dawid/DeGroot-Fienberg → Gneiting-Raftery
2. **Post-hoc calibration**: Platt → Zadrozny-Elkan → Beta/Temperature/Dirichlet
3. **Uncertainty approximation**: Dropout as VI → Deep ensembles → ...
4. **Selective prediction**: Chow → El-Yaniv-Wiener → Geifman-El-Yaniv → SelectiveNet/Deep Gamblers

### 주요 연도별 이정표
| 연도 | 작업 | 의미 |
|------|------|------|
| 1950 | Brier | Brier score 도입 |
| 1957, 1970 | Chow | Reject-option / error-reject trade-off |
| 1973 | Murphy | Brier 분해 |
| 1999 | Platt | Logistic post-hoc calibration |
| 2001-02 | Zadrozny-Elkan | Histogram binning, isotonic |
| 2007 | Gneiting-Raftery | Proper scoring rules 정립 |
| 2015 | Naeini et al. | Bayesian binning |
| **2016** | **Gal-Ghahramani** | **Dropout as VI** |
| **2017** | **Guo et al.** | **Modern NN이 miscalibrated; Temperature scaling** |
| **2017** | **Lakshminarayanan** | **Deep ensembles** |
| 2017 | Geifman-El-Yaniv | Practical selective classification |
| **2019** | **Ovadia et al.** | **Shift 하에서 uncertainty 평가** |
| 2019 | SelectiveNet | End-to-end reject option |
| 2021 | Minderer et al. | Modern NN calibration fragility |
| 2022 | Hébert-Johnson | Multicalibration |
| 2023-25 | 최근 wave | Many-class, shift-aware, truthfulness 등 |

---

## 14. Practical Workflow ⭐ (시험 필수)

### 14-1. 기본 7단계
1. Cross-entropy로 분류기 학습
2. Accuracy + Proper score (NLL/Brier) 보고
3. Reliability diagram + ECE (binning rule 명시)
4. Clean calibration split에서 **Temperature scaling** 피팅
5. Uncertainty 품질 중요하면 deep ensemble 비교
6. Abstention 중요하면 전체 risk-coverage curve + 몇 operating point
7. 최소 하나의 plausible deployment shift로 stress test

### 14-2. 상황별 조언
| 상황 | 권장 |
|------|------|
| 사용자가 top label + confidence만 봄 | Top-label calibration 진단; proper score 병행 |
| 전체 확률 벡터 사용 | Classwise / vector-aware diagnostics; Dirichlet calibration |
| Selective prediction 배포 | Ranking 품질 별도 평가; risk-coverage curve |
| Deployment shift 중요 | Label-shift / OOD-aware 방법 |

---

## 15. 최근 연구 동향 (4가지 질문)

| 질문 | 주요 움직임 | 대표 연구 |
|------|-------------|-----------|
| **Calibration metric 믿을 만한가?** | Binning 대체, stronger testing, truthfulness 분석 | Marx et al., Hu et al., Haghtalab et al. |
| **어떤 calibration notion이 task에 맞나?** | Top-confidence, full-vector, many-class, subgroup 분리 | Le Coz et al., Chidambaram-Ge |
| **Shift 하에서 calibration을 어떻게 바꾸나?** | Label-shift/target-shift 추정을 recalibration에 삽입 | **LaSCal** (Popordanoska) |
| **좋은 selective classifier는?** | Ranking 직접 평가, safe vs risky ordering 개선 | Pugnana-Ruggieri, Traub et al., Mao et al. |

---

## 16. 핵심 결론 (Final Conclusion)

> Reliable classification = **chain**:
> **probability estimation → scoring → post-hoc repair → uncertainty → decision**

- **Proper scoring rules**: 이론적 뿌리 (honest probability estimation이란 무엇인가)
- **ECE**: descriptive summary일 뿐, proper objective 아님
- **Temperature scaling**: practical workhorse
- **Deep ensembles / MC dropout**: single softmax 이상의 uncertainty 신호
- **Selective prediction**: signal로 action 결정
- **Deployment shift**: reliability는 항상 distribution-dependent

---

## 📝 시험 필수 암기 체크리스트

### 🔑 반드시 쓸 수 있어야 할 공식
- [ ] Brier score: `BS = (1/n) Σ Σ (p_{i,k} - e_{y_i,k})²`
- [ ] Log loss: `NLL = -(1/n) Σ log p_{i,y_i}`
- [ ] **Murphy 분해**: `BS = ȳ(1-ȳ) - Var(Q) + E[(Q-C)²]`
- [ ] ECE: `Σ (|B_m|/n) |acc(B_m) - conf(B_m)|`
- [ ] **Temperature scaling**: `p^(T) = softmax(z/T)`
- [ ] Ensemble: `p̄ = (1/M) Σ p^(m)`
- [ ] MC dropout: `p̄ = (1/T) Σ p^(t)`
- [ ] Mutual information: `MI = H(p̄) - (1/T) Σ H(p^(t))`
- [ ] Reject threshold: `max_k p_k ≥ 1 - λ`
- [ ] SelectiveNet objective (coverage-constrained)
- [ ] **Label-shift 보정**: `w_y = P_t(Y=y)/P_s(Y=y)`
- [ ] Log loss strict properness 유도: `R_log = H(q) + KL(q‖p)`

### 🧠 개념 이해 (서술형)
- [ ] Reliability pipeline 4단계 설명
- [ ] Clean split discipline 중요성
- [ ] Calibration ≠ Accuracy ≠ Sharpness
- [ ] Reliability diagram 해석 (over/underconfidence)
- [ ] ECE의 6가지 함정
- [ ] Example C의 교훈 (ECE가 blandness를 보상)
- [ ] Temperature scaling이 top class 보존하는 이유
- [ ] Strictly proper scoring rule이란?
- [ ] Deep ensemble이 convex proper score에서 잘 되는 이유 (Jensen)
- [ ] MC dropout의 Bayesian 해석
- [ ] Calibration vs Selective prediction 차이
- [ ] Monotone recalibration이 ranking을 못 바꾸는 이유
- [ ] 3가지 shift 시나리오 설명
- [ ] Murphy 분해의 3성분 의미
- [ ] Global calibration이 subgroup failure를 숨기는 예

### 🎯 비교표 암기
- [ ] Post-hoc calibration 방법 (Platt/Temperature/Isotonic/Beta/Dirichlet)
- [ ] Uncertainty 방법 3종 비교 (Temp/Ensemble/Dropout)
- [ ] Selective prediction 4종 계열 (Thresholding/SelectiveNet/Deep Gamblers/OOD-aware)
- [ ] Shift 3종 (Covariate/Label/OOD)
- [ ] Beyond global: Local / Multicalibration / Strong testing
- [ ] Practical workflow 7단계
