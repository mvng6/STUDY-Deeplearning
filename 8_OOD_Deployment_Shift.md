# 📓 8강. Reliability Under Deployment Shift (배포 환경에서의 신뢰도)

> **핵심 메시지**: IID 환경에서 잘 작동하는 모델도 배포(deployment) 환경에서는 깨진다.
> ML 시스템은 **세 가지 질문**에 답해야 한다:
> 1. 환경이 바뀌면 어떻게 되나? (Distribution Shift)
> 2. 모르는 입력을 알아볼 수 있나? (OOD Detection)
> 3. 의도적 공격에 버틸 수 있나? (Adversarial Robustness)
> 그리고 이 모든 것에 대해 **통계적 보장**을 줄 수 있나? (Conformal Prediction)

---

## 🎯 학습 목표 (Learning Goals)

- Deployment shift의 3가지 유형 (covariate, label, concept) 구분
- OOD detection의 주요 score 함수 이해 및 비교
- Adversarial robustness의 공격/방어 이해 (FGSM, PGD, Adversarial Training)
- Conformal Prediction의 **distribution-free, finite-sample** 보장 이해
- 4개 기둥이 어떻게 연결되는지 통합적 관점

---

## 1. 왜 배포가 어려운가? — 4개 기둥

```
                ┌─ Distribution Shift ─ "환경이 바뀌면?"
                │
배포 환경       ├─ OOD Detection ────── "모르는 입력?"
신뢰도 문제    │
                ├─ Adversarial Robust ── "의도적 공격?"
                │
                └─ Conformal Prediction ─ "통계적 보장?"
```

### ⚠️ IID 가정의 한계
- 전통적 ML: train = test distribution (IID)
- 현실: **배포 환경은 학습과 다를 수 있음**
- Calibration, accuracy, uncertainty — 모두 shift에 취약

---

# 🟦 Part 1. Distribution Shift (분포 변화)

## 2. 3가지 Shift 유형 ⭐⭐⭐ (시험 최우선)

| Shift 유형 | 수식 조건 | 변하는 것 | 예시 |
|-----------|-----------|-----------|------|
| **Covariate Shift** | `P_s(X) ≠ P_t(X)`, `P(Y\|X)` 동일 | 입력 분포 | 낮→밤 이미지, 병원 A→B 스캐너 |
| **Label Shift (Prior Shift)** | `P_s(Y) ≠ P_t(Y)`, `P(X\|Y)` 동일 | 클래스 빈도 | 독감 유행 시즌별 변동 |
| **Concept Drift** | `P(Y\|X)` 자체가 변함 | 레이블 의미 | 스팸 정의 변화, 트렌드 변화 |

### 왜 Reliability가 Shift에 취약한가
Calibration은 예측-결과의 joint 분포에 대한 **conditional statement**
→ joint 분포가 바뀌면 모델 가중치 고정되어도 calibration 깨짐

---

## 3. Label Shift 대응 ⭐⭐

### 3-1. Label-Shift Correction Formula (시험 필수)
```
P_t(Y=y | X=x) = w_y · P_s(Y=y | X=x) / Σ_j w_j · P_s(Y=j | X=x)
w_y = P_t(Y=y) / P_s(Y=y)
```
- Source에서 학습한 분류기 posterior에 class prior ratio `w_y`를 가중치로
- Source calibrated classifier도 target prior 바뀌면 잘못됨

### 3-2. BBSE (Black-Box Shift Estimation)
- 레이블 없이도 `w_y`를 추정 가능
- Confusion matrix 기반

### 3-3. LaSCal (Popordanoska et al., NeurIPS 2024) ⭐
- **Label-shift Calibration without target labels**
- Target prediction에서 shift weight 추정 후 calibration에 삽입

**Derivation chain**:
```
shift identity → estimate weights from target predictions → shift-aware recalibration
```

---

## 4. Covariate Shift 대응

### 4-1. Importance Weighting
```
E_t[ℓ(f(X), Y)] = E_s[ (P_t(X)/P_s(X)) · ℓ(f(X), Y) ]
```
- 학습 시 각 샘플을 밀도 비율로 가중치
- 실전에서는 density ratio 추정이 어려움

### 4-2. Domain Adaptation (DA)
- **Target domain 데이터 사용 가능** (unlabeled가 일반적)
- 예: DANN, Test-Time Training, TENT

### 4-3. Domain Generalization (DG)
- **Target domain을 전혀 보지 않고** 학습
- DA보다 어려움
- 예: IRM (Invariant Risk Minimization)

### 4-4. 주요 방법

**DANN (Domain-Adversarial Neural Network)**
- Feature extractor + label classifier + **domain classifier**
- Feature가 domain을 구별 못하게 adversarial 학습
- Gradient reversal layer

**IRM (Invariant Risk Minimization)**
- 모든 environment에서 동시에 최적인 predictor 학습
- Spurious correlation 제거

**Test-Time Adaptation (TTA)**
- 배포 시 target 데이터로 업데이트
- **TENT**: test entropy 최소화로 BN 통계 적응
- **BN adaptation**: target 데이터로 BatchNorm 통계만 업데이트

---

## 5. 중요 실증 논문

### 5-1. Ovadia et al. (NeurIPS 2019) ⭐
- 불확실성이 shift 하에서 급격히 악화됨을 benchmark
- 깨끗한 IID에서는 괜찮아 보이던 uncertainty가 shift에서 무너짐
- Deep ensemble이 가장 robust한 경향

### 5-2. Minderer et al. (NeurIPS 2021)
- 최신 신경망에서 calibration fragility 재확인
- Vision Transformer 등 신규 구조도 여전히 취약

---

# 🟩 Part 2. OOD Detection (분포 외 탐지)

## 6. 기본 개념

### 6-1. 문제 설정
- 학습 분포 `D_train` vs 배포 입력
- 학습에 없던 입력이 들어오면 "모른다"고 거부해야 함
- **Closed-world**: 모든 test = training class
- **Open-world**: test에 새로운 class 가능

### 6-2. OOD가 Closed-world 시스템을 깨뜨리는 이유
- Softmax 분류기는 known class 중 선택하도록 강제됨
- 따라서 **자신 있게 틀림** (confidently wrong)
- 높은 confidence가 신뢰성 보장 안 됨

---

## 7. OOD Score 함수 ⭐⭐⭐ (시험 핵심)

### 7-1. MSP (Maximum Softmax Probability) — Baseline
```
s(x) = max_k p_k(x)
```
- 가장 단순. Hendrycks & Gimpel (2017)
- **한계**: Softmax는 정규화 때문에 overconfident 경향

### 7-2. ODIN (Liang et al., 2018)
- MSP의 개선
- 2가지 기법 조합:
  1. **Temperature scaling** (T>1 → 분포 부드럽게)
  2. **Input perturbation** (작은 gradient 역방향)

### 7-3. Mahalanobis Distance (Lee et al., 2018)
```
s(x) = -min_k (f(x) - μ_k)ᵀ Σ⁻¹ (f(x) - μ_k)
```
- Feature space에서 class 중심까지 거리
- Class-conditional Gaussian 가정
- 각 class의 mean `μ_k`와 공유 공분산 `Σ` 추정

### 7-4. Energy Score (Liu et al., 2020) ⭐⭐
```
E(x) = -T · log Σ_k exp(z_k(x) / T)
```
- Logit의 **logsumexp**
- **왜 MSP보다 나은가**:
  - Softmax는 정규화 때문에 정보 손실
  - Energy는 **로짓 전체 크기**를 반영
  - 이론적으로 **log-likelihood와 직접 연결** (free energy)

### 7-5. Score 함수 비교표 ⭐
| 방법 | 수식 | 특징 |
|------|------|------|
| **MSP** | `max_k p_k(x)` | 가장 단순, overconfident |
| **ODIN** | MSP + temperature + perturbation | MSP 개선판 |
| **Mahalanobis** | `-min_k (f-μ_k)ᵀΣ⁻¹(f-μ_k)` | Feature-level, class-conditional |
| **Energy** | `-T log Σ e^(z_k/T)` | 이론적으로 가장 원리적 |

---

## 8. Outlier Exposure (OE)

### 아이디어
- 학습 중 외부 OOD 데이터 `D_OE` 사용
- OOD 입력에 대해 **uniform 분포** 출력하도록 규제
```
L_OE = E_{x ~ D_OE} [KL(uniform ‖ p(x))]
```
- 전체 loss: `L = L_CE + λ · L_OE`

### 장단점
- **장점**: 실제 OOD 샘플 활용 가능
- **단점**: 대표적 OOD 확보 어려움

---

## 9. OOD Detection 평가 지표 ⭐

| 지표 | 정의 | 해석 |
|------|------|------|
| **AUROC** | ROC curve 아래 넓이 | ID vs OOD ranking quality |
| **FPR@95TPR** | TPR=95%일 때 FPR | 낮을수록 좋음 |
| **AUPR** | Precision-Recall curve 아래 | Imbalanced에서 정보량 많음 |

---

## 10. OOD Detection과 Selective Prediction 결합

### Narasimhan et al. (ICLR 2024) — Plugin Estimators ⭐
- Selective prediction framework를 open-world로 확장
- Bayes-optimal open-world reject rule:
```
decision = f( P(correct|x), P(ID|x), τ )
```
- 여러 요소를 각각 추정 후 plugin
- **Derivation chain**: Bayes-optimal formula → 성분 추정 → plugin selective classifier

---

# 🟥 Part 3. Adversarial Robustness (적대적 강건성)

## 11. Adversarial Example 정의

### 정의
사람 눈엔 동일해 보이는 미세한 perturbation `δ`를 더해 모델을 속이는 입력:
```
‖δ‖_p ≤ ε  이면서  f(x+δ) ≠ f(x)
```
- 일반적으로 `p = ∞` (각 픽셀 변화량의 최댓값)
- `ε` = budget (보통 아주 작음, 예: 8/255)

### 왜 존재하는가
- 고차원 공간의 기하학적 특성
- 선형 근사에서도 많은 작은 변화가 축적됨

---

## 12. 주요 Attack (공격) ⭐⭐⭐

### 12-1. FGSM (Fast Gradient Sign Method) — Goodfellow et al. (2015)
```
x_adv = x + ε · sign(∇_x L(f(x), y))
```
- **1-step 공격**, 빠르고 약함
- Gradient의 **부호**만 사용 → L_∞ ball 안 최대 손실 방향

### 12-2. PGD (Projected Gradient Descent) — Madry et al. (2018) ⭐
```
x^(t+1) = Π_{B_ε(x)} ( x^t + α · sign(∇_x L(f(x^t), y)) )
```
- FGSM을 **반복 적용** + `ε`-ball 안으로 projection
- **현재 표준 공격** (strongest first-order)
- 수십 번 반복이 일반적

### 12-3. C&W (Carlini-Wagner)
- 최적화 기반, 매우 강력하지만 느림
- L_2 norm 공격에 주로 사용

### 12-4. Attack 비교
| 공격 | 특성 | 사용 |
|------|------|------|
| **FGSM** | 1-step, 빠름, 약함 | 초기 탐색 |
| **PGD** | 반복, 강함 | 표준 평가 |
| **C&W** | 최적화 기반, 매우 강함 | 정밀 평가 |

---

## 13. Threat Model (공격 모델)

### 13-1. 정보 접근 수준
| 모델 | 공격자가 아는 것 |
|------|-----------------|
| **White-box** | 모델 구조 + gradient 모두 (강한 가정) |
| **Black-box** | 입출력만 |
| **Gray-box** | 일부 정보만 |

### 13-2. Transfer Attack
- 다른 모델에서 만든 adversarial example이 종종 작동
- 이는 adversarial 방향이 어느 정도 **universal**임을 시사

---

## 14. Adversarial Training (방어) ⭐⭐⭐

### 14-1. Min-Max Formulation (시험 필수)
```
min_θ  E_{(x,y) ~ D} [ max_{‖δ‖_p ≤ ε} L(f_θ(x+δ), y) ]
```

### 14-2. 해석
- **내부 max**: 최악의 perturbation 찾기 (보통 PGD 사용)
- **외부 min**: 그 최악 상황에서 loss 최소화
- **Saddle-point 문제**

### 14-3. Madry et al. (2018) ⭐
- **PGD adversarial training 표준화**
- 현대 adversarial robustness 연구의 출발점
- "우리 모델이 PGD에도 강하면 아마 웬만한 공격에도 강할 것"

---

## 15. Robustness-Accuracy Trade-off ⭐

### 15-1. 현상
- 강건하게 만들수록 **clean accuracy 떨어짐**
- 이론적으로도 증명됨 (일부 상황)

### 15-2. TRADES (Zhang et al., 2019)
Trade-off를 손실함수로 명시:
```
L_TRADES = L(f(x), y) + β · KL(f(x) ‖ f(x_adv))
```
- 첫 항: clean 정확도
- 둘째 항: adversarial 근방에서 **출력 일관성** (smoothness)
- `β`: trade-off 조절

---

## 16. Certified Robustness (인증 가능한 강건성) ⭐

### 16-1. Empirical vs Certified
- **Empirical**: PGD에 버티는가? (경험적)
- **Certified**: 수학적으로 **보장**됨

### 16-2. Randomized Smoothing (Cohen et al., 2019)
**아이디어**: 입력에 가우시안 노이즈 주입 후 다수결
```
g(x) = argmax_c P_{δ ~ N(0, σ²I)} [f(x + δ) = c]
```

**보장**: 어떤 `‖δ‖_2 ≤ R`에 대해서도 `g(x + δ) = g(x)`
- `R`은 `σ`와 투표 margin에서 계산 가능
- **L_2 ball 안 강건성 통계적 증명**

### 16-3. 왜 중요한가
- Empirical 방어는 새 공격에 깨질 수 있음
- Certified는 **수학적 보장** → 안전 critical 응용에 중요

---

## 17. Distribution Shift와 Adversarial의 관계 ⭐

### 통합 관점
- **Adversarial = worst-case distribution shift**
  - ε-ball 안에서 **최악의** shift
- **Distribution shift = natural shift**
  - 자연스러운 변화

둘은 **연속체**로 연결됨:
```
IID ——— Natural Shift ——— Adversarial (worst-case)
```

---

# 🟨 Part 4. Conformal Prediction (등각 예측) ⭐⭐⭐

> **8강에서 가장 중요한 주제** — 최근 연구에서 급부상

## 18. 핵심 아이디어

### 18-1. Point Prediction vs Set Prediction
- 전통적: 하나의 class/value 예측
- Conformal: **prediction set** `C(x) ⊆ Y` 출력

### 18-2. 보장
```
P(Y_test ∈ C(X_test)) ≥ 1 - α
```
- `α`: miscoverage rate (예: α=0.1 → 90% coverage)
- **Distribution-free**: 데이터 분포 가정 없음
- **Finite-sample**: 유한 샘플에서 정확히 성립
- **Model-agnostic**: 어떤 모델이든 사용 가능 (deep net, XGBoost, ...)

### 18-3. 조건
**Exchangeability**만 있으면 됨 (IID보다 약한 조건)
- 샘플 순서만 바꿔도 joint 분포 동일

---

## 19. Split Conformal Prediction 알고리즘 ⭐⭐ (시험 필수)

### Step 1: Data Split
데이터를 둘로:
- Training set
- **Calibration set** (이게 핵심!)

### Step 2: 모델 학습
Training set으로 `f̂` 학습

### Step 3: Nonconformity Score 계산
Calibration set `{(x_i, y_i)}_{i=1}^n`에서:

**분류**:
```
s_i = 1 - f̂(x_i)_{y_i}
```
- True class 확률의 여수 (작을수록 잘 맞춤)

**회귀**:
```
s_i = |y_i - f̂(x_i)|
```
- 절대 오차

### Step 4: Quantile 계산 ⭐
```
q̂ = Quantile( s_1, ..., s_n ; ⌈(n+1)(1-α)⌉ / n )
```
- 약간 보정된 quantile (finite-sample 보장을 위해)

### Step 5: Prediction Set 구성
새 입력 `x_test`:

**분류**:
```
C(x_test) = { y : 1 - f̂(x_test)_y ≤ q̂ }
```

**회귀**:
```
C(x_test) = [ f̂(x_test) - q̂, f̂(x_test) + q̂ ]
```

---

## 20. 핵심 보장 (Marginal Coverage) ⭐

### 정리 (Vovk et al.)
```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
```
- `(n+1)`개 샘플의 exchangeability만 필요!

### ⚠️ 매우 중요한 구분
**Marginal Coverage** vs **Conditional Coverage**
- **Marginal**: 평균적으로 `1-α` 보장 (전체 기대값)
- **Conditional**: 특정 `x`에서 `P(Y ∈ C(x) | X=x) ≥ 1-α` (더 강함)

**Standard Conformal = Marginal only!**
- 즉, 특정 그룹에서는 coverage가 깨질 수 있음
- 이는 한계이자 향후 연구 방향

---

## 21. 고급 Conformal 방법

### 21-1. Adaptive Prediction Sets (APS) — Romano et al. (2020)
**목표**: 불확실한 입력엔 큰 set, 확실한 입력엔 작은 set

Nonconformity score를 누적 확률로:
```
s(x, y) = Σ_{k : f̂(x)_k ≥ f̂(x)_y} f̂(x)_k
```
- 더 **informative**한 set 생성

### 21-2. Conformalized Quantile Regression (CQR) — 회귀용
1. Quantile regression `q̂_α`, `q̂_{1-α}` 학습
2. Conformal로 보정:
```
s_i = max( q̂_α(x_i) - y_i, y_i - q̂_{1-α}(x_i) )
C(x) = [ q̂_α(x) - q̂, q̂_{1-α}(x) + q̂ ]
```

### 21-3. Weighted Conformal Prediction — Shift 하
- Standard CP는 **exchangeability** 요구
- Covariate shift 하에선 exchangeability 깨짐
- **Importance weight** `w(x)`로 nonconformity score 재가중
- Shift 하에서도 coverage 보장 복원

---

## 22. Conformal Prediction vs Calibration ⭐⭐ (단골 출제)

| 특성 | Calibration (Temperature Scaling 등) | Conformal Prediction |
|------|--------------------------------------|----------------------|
| **출력** | 보정된 확률값 | Prediction set |
| **보장** | Asymptotic, 분포 의존적 | **Finite-sample**, 분포 무관 |
| **가정** | 많음 (모델, 데이터) | **Exchangeability만** |
| **해석** | "이 숫자는 정확" | "이 set 안에 정답이 1-α 확률로 있음" |
| **Coverage** | 항상 보장 안 됨 | **수학적 보장** |
| **유연성** | 기존 pipeline 유지 | Set 출력 사용 필요 |

### 언제 무엇을 쓰나?
- **Calibration**: 확률 숫자 자체가 필요할 때 (예: 의사결정 시스템)
- **Conformal**: 통계적 보장이 필요할 때 (예: 의료, 법적 책임)
- 둘은 **상호보완적** (하나가 다른 하나를 대체 안 함)

---

## 23. 최근 연구 방향

### 23-1. Conditional Coverage 확보
- Marginal을 넘어 subgroup/feature-conditional coverage 추구
- Multicalibration과 연결

### 23-2. Distribution-Shift Robust Conformal
- Weighted CP
- Covariate/label shift adaptation

### 23-3. LLM에서의 Conformal Prediction
- Sampling 기반 uncertainty
- Token-level vs response-level

### 23-4. Conformal Risk Control
- Coverage 말고 더 일반적인 risk 제어

---

# 🔗 Part 5. 통합 관점

## 24. 4개 기둥 연결 고리

```
Distribution Shift ←── 넓게는 ──→ Adversarial Attack
       ↓                               ↓
   (natural)                     (worst-case)

OOD Detection ←── 입력이 ID인지 ID가 아닌지 결정
       ↓
Conformal Prediction ←── 예측에 통계적 보장 부여
                          (distribution-free, finite-sample)
```

### 통합 교훈
1. **Distribution shift와 adversarial은 연속체**: natural → worst-case
2. **OOD detection은 shift의 극단 케이스**: "완전 다른 분포"
3. **Conformal은 위의 모든 것에 대한 통계적 안전망 제공**
4. 안전 critical 시스템은 **네 가지 모두 필요**

---

## 25. Deployment 실무 체크리스트 ⭐

### 배포 전 점검 7단계
1. Source/target distribution 차이를 **가설화**
2. 3가지 shift 중 어떤 것이 예상되는가?
3. OOD 입력 가능성이 있는가? → OOD detector 추가
4. 적대적 입력 가능성이 있는가? → Adversarial training 고려
5. 의사결정에 통계적 보장이 필요한가? → Conformal
6. Calibration set을 deployment distribution에 가깝게 확보
7. **Stress test**: 예상되는 shift로 시뮬레이션

---

## 📝 시험 필수 암기 체크리스트

### 🔑 반드시 쓸 수 있어야 할 공식 7개
- [ ] **Label shift 보정**: `w_y = P_t(Y=y) / P_s(Y=y)`
- [ ] **Importance weighting**: `E_t[ℓ] = E_s[(P_t/P_s) · ℓ]`
- [ ] **Energy Score**: `E(x) = -T log Σ_k exp(z_k/T)`
- [ ] **FGSM**: `x_adv = x + ε · sign(∇_x L)`
- [ ] **PGD**: `x^(t+1) = Π_{B_ε}(x^t + α · sign(∇_x L))`
- [ ] **Adversarial Training**: `min_θ E[max_{‖δ‖≤ε} L(f(x+δ), y)]`
- [ ] **Conformal Quantile**: `q̂ = Quantile(s_1,...,s_n; ⌈(n+1)(1-α)⌉/n)`

### 🧠 개념 이해 (서술형 자주 출제)
- [ ] Covariate / Label / Concept shift 차이와 각 예시
- [ ] Calibration이 shift에 취약한 이유 (conditional statement)
- [ ] **MSP vs Energy Score**, Energy가 나은 이유
- [ ] **ODIN, Mahalanobis, Energy** 각각의 작동 방식
- [ ] PGD adversarial training의 **min-max 구조** 설명
- [ ] **Randomized smoothing**이 empirical 방어와 다른 점
- [ ] Robustness-Accuracy trade-off 설명
- [ ] **Conformal Prediction이 "distribution-free"인 이유**
- [ ] Conformal coverage가 **marginal인가 conditional인가** (중요!)
- [ ] Exchangeability가 IID보다 약한 조건인 이유
- [ ] **Adversarial attack vs Distribution shift** 관계
- [ ] TRADES의 두 항의 의미
- [ ] Split Conformal 5단계 알고리즘
- [ ] Marginal vs Conditional coverage 차이
- [ ] Calibration vs Conformal Prediction 비교

### 🎯 비교표 암기
- [ ] Shift 3종 (Covariate / Label / Concept)
- [ ] OOD score 4종 (MSP / ODIN / Mahalanobis / Energy)
- [ ] Attack 3종 (FGSM / PGD / C&W)
- [ ] Threat model 3종 (White-box / Black-box / Gray-box)
- [ ] Domain Adaptation vs Domain Generalization
- [ ] Calibration vs Conformal Prediction
- [ ] Empirical vs Certified robustness

### 📐 알고리즘 단계 암기
- [ ] **Split Conformal Prediction 5단계** ⭐ (가장 중요!)
- [ ] Adversarial Training 반복 구조
- [ ] PGD 반복 업데이트

### 🔗 통합 질문 대비
- [ ] 4개 기둥이 어떻게 연결되는가?
- [ ] 특정 배포 환경에서 어느 기법을 선택할지?
- [ ] Medical AI, Autonomous driving 같은 시나리오에서 필요한 조합

---

## 💡 시험 직전 마지막 정리

**세 줄 요약**:
1. **Distribution Shift**: 배포 환경이 바뀌면 calibration/정확도가 무너짐 → Label shift 보정, DA, DG
2. **OOD Detection**: 모르는 입력을 거부 → Energy Score가 이론적으로 가장 원리적
3. **Adversarial Robustness**: 의도적 공격에 버티기 → PGD adversarial training (min-max)
4. **Conformal Prediction**: 통계적 보장 → Distribution-free, finite-sample, marginal coverage

**가장 자주 나올 문제 Top 3**:
1. Split Conformal Prediction 알고리즘 단계와 coverage 보장 설명
2. PGD adversarial training의 min-max 구조 유도
3. OOD score 함수들 (MSP/Mahalanobis/Energy) 비교 및 수식
