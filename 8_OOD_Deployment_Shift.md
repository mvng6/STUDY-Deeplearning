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

---

# 📚 Part 6. 주요 개념 심화 보충 학습

> 위의 핵심 정리를 바탕으로, 시험에서 자주 나오는 주요 개념들을 **쉬운 언어와 예시, 코드**로 풀어 설명한 보충 자료

---

## A. Mahalanobis Distance 심화 이해

### A-1. 핵심 아이디어: "분포를 고려한 거리"

일반적인 **유클리드 거리**는 자로 잰 직선 거리로, 데이터가 어떻게 분포되어 있는지 전혀 신경 쓰지 않는다. 반면 **마할라노비스 거리**는 "이 데이터 분포 기준으로 얼마나 이상한 점인가?"를 측정한다.

**직관 예시**: 데이터가 타원형으로 분포되어 있을 때
- 두 점 A, B가 중심에서 같은 유클리드 거리에 있어도
- 점 A는 데이터가 퍼진 방향을 **따라** 있고 → 정상 샘플
- 점 B는 데이터가 퍼진 방향을 **벗어나** 있으면 → OOD 의심!
- 마할라노비스 거리는 이 차이를 구분해준다

### A-2. 수식을 한 덩어리씩 이해하기

```
s(x) = -min_k (f(x) - μ_k)ᵀ Σ⁻¹ (f(x) - μ_k)
```

| 기호 | 의미 | 로봇 예시 비유 |
|------|------|--------------|
| `f(x)` | 입력 x의 특징 벡터 (neural net의 중간층 출력) | 센서 측정값 벡터 |
| `μ_k` | 클래스 k의 평균 특징 벡터 | 클래스 k의 "대표값" |
| `Σ` | 공분산 행렬 | 특징들이 서로 어떻게 퍼져있는지 |
| `Σ⁻¹` | 공분산의 역행렬 | 분포 방향을 "정규화"하는 역할 |

**수식을 말로 풀면**:
1. `(f(x) - μ_k)`: 입력 특징이 클래스 k 평균에서 얼마나 떨어졌나?
2. `Σ⁻¹`를 가운데 곱함: 데이터가 퍼진 방향을 고려해 거리를 **보정**
3. `min_k`: 모든 클래스 중 **가장 가까운** 클래스까지의 거리
4. 앞에 `-` 붙임: 거리가 멀수록 score가 작아짐 (OOD는 score 매우 낮음)

### A-3. 왜 공분산 Σ가 필요한가?

**쉬운 예시**: 로봇의 두 센서값
- 센서 A: 0~100 범위
- 센서 B: 0~1 범위

유클리드 거리를 그냥 쓰면 **센서 A가 거리를 지배**해버린다 (숫자 크기가 크니까).
공분산 행렬 Σ는 이런 **스케일 차이를 자동으로 조정**해주고, 두 센서 간 **상관관계**도 고려한다.

### A-4. OOD Detection 실제 절차 (3단계)

**1단계: 학습 데이터로 통계 추정 (오프라인)**
- 각 클래스 k의 평균 특징벡터 μ_k 계산
- 전체 공분산 행렬 Σ 계산 (모든 클래스 공유)
- 이 둘은 저장해놓고 추후 사용

**2단계: 새 입력 x가 들어오면 (실시간)**
- 신경망 통과시켜 특징 f(x) 추출
- 모든 클래스 k에 대해 마할라노비스 거리 계산
- 가장 가까운 거리 선택

**3단계: 임계값과 비교해 판단**
- 거리가 작음 → 어떤 클래스의 "정상" 분포에 가까움 → ID
- 거리가 큼 → 어느 클래스에도 속하지 않음 → OOD!
- 임계값 τ는 validation set으로 결정

### A-5. 간단한 파이썬 코드

```python
import numpy as np

# 1단계: 학습 데이터로 μ_k와 Σ 미리 계산
def fit_mahalanobis(features, labels, num_classes):
    class_means = []  # 각 클래스의 평균 벡터
    centered = []  # 공분산 계산용
    
    for k in range(num_classes):
        # k번 클래스에 속한 샘플만 뽑기
        class_features = features[labels == k]
        # 평균 계산 (μ_k)
        mu_k = class_features.mean(axis=0)
        class_means.append(mu_k)
        # 중심에서 뺀 값 저장
        centered.append(class_features - mu_k)
    
    # 모든 클래스의 편차로 공유 공분산 Σ 계산
    centered_all = np.concatenate(centered, axis=0)
    cov = np.cov(centered_all.T)
    # 역행렬 Σ⁻¹ 미리 계산 (한 번만 하면 됨)
    cov_inv = np.linalg.inv(cov)
    
    return class_means, cov_inv


# 2단계: 새 입력의 OOD score 계산
def mahalanobis_score(x_feature, class_means, cov_inv):
    distances = []
    for mu_k in class_means:
        # 중심에서 떨어진 양
        diff = x_feature - mu_k
        # 마할라노비스 거리 제곱: diff^T @ Σ⁻¹ @ diff
        dist = diff @ cov_inv @ diff
        distances.append(dist)
    # 가장 가까운 클래스까지의 거리 (음수로 해서 score로 사용)
    score = -min(distances)
    return score


# 3단계: 임계값으로 OOD 판정
def is_ood(score, threshold):
    return score < threshold
```

### A-6. 장단점

**장점**
- Feature 공간에서 작동 → softmax보다 풍부한 정보 사용
- 클래스별 구조를 고려 → 각 클래스의 "정상 영역"을 타원으로 모델링
- 학습 후 재학습 없이 적용 가능 (μ_k, Σ만 한 번 계산)

**단점**
- 가우시안 가정: 각 클래스 특징이 정규분포라고 가정 (실제론 그렇지 않을 수 있음)
- 고차원에서 Σ 추정 어려움: 특징 차원이 크면 공분산 계산 불안정
- 역행렬 계산 비용: Σ⁻¹ 구하는 게 무거움 (하지만 한 번만 하면 됨)

---

## B. ODIN 심화 이해

### B-1. 왜 ODIN이 필요한가

MSP의 문제점은 신경망이 **OOD 입력에도 높은 확률**을 부여한다는 것. 모르는 입력인데도 "나 이거 확신해!"라고 자신감 있게 틀린다 (confidently wrong).

ODIN은 이 문제를 해결하려고 **두 가지 트릭**을 조합한다.

### B-2. 기법 1: Temperature Scaling (온도 조절)

기존 softmax에 **온도 T**를 추가:
```
p_k(x; T) = exp(z_k/T) / Σ_j exp(z_j/T)
```

**온도가 하는 역할**:
- T = 1: 원래 softmax
- T > 1: 분포가 **부드러워짐** (확률값들이 평평해짐)
- T → ∞: 모든 클래스가 거의 균등 확률

**왜 T를 크게 하면 ID/OOD 구분이 잘 될까?**

핵심 통찰: **ID와 OOD는 logit 값의 "차이"가 다름**
- **ID 입력**: 정답 클래스 logit이 다른 것들보다 훨씬 큼 (격차가 큼)
- **OOD 입력**: logit들이 비교적 고만고만함 (격차가 작음)

T로 나눠서 분포를 부드럽게 만들면, 이 "격차"가 확률값에 더 잘 드러난다. 논문에선 T = 1000 추천.

### B-3. 기법 2: Input Perturbation (입력 살짝 건드리기)

```
x̃ = x - ε · sign(-∇_x log p_ŷ(x; T))
```

**풀어서 설명**:
1. 입력 x에 대해 **예측된 클래스 ŷ의 확률을 더 높이는 방향**으로 x를 살짝 민다
2. 그 방향으로 작은 섭동 ε만큼 이동
3. 수정된 x̃로 다시 예측 → 이 값으로 OOD 판단

**왜 효과가 있을까?**

핵심 통찰: **ID와 OOD는 perturbation에 대한 반응이 다름**
- **ID 입력**: 이미 자신있는 예측 → 살짝 밀면 확률이 **더 크게 증가**
- **OOD 입력**: 애매한 예측 → 밀어도 확률이 **덜 증가**

즉, "밀었을 때 얼마나 더 확신하게 되는가?"가 ID/OOD 구분 신호가 된다.

### B-4. ODIN 전체 절차

1. **Input Perturbation**: 예측 확률을 높이는 방향으로 x를 ε만큼 밀기 → x̃
2. **Temperature-scaled Softmax**: x̃를 신경망에 통과시켜 logit 얻고, T로 나눠 softmax
3. **MSP로 판단**: s(x) = max_k p_k(x̃; T), 임계값 τ보다 크면 ID, 작으면 OOD

### B-5. 간단한 파이썬 코드

```python
import torch
import torch.nn.functional as F

def odin_score(model, x, temperature=1000, epsilon=0.0014):
    """
    ODIN OOD score 계산
    """
    # 입력에 gradient 계산 가능하게 설정
    x = x.clone().detach().requires_grad_(True)
    
    # 1단계 준비: logit 뽑고 temperature 적용
    logits = model(x)
    scaled_logits = logits / temperature
    
    # log softmax로 안정적으로 계산
    log_probs = F.log_softmax(scaled_logits, dim=1)
    max_log_prob, _ = log_probs.max(dim=1)
    
    # 1단계: 입력 섭동
    # 예측 확률을 높이는 방향의 gradient 계산
    loss = -max_log_prob.sum()
    loss.backward()
    # gradient의 부호 방향으로 x를 아주 살짝 밈
    x_perturbed = x - epsilon * x.grad.sign()
    
    # 2단계: 수정된 입력으로 다시 softmax
    with torch.no_grad():
        logits_new = model(x_perturbed)
        scaled_logits_new = logits_new / temperature
        probs_new = F.softmax(scaled_logits_new, dim=1)
    
    # 3단계: MSP 계산
    score, _ = probs_new.max(dim=1)
    return score
```

### B-6. 장단점

**장점**
- 재학습 불필요: 이미 학습된 모델에 바로 적용
- 구현 간단: MSP + 두 가지 트릭만 추가
- MSP 대비 성능 향상 확실함

**단점**
- Hyperparameter 2개: T와 ε을 validation set으로 튜닝 필요
- Gradient 계산 필요: 추론 시 역전파 한 번 → 약간 느림
- 여전히 softmax 기반: 근본적 한계 존재

---

## C. Adversarial Robustness 심화 이해

### C-1. Adversarial Example의 놀라운 현상

사람 눈에는 **완전히 똑같아 보이는** 이미지인데, 픽셀 몇 개를 아주 미세하게 바꾸면 신경망이 완전히 다른 답을 내놓는다.

**예시**: 판다 사진 + 사람 눈에 안 보이는 미세 노이즈 = 모델이 "긴팔원숭이" (99.3% 확신)로 오답

**왜 이런 일이 생길까?**
고차원 공간의 기하학적 특성 때문. 신경망은 결정 경계 근처에서 **선형에 가깝게** 동작하는데, 작은 변화들이 여러 차원에 걸쳐 축적되면 경계를 넘어가버린다.

**로봇 제어 맥락**:
- 카메라 센서에 약간의 스티커 → 자율주행차가 표지판 오인식
- LiDAR 포인트 클라우드에 미세 조작 → 장애물 감지 실패
- 안전이 중요한 시스템에서 치명적인 문제

### C-2. FGSM 공격 구현

```python
def fgsm_attack(model, x, y, epsilon=0.03):
    # 입력에 gradient 계산 켜기
    x = x.clone().detach().requires_grad_(True)
    # forward pass
    output = model(x)
    loss = F.cross_entropy(output, y)
    # gradient 계산
    loss.backward()
    # gradient 부호 방향으로 한 번에 이동
    x_adv = x + epsilon * x.grad.sign()
    # 픽셀 값이 [0, 1] 범위 벗어나지 않게 자르기
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()
```

### C-3. PGD 공격 구현

PGD = **FGSM을 여러 번 반복** + **안전 범위 안에 머물게 projection**

1. 작은 step size α로 FGSM을 반복 적용
2. 매번 ε-ball 안으로 투영 (원본에서 너무 멀어지지 않게)
3. 수십 번 반복하면 훨씬 강한 공격

```python
def pgd_attack(model, x, y, epsilon=0.03, alpha=0.007, steps=10):
    # 원본 저장 (projection 기준점)
    x_original = x.clone().detach()
    
    # 랜덤 초기화 (ε-ball 안에서)
    x_adv = x_original + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1).detach()
    
    # 반복 공격
    for _ in range(steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        loss.backward()
        # FGSM 스타일로 한 step 이동
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        # ε-ball 안으로 projection
        delta = torch.clamp(x_adv - x_original, -epsilon, epsilon)
        x_adv = torch.clamp(x_original + delta, 0, 1).detach()
    
    return x_adv
```

### C-4. Adversarial Training Min-Max 구조 이해

```
min_θ  E_{(x,y) ~ D} [ max_{‖δ‖_p ≤ ε} L(f_θ(x+δ), y) ]
```

**두 덩어리로 이해**:

**내부 max: 공격자 역할**
- 수식: `max_{‖δ‖≤ε} L(f(x+δ), y)`
- 의미: "이 입력 x를 가장 잘 속일 수 있는 perturbation δ를 찾자!"
- 보통 PGD로 계산 (수십 번 반복)
- 각 학습 샘플마다 adversarial example 생성

**외부 min: 방어자 역할 (모델 학습)**
- 수식: `min_θ E[...]`
- 의미: "그 최악의 공격에도 손실이 작도록 모델 파라미터 θ를 학습!"
- 일반적인 SGD로 θ 업데이트
- 단, 깨끗한 x 대신 adversarial x+δ 사용

두 단계를 매 배치마다 반복 — 게임처럼 공격과 방어가 공진화한다.

### C-5. Adversarial Training 알고리즘 구조

```python
# Adversarial Training 기본 구조
for epoch in range(num_epochs):
    for x, y in train_loader:
        # 내부 max: PGD로 최악의 공격 찾기
        x_adv = pgd_attack(model, x, y, epsilon, alpha, steps=7)
        
        # 외부 min: adversarial 입력으로 학습
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        
        # 모델 파라미터 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### C-6. Randomized Smoothing의 의미

**아이디어**: 입력에 **가우시안 노이즈**를 여러 번 주입하고 **다수결**로 답 결정

```
g(x) = argmax_c P_{δ ~ N(0, σ²I)} [f(x+δ) = c]
```

**왜 이게 강건한가?**
- 노이즈를 **많이** 주입하므로 작은 공격 perturbation은 묻혀버림
- 수학적으로 어떤 `‖δ‖_2 ≤ R` 반경 안에서는 **답이 안 바뀜**이 증명됨
- R은 σ와 투표 margin에서 계산됨

**왜 중요한가?**
- 안전 critical 시스템 (의료, 자율주행)에 **수학적 보장** 제공
- 새 공격이 나와도 보장이 유지됨

---

## D. TRADES 심화 이해

### D-1. 왜 TRADES가 필요한가

기존 Madry의 adversarial training은 adversarial 입력으로만 학습시킨다:
```
min_θ E[max L(f(x+δ), y)]
```

**문제**: 깨끗한 입력에 대한 정확도(clean accuracy)가 많이 떨어지고, **trade-off를 조절할 수 있는 손잡이**가 없음.

**TRADES의 해결책**: 손실함수를 **두 항으로 쪼개고**, 사이에 조절 knob β를 넣음

### D-2. TRADES 손실 함수 두 항 이해

```
L_TRADES(x, y) = -log p_θ(y|x) + β · max_{‖δ‖≤ε} KL(p_θ(·|x) ‖ p_θ(·|x+δ))
```

**첫 번째 항: `-log p(y|x)`**
- 일반 cross-entropy loss
- 깨끗한 x로 정답 맞추기
- **clean accuracy 유지** 담당
- "기본기를 잃지 말자"

**두 번째 항 (× β): `KL(p(·|x) ‖ p(·|x+δ))`**
- KL divergence로 출력 분포 간 차이 측정
- δ 섭동 전후의 예측 비교
- **출력 일관성 (smoothness)** 담당
- "흔들리지 말자"

### D-3. KL Divergence의 역할

KL divergence는 **두 확률 분포가 얼마나 다른가**를 측정한다:
```
KL(P ‖ Q) = Σ_k P(k) log(P(k) / Q(k))
```
- 두 분포가 똑같으면 KL = 0
- 두 분포가 다를수록 KL이 커짐

TRADES에서는 "**원본 x에 대한 예측**"과 "**공격당한 x+δ에 대한 예측**"이 얼마나 다른지를 측정한다. TRADES는 KL이 큰 모델을 penalize해서, 학습이 진행되면서 공격 전후 출력이 일관되도록 유도한다. 이게 "**predictive distribution을 stabilize**"한다는 의미.

### D-4. β 값의 영향 (실험적 감각)

| β 값 | Clean Acc | Robust Acc | 특징 |
|------|-----------|------------|------|
| 0 | 95% | 0% | 일반 학습과 동일 |
| 1 | 90% | 40% | 약한 robust |
| **6** | **85%** | **55%** | **TRADES 논문 추천값** |
| 10 | 80% | 58% | robust 우선 |

핵심: β는 **task 요구사항에 맞춰 선택**하는 하이퍼파라미터. 안전이 중요하면 크게, 성능이 중요하면 작게.

### D-5. Madry AT vs TRADES 비교

| 특성 | Madry AT | TRADES |
|------|----------|--------|
| **수식 구조** | min-max 한 줄 | 두 항의 합 |
| **학습 입력** | adversarial만 사용 | clean + adversarial 둘 다 |
| **Clean accuracy** | 크게 감소 | 덜 감소 |
| **Trade-off 조절** | 불가능 | β로 가능 |
| **철학** | "공격받은 입력에서도 정답을 맞춰라" | "공격 전후 예측이 비슷해야 한다" |

### D-6. 간단한 파이썬 코드

```python
import torch
import torch.nn.functional as F

def trades_loss(model, x, y, epsilon=0.03, alpha=0.007, 
                steps=10, beta=6.0):
    # 1. 원본 x의 출력 분포 계산
    model.eval()
    with torch.no_grad():
        logits_clean = model(x)
        probs_clean = F.softmax(logits_clean, dim=1)
    
    # 2. PGD로 최악의 perturbation δ 찾기 (KL 최대화)
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits_adv = model(x_adv)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        # KL divergence 계산 (최대화할 대상)
        kl = F.kl_div(log_probs_adv, probs_clean, 
                      reduction='batchmean')
        # gradient로 KL이 커지는 방향 찾기
        grad = torch.autograd.grad(kl, x_adv)[0]
        # PGD 업데이트
        x_adv = x_adv.detach() + alpha * grad.sign()
        # ε-ball 안으로 projection
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
    
    # 3. TRADES 손실 계산
    model.train()
    # 첫 항: clean 입력으로 cross-entropy
    logits_clean = model(x)
    loss_clean = F.cross_entropy(logits_clean, y)
    # 둘째 항: KL divergence (출력 일관성)
    logits_adv = model(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(logits_adv, dim=1),
        F.softmax(logits_clean, dim=1),
        reduction='batchmean'
    )
    # 두 항 결합
    total_loss = loss_clean + beta * loss_kl
    return total_loss
```

### D-7. TRADES의 철학적 의미

**"Stabilize the predictive distribution"**
- Madry AT: "공격당한 입력에서도 정답을 맞춰라"
- TRADES: "공격당하기 전과 후의 예측이 비슷해야 한다"

**왜 더 원리적인가?**
- Adversarial 문제의 본질 = **결정 경계가 입력 근처에 너무 가깝게 있음**
- TRADES는 이 본질을 **직접 공략** (경계를 멀리 밀어내는 효과)
- Madry AT는 간접적으로만 공략

---

## E. Conformal Prediction 심화 이해

### E-1. Point Prediction vs Set Prediction

**기존 방식 (Point Prediction)**
- 입력: 고양이 사진
- 출력: "고양이" (확률 85%)
- 문제: 모델이 애매할 때도 무조건 하나만 고르게 됨. 틀리면 그냥 틀린 거.

**Conformal Prediction (Set Prediction)**
- 입력: 애매한 동물 사진
- 출력: **{고양이, 강아지, 토끼}** ← 집합!
- 보장: 이 집합 안에 정답이 1-α 확률로 있음

### E-2. 핵심 보장 공식 풀어보기

```
P{Y_{n+1} ∈ C(X_{n+1})} ≥ 1 - α
```

| 기호 | 의미 |
|------|------|
| `X_{n+1}` | 새로운 입력 |
| `Y_{n+1}` | 그 입력의 **진짜 정답** |
| `C(X_{n+1})` | 모델이 반환한 **prediction set** |
| `α` | **miscoverage rate** (틀릴 허용치) |
| `1-α` | **coverage** (맞을 최소 확률) |

**말로 풀면**: "새로운 입력의 **진짜 정답**이 내가 반환한 **집합 안에** 들어있을 확률이 **최소 1-α이다**"

**예시 (α = 0.1)**:
- Coverage = 90%
- "100번 예측하면 **최소 90번**은 정답이 내 집합 안에 있다"
- 나머지 10번은 집합이 정답을 놓칠 수 있음 (허용 오차)

### E-3. 세 가지 놀라운 특성

**1. Distribution-free (분포 무관)**
- 데이터 분포에 대한 가정 전혀 없음
- 가우시안? 아니어도 OK
- 단, **exchangeability**만 만족하면 됨 (IID보다 약한 조건)

**2. Finite-sample (유한 샘플 보장)**
- 샘플이 적어도 바로 보장 성립
- n → ∞일 때만이 아니라 n = 100에서도 OK
- 일반 통계와 다른 점: 점근적(asymptotic)이 아님!

**3. Model-agnostic (모델 무관)**
- 어떤 모델이든 사용 가능
- Deep Neural Network, XGBoost, Random Forest, 선형 회귀 등등
- 기존 모델 위에 얇은 wrapper로 추가

### E-4. 전통 방법과의 차이

| 구분 | 전통 방법 | Conformal Prediction |
|------|----------|---------------------|
| **분포 가정** | "정규분포" 가정 | **가정 없음** |
| **보장** | 샘플 무한대 가정 | **유한 샘플에서도 OK** |
| **모델** | 특정 모델에 한정 | **어떤 모델이든 OK** |
| **조건** | IID 필요 | **Exchangeability만** (더 약함) |

### E-5. "conformal"이란 말의 뜻

영어로 "**맞아떨어지는, 부합하는**"이라는 뜻.

**알고리즘 관점**:
1. Calibration set: 검증용으로 따로 떼어둔 데이터
2. 새 입력이 들어오면, 각 후보 레이블에 대해 "이게 얼마나 **conform (부합)**하는가?" 점수 계산
3. 그 점수가 **충분히 conform**하면 집합에 포함
4. "충분히"의 기준은 **coverage 1-α**를 만족하도록 맞춤

즉, 이름 자체가 "**calibration set에 잘 맞아떨어지는 레이블만 선택**"이라는 뜻.

### E-6. 실용적 예시: "dog or rabbit"

모델이 **확신할 수 없을 때**, 하나를 찍어서 틀리느니 **두 개 다 말하는 게 낫다**:
- 기존: "dog" (51%) ← 틀리면 오답
- CP: **{dog, rabbit}** ← 정답이 이 안에 90% 확률로 있음

**로봇 공학 관점 활용**:
- **물체 인식**: 애매하면 "{머그컵, 유리잔, 텀블러}" 반환 → 더 확인 후 집기
- **경로 계획**: 95% 확률로 충돌 없는 경로 집합 반환 → 안전 margin 확보
- **센서 기반 판단**: 불확실하면 크게, 확실하면 작게 set → 적응적 의사결정

### E-7. Set 크기와 α의 관계

| α | Coverage | 기대 set 크기 | 해석 |
|---|----------|-------------|------|
| 0.01 | 99% | 보통 3-5개 | 매우 안전, 큰 집합 |
| 0.05 | 95% | 보통 2-3개 | 표준적 |
| 0.10 | 90% | 보통 1-2개 | 실용적 균형 |
| 0.20 | 80% | 보통 1개 | 느슨함 |

- α가 **작을수록** (예: 0.01) → 더 엄격한 coverage → **집합이 커짐**
- α가 **클수록** (예: 0.2) → 느슨한 coverage → **집합이 작아짐**

### E-8. Split Conformal Prediction 5단계 상세

**1단계: 데이터 분할**
- 전체 데이터를 Training set + Calibration set으로 나눔
- 보통 80:20 정도

**2단계: 모델 학습**
- Training set으로 모델 f̂ 학습 (일반적인 학습 과정과 동일)

**3단계: Nonconformity Score 계산**
- Calibration set의 각 샘플에서 s_i = 1 - f̂(x_i)_{y_i}
- (작을수록 잘 맞춘 것)

**4단계: Quantile 계산**
- `q̂ = Quantile(s_1, ..., s_n ; ⌈(n+1)(1-α)⌉ / n)`
- 유한 샘플 보장을 위해 약간 보정된 quantile

**5단계: Prediction Set 구성**
- 새 입력 x_test에 대해:
- `C(x_test) = { y : 1 - f̂(x_test)_y ≤ q̂ }`
- 즉, "score가 임계값 q̂ 이하인 모든 레이블"을 집합에 포함

이 절차로 `P(Y ∈ C(X)) ≥ 1 - α`가 수학적으로 보장됨 (Exchangeability만 만족하면 됨).

### E-9. ⚠️ Marginal vs Conditional Coverage (시험 필수!)

**Marginal Coverage (Standard CP가 주는 것)**
```
P(Y ∈ C(X)) ≥ 1 - α
```
- **전체 평균적으로** 1-α 보장
- 모든 입력을 통틀어 평균

**Conditional Coverage (더 강한 조건)**
```
P(Y ∈ C(X) | X = x) ≥ 1 - α
```
- **특정 입력 x**에 대해서도 1-α 보장
- 훨씬 강한 조건

**왜 중요한가?**
- Standard Conformal = Marginal only!
- 평균은 90%지만, **특정 그룹에서는 70%만** 보장될 수 있음
- 예: 전체적으로 90% coverage지만, 흑인 환자 그룹에서는 75%만 → 공정성 문제
- 이게 현재 CP 연구의 활발한 방향

### E-10. 직관 한 줄 요약

> **"하나를 찍어 틀리느니, 가능한 후보를 집합으로 반환하고 수학적으로 보장하자"**

이게 Conformal Prediction의 철학이다. 기존 ML의 **overconfident** 문제를 근본적으로 다른 관점에서 해결하는 것.

---

## 🎓 보충 학습 요약 — 한 페이지 Cheat Sheet

### Mahalanobis Distance
- Feature space에서 **분포를 고려한** 거리
- 공분산 Σ가 스케일/상관관계 자동 조정
- Class-conditional Gaussian 가정

### ODIN
- MSP 개선판 = **Temperature (T) + Input Perturbation (ε)**
- ID와 OOD의 logit 격차 증폭 + 반응 민감도 차이 활용

### Adversarial Robustness
- **FGSM**: 1-step, 빠르고 약함
- **PGD**: 반복+projection, 표준 공격
- **Adversarial Training**: min-max 구조, 공격-방어 공진화
- **Randomized Smoothing**: 수학적 보장 있는 certified 방어

### TRADES
- Loss = clean CE + β · KL(p(x) ‖ p(x+δ))
- β로 clean-robust trade-off 명시적 조절
- "Predictive distribution을 stabilize"

### Conformal Prediction
- **Point → Set prediction** 패러다임 전환
- **Distribution-free, finite-sample, model-agnostic**
- Marginal coverage만 보장 (Conditional은 더 강함)
- Exchangeability만 있으면 됨

---

> 📌 이 보충 자료는 핵심 정리 내용을 바탕으로, 시험 및 실무에서 자주 혼동되는 개념들을 쉬운 예시와 코드로 풀어 정리한 것. 특히 로봇 제어 맥락의 응용 가능성을 함께 고려함.
