# GPT 2.0 Implementation — Tiny Character-Level GPT

이 저장소는 **GPT-2의 핵심 구조를 작은 문자 단위 언어모델로 구현한 프로젝트**입니다.

강의 노트의 다음 흐름을 통합했습니다.

- `notebook_04.ipynb`: GPT-style next-token dataset
- `notebook_05.ipynb`: masked self-attention
- `notebook_06.ipynb`: multi-head attention, feedforward network, residual connection, layer normalization, Tiny GPT
- 참고 흐름: Andrej Karpathy의 `nn-zero-to-hero`

> 실제 OpenAI GPT-2 전체 규모를 재현한 것이 아니라, GPT-2의 decoder-only Transformer 구조를 교육용 소형 모델로 구현했습니다.

## 1. 주요 구현 내용

### Character-level tokenization

Tiny Shakespeare 데이터의 각 문자를 하나의 토큰으로 취급합니다.

예를 들어 `ROMEO`는 다섯 개의 문자 토큰으로 변환됩니다.

### Next-token prediction

입력 시퀀스가 다음과 같으면:

```text
ROMEO:
```

모델의 정답은 입력을 한 문자 오른쪽으로 이동한 시퀀스입니다.

```text
OMEO:...
```

모델은 각 위치에서 다음 문자를 예측하도록 학습됩니다.

### Token embedding과 positional embedding

- token embedding: 각 문자 토큰을 벡터로 변환
- positional embedding: 토큰의 순서 정보를 제공

두 embedding을 더한 결과가 Transformer block에 입력됩니다.

### Masked self-attention

미래 토큰을 미리 보지 못하도록 하삼각행렬 causal mask를 사용합니다.

각 토큰은 자신과 이전 토큰만 참고하여 다음 토큰을 예측합니다.

### Multi-head attention

여러 개의 attention head가 서로 다른 관계를 병렬로 학습합니다.  
각 head의 결과를 이어 붙인 후 linear projection을 적용합니다.

### Transformer block

각 block은 다음 구조를 가집니다.

```text
LayerNorm
→ Masked Multi-Head Self-Attention
→ Residual Connection
→ LayerNorm
→ Feedforward Network
→ Residual Connection
```

### Language-model head

마지막 hidden state를 vocabulary 크기의 logits으로 변환합니다.  
Cross-entropy loss를 이용하여 실제 다음 문자와 비교합니다.

## 2. 프로젝트 구조

```text
gpt2_tiny_project/
├── GPT2_implementation.ipynb
├── model.py
├── train.py
├── generate.py
├── generated_text.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. 설치

```bash
pip install -r requirements.txt
```

## 4. 모델 학습

```bash
python train.py
```

기본 설정:

- block size: 64
- batch size: 64
- embedding dimension: 128
- attention heads: 4
- Transformer blocks: 4
- epochs: 10
- epoch당 최대 step: 300

빠른 실행 테스트:

```bash
python train.py --epochs 1 --max-steps-per-epoch 20
```

GPU 환경에서는 기본 설정을 사용할 수 있습니다.

학습이 완료되면 다음 파일이 생성됩니다.

```text
tiny_gpt_checkpoint.pt
training_history.json
```

## 5. 텍스트 생성

```bash
python generate.py --prompt "ROMEO:" --max-new-tokens 500
```

생성 결과는 터미널에 출력되고 다음 파일에도 저장됩니다.

```text
generated_text.txt
```

## 6. 노트북 실행

과제 제출용 결과는 `GPT2_implementation.ipynb`의 셀을 위에서 아래로 실행하여 생성할 수 있습니다.

마지막 셀에는 다음 결과가 포함됩니다.

- epoch별 training loss
- validation loss
- 학습된 모델을 이용한 텍스트 생성
- `generated_text.txt` 저장

## 7. 학습 결과 해석

학습 초기에는 무작위에 가까운 문자가 생성됩니다.  
학습이 진행되면서 다음과 같은 특징이 나타납니다.

- 등장인물 이름과 콜론 형식 학습
- 줄바꿈과 문장 부호 패턴 학습
- 영어 단어와 유사한 문자열 생성
- Shakespeare 대사 구조와 비슷한 텍스트 생성

작은 모델이므로 문법적으로 완벽한 문장을 생성하지는 않지만, 다음 문자 확률분포를 학습하면서 데이터의 형식과 반복 패턴을 재현합니다.

## 8. 과제 제출 전 확인

노트북의 모든 셀을 실행한 뒤 출력이 포함된 상태로 저장합니다.

```bash
git add .
git commit -m "Implement Tiny GPT language model"
git push
```

다음 파일은 GitHub에 올리지 않습니다.

```text
__pycache__/
*.pyc
tiny_gpt_checkpoint.pt
input.txt
```
