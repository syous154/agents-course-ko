# 함수 호출을 위해 모델 미세 조정하기

이제 함수 호출을 위해 첫 번째 모델을 미세 조정할 준비가 되었습니다 🔥.

## 함수 호출을 위해 모델을 어떻게 훈련시키나요?

> 답변: **데이터**가 필요합니다.

모델 훈련 과정은 3단계로 나눌 수 있습니다.

1. **모델은 대량의 데이터로 사전 훈련됩니다**. 이 단계의 결과물은 **사전 훈련된 모델**입니다. 예를 들어, [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)가 있습니다. 이것은 기본 모델이며 **강력한 지침 준수 기능 없이 다음 토큰을 예측하는 방법**만 알고 있습니다.

2. 채팅 컨텍스트에서 유용하려면 모델을 **미세 조정**하여 지침을 따라야 합니다. 이 단계에서는 모델 제작자, 오픈 소스 커뮤니티, 여러분 또는 누구나 훈련시킬 수 있습니다. 예를 들어, [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)는 Gemma 프로젝트를 담당하는 Google 팀이 지침을 튜닝한 모델입니다.

3. 그런 다음 모델을 제작자의 선호도에 맞게 **조정**할 수 있습니다. 예를 들어, 고객에게 절대 무례해서는 안 되는 고객 서비스 채팅 모델이 있습니다.

일반적으로 Gemini 또는 Mistral과 같은 완전한 제품은 **세 단계를 모두 거치지만**, Hugging Face에서 찾을 수 있는 모델은 이 훈련의 하나 이상의 단계를 완료했습니다.

이 튜토리얼에서는 [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)를 기반으로 함수 호출 모델을 구축할 것입니다. 미세 조정된 모델이 우리의 사용 사례에 더 적합하기 때문에 기본 모델 [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) 대신 미세 조정된 모델 [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)를 선택합니다.

사전 훈련된 모델부터 시작하면 **지침 준수, 채팅 및 함수 호출을 배우기 위해 더 많은 훈련이 필요합니다**.

지침 튜닝된 모델부터 시작하면 **모델이 배워야 할 정보의 양을 최소화합니다**.

## LoRA (대규모 언어 모델의 저순위 적응)

LoRA는 **훈련 가능한 매개변수의 수를 크게 줄이는** 인기 있고 가벼운 훈련 기술입니다.

**모델에 더 적은 수의 새로운 가중치를 어댑터로 삽입하여 훈련**하는 방식으로 작동합니다. 이렇게 하면 LoRA를 사용한 훈련이 훨씬 빠르고 메모리 효율적이며 더 작은 모델 가중치(수백 MB)를 생성하여 저장하고 공유하기가 더 쉽습니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/blog_multi-lora-serving_LoRA.gif" alt="LoRA 추론" width="50%"/>

LoRA는 일반적으로 선형 레이어에 초점을 맞춰 Transformer 레이어에 순위 분해 행렬 쌍을 추가하여 작동합니다. 훈련 중에 모델의 나머지 부분은 "고정"하고 새로 추가된 어댑터의 가중치만 업데이트합니다.

이렇게 하면 어댑터의 가중치만 업데이트하면 되므로 **훈련해야 할 매개변수**의 수가 상당히 줄어듭니다.

추론 중에 입력은 어댑터와 기본 모델로 전달되거나 이러한 어댑터 가중치를 기본 모델과 병합하여 추가적인 대기 시간 오버헤드가 발생하지 않습니다.

LoRA는 리소스 요구 사항을 관리 가능한 수준으로 유지하면서 **대규모** 언어 모델을 특정 작업이나 도메인에 적용하는 데 특히 유용합니다. 이는 모델을 훈련하는 데 **필요한** 메모리를 줄이는 데 도움이 됩니다.

LoRA가 작동하는 방식에 대해 더 자세히 알고 싶다면 이 [튜토리얼](https://huggingface.co/learn/nlp-course/chapter11/4?fw=pt)을 확인해야 합니다.

## 함수 호출을 위한 모델 미세 조정

튜토리얼 노트북에 액세스할 수 있습니다 👉 [여기](https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb).

그런 다음 [![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb)를 클릭하여 Colab 노트북에서 실행할 수 있습니다.