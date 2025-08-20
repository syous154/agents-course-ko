# LlamaHub 소개

**LlamaHub은 LlamaIndex 내에서 사용할 수 있는 수백 가지 통합, 에이전트 및 도구의 레지스트리입니다.**

![LlamaHub](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/llama-hub.png)

이 과정에서는 다양한 통합을 사용할 예정이므로, 먼저 LlamaHub과 그것이 어떻게 도움이 되는지 살펴보겠습니다.

필요한 구성 요소의 종속성을 찾고 설치하는 방법을 알아보겠습니다.

## 설치

LlamaIndex 설치 지침은 **LlamaHub([https://llamahub.ai/](https://llamahub.ai/))에 잘 구성된 개요**로 제공됩니다.
처음에는 다소 부담스러울 수 있지만, 대부분의 **설치 명령은 일반적으로 기억하기 쉬운 형식을 따릅니다**:

```bash
pip install llama-index-{component-type}-{framework-name}
```

[Hugging Face 추론 API 통합](https://llamahub.ai/l/llms/llama-index-llms-huggingface-api?from=llms)을 사용하여 LLM 및 임베딩 구성 요소의 종속성을 설치해 보겠습니다.

```bash
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```

## 사용법

설치되면 사용 패턴을 확인할 수 있습니다. 가져오기 경로가 설치 명령을 따른다는 것을 알 수 있습니다!
아래에서 **LLM 구성 요소를 위한 Hugging Face 추론 API 사용 예시**를 볼 수 있습니다.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)

response = llm.complete("Hello, how are you?")
print(response)
# I am good, how can I help you today?
```

이제 필요한 구성 요소에 대한 통합을 찾고, 설치하고, 사용하는 방법을 알게 되었습니다.
**구성 요소에 대해 더 자세히 알아보고** 이를 사용하여 자신만의 에이전트를 구축하는 방법을 살펴보겠습니다.