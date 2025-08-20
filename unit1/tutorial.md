# smolagents를 사용하여 첫 번째 에이전트 만들기

지난 섹션에서는 Python 코드를 사용하여 처음부터 에이전트를 만드는 방법을 배웠고, **그 과정이 얼마나 지루할 수 있는지 확인했습니다.** 다행히도 많은 에이전트 라이브러리는 **많은 번거로운 작업을 대신 처리하여** 이 작업을 단순화합니다.

이 튜토리얼에서는 이미지 생성, 웹 검색, 시간대 확인 등 다양한 작업을 수행할 수 있는 **첫 번째 에이전트를 만들게 됩니다!**

또한 에이전트를 **Hugging Face Space에 게시하여 친구 및 동료와 공유할 수 있습니다.**

시작해 봅시다!

## smolagents란 무엇인가요?

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/smolagents.png" alt="smolagents"/>

이 에이전트를 만들기 위해 **에이전트를 쉽게 개발할 수 있는 프레임워크를 제공하는** `smolagents` 라이브러리를 사용할 것입니다.

이 경량 라이브러리는 단순성을 위해 설계되었지만, 에이전트 구축의 복잡성을 상당 부분 추상화하여 에이전트의 동작 설계에 집중할 수 있도록 합니다.

다음 유닛에서 smolagents에 대해 더 자세히 알아볼 것입니다. 그동안 이 [블로그 게시물](https://huggingface.co/blog/smolagents) 또는 라이브러리의 [GitHub 저장소](https://github.com/huggingface/smolagents)를 확인할 수도 있습니다.

요컨대, `smolagents`는 코드 블록을 통해 **"액션"**을 수행한 다음 코드를 실행하여 결과를 **"관찰"**하는 일종의 에이전트인 **codeAgent**에 중점을 둔 라이브러리입니다.

다음은 우리가 만들 예시입니다!

에이전트에 **이미지 생성 도구**를 제공하고 고양이 이미지를 생성하도록 요청했습니다.

`smolagents` 내부의 에이전트는 **이전에 구축한 사용자 지정 에이전트와 동일한 동작**을 가질 것입니다. 즉, 최종 답변에 도달할 때까지 **생각하고, 행동하고, 관찰하는 주기를 반복합니다.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/PQDKcWiuln4?si=ysSTDZoi8y55FVvA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

흥미롭죠?

## 에이전트를 만들어 봅시다!

시작하려면 이 Space를 복제하세요: <a href="https://huggingface.co/spaces/agents-course/First_agent_template" target="_blank">https://huggingface.co/spaces/agents-course/First_agent_template</a>
> 이 템플릿을 제공해 주신 <a href="https://huggingface.co/m-ric" target="_blank">Aymeric</a>에게 감사드립니다! 🙌

이 Space를 복제한다는 것은 **자신의 프로필에 로컬 복사본을 만드는 것**을 의미합니다.

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/duplicate-space.gif" alt="Duplicate"/>

Space를 복제한 후 에이전트가 모델 API에 액세스할 수 있도록 Hugging Face API 토큰을 추가해야 합니다.

1. 먼저, 아직 Hugging Face 토큰이 없다면 [https://hf.co/settings/tokens](https://huggingface.co/settings/tokens)에서 추론 권한이 있는 토큰을 받으세요.
2. 복제한 Space로 이동하여 **Settings** 탭을 클릭합니다.
3. **Variables and Secrets** 섹션으로 스크롤하여 **New Secret**을 클릭합니다.
4. `HF_TOKEN`이라는 이름으로 시크릿을 생성하고 토큰을 값으로 붙여넣습니다.
5. **Save**를 클릭하여 토큰을 안전하게 저장합니다.

이 강의 전체에서 수정해야 할 유일한 파일은 (현재 불완전한) **"app.py"**입니다. 여기에서 [템플릿의 원본 파일](https://huggingface.co/spaces/agents-course/First_agent_template/blob/main/app.py)을 볼 수 있습니다. 자신의 파일을 찾으려면 Space의 복사본으로 이동한 다음 `Files` 탭을 클릭하고 디렉토리 목록에서 `app.py`를 클릭합니다.

코드를 함께 분석해 봅시다.

- 파일은 몇 가지 간단하지만 필요한 라이브러리 임포트로 시작합니다.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
```

앞서 설명했듯이, **smolagents**의 **CodeAgent** 클래스를 직접 사용할 것입니다.

### 도구

이제 도구에 대해 알아봅시다! 도구에 대한 복습이 필요하다면 주저하지 말고 코스의 [도구](tools) 섹션으로 돌아가세요.

```python
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # 반환 유형을 지정하는 것이 중요합니다.
    # 도구 설명/인수 설명에 이 형식을 유지하되, 도구를 자유롭게 수정하세요.
    """아직 아무것도 하지 않는 도구
    Args:
        arg1: 첫 번째 인수
        arg2: 두 번째 인수
    """
    return "어떤 마법을 만들 건가요?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """지정된 시간대의 현재 현지 시간을 가져오는 도구입니다.
    Args:
        timezone: 유효한 시간대를 나타내는 문자열 (예: 'America/New_York').
    """
    try:
        # 시간대 객체 생성
        tz = pytz.timezone(timezone)
        # 해당 시간대의 현재 시간 가져오기
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone}의 현재 현지 시간은: {local_time}입니다."
    except Exception as e:
        return f"시간대 '{timezone}'의 시간을 가져오는 중 오류 발생: {str(e)}"
```

도구는 이 섹션에서 여러분이 만들도록 권장하는 것입니다! 두 가지 예시를 제공합니다.

1. 유용한 것을 만들기 위해 수정할 수 있는 **작동하지 않는 더미 도구**.
2. 전 세계 어딘가의 현재 시간을 가져오는 **실제로 작동하는 도구**.

도구를 정의하려면 다음이 중요합니다.

1. `get_current_time_in_timezone(timezone: str) -> str:`와 같이 함수에 대한 입력 및 출력 유형을 제공합니다.
2. **잘 형식화된 독스트링**. `smolagents`는 모든 인수에 **독스트링에 텍스트 설명**이 있어야 한다고 예상합니다.

### 에이전트

LLM 엔진으로 [`Qwen/Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)를 사용합니다. 이것은 서버리스 API를 통해 액세스할 수 있는 매우 유능한 모델입니다.

```python
final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
# CodeAgent를 생성합니다.
agent = CodeAgent(
    model=model,
    tools=[final_answer], # 여기에 도구를 추가하세요 (final_answer를 제거하지 마세요)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
```

이 에이전트는 **InferenceClientModel** 클래스 뒤에 있는 이전 섹션에서 보았던 `InferenceClient`를 여전히 사용합니다!

유닛 2에서 프레임워크를 소개할 때 더 심층적인 예시를 제공할 것입니다. 지금은 에이전트의 `tools` 매개변수를 사용하여 **도구 목록에 새 도구를 추가하는 것**에 집중해야 합니다.

예를 들어, 코드의 첫 줄에서 임포트된 `DuckDuckGoSearchTool`을 사용하거나, 코드에서 나중에 허브에서 로드되는 `image_generation_tool`을 검토할 수 있습니다.

**도구를 추가하면 에이전트에 새로운 기능이 부여됩니다.** 여기서 창의력을 발휘해 보세요!

### 시스템 프롬프트

에이전트의 시스템 프롬프트는 별도의 `prompts.yaml` 파일에 저장됩니다. 이 파일에는 에이전트의 동작을 안내하는 미리 정의된 지침이 포함되어 있습니다.

프롬프트를 YAML 파일에 저장하면 다른 에이전트나 사용 사례에서 쉽게 사용자 지정하고 재사용할 수 있습니다.

[Space의 파일 구조](https://huggingface.co/spaces/agents-course/First_agent_template/tree/main)를 확인하여 `prompts.yaml` 파일이 어디에 있고 프로젝트 내에서 어떻게 구성되어 있는지 확인할 수 있습니다.

전체 "app.py":

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# 아래는 아무것도 하지 않는 도구의 예시입니다. 여러분의 창의력으로 저희를 놀라게 해주세요!
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # 반환 유형을 지정하는 것이 중요합니다.
    # 도구 설명/인수 설명에 이 형식을 유지하되, 도구를 자유롭게 수정하세요.
    """아직 아무것도 하지 않는 도구
    Args:
        arg1: 첫 번째 인수
        arg2: 두 번째 인수
    """
    return "어떤 마법을 만들 건가요?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """지정된 시간대의 현재 현지 시간을 가져오는 도구입니다.
    Args:
        timezone: 유효한 시간대를 나타내는 문자열 (예: 'America/New_York').
    """
    try:
        # 시간대 객체 생성
        tz = pytz.timezone(timezone)
        # 해당 시간대의 현재 시간 가져오기
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone}의 현재 현지 시간은: {local_time}입니다."
    except Exception as e:
        return f"시간대 '{timezone}'의 시간을 가져오는 중 오류 발생: {str(e)}"


final_answer = FinalAnswerTool()
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)


# Hub에서 도구 임포트
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# prompt.yaml 파일에서 시스템 프롬프트 로드
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer], # 여기에 도구를 추가하세요 (final_answer를 제거하지 마세요)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates # 시스템 프롬프트를 CodeAgent에 전달
)

GradioUI(agent).launch()
```

여러분의 **목표**는 Space와 에이전트에 익숙해지는 것입니다.

현재 템플릿의 에이전트는 **어떤 도구도 사용하지 않으므로, 미리 만들어진 도구 중 일부를 제공하거나 직접 새로운 도구를 만들어 보세요!**

디스코드 채널 **#agents-course-showcase**에서 여러분의 멋진 에이전트 결과물을 간절히 기다리고 있습니다!

---
축하합니다. 첫 번째 에이전트를 만들었습니다! 친구 및 동료와 공유하는 것을 주저하지 마세요.

첫 시도이므로 약간의 버그가 있거나 느리더라도 완벽하게 정상입니다. 다음 유닛에서는 더 나은 에이전트를 만드는 방법을 배울 것입니다.

가장 좋은 학습 방법은 시도하는 것이므로, 업데이트하고, 더 많은 도구를 추가하고, 다른 모델로 시도하는 것을 주저하지 마세요.

다음 섹션에서는 최종 퀴즈를 풀고 수료증을 받게 됩니다!