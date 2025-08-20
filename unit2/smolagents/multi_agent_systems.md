<CourseFloatingBanner
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/multiagent_notebook.ipynb"},
]}
askForHelpUrl="http://hf.co/join/discord" />

# 다중 에이전트 시스템

다중 에이전트 시스템은 **전문 에이전트가 복잡한 작업을 공동으로 수행**할 수 있도록 하여 모듈성, 확장성 및 견고성을 향상시킵니다. 단일 에이전트에 의존하는 대신, 작업은 고유한 기능을 가진 에이전트들 사이에 분산됩니다.

**smolagents**에서는 다양한 에이전트를 결합하여 Python 코드를 생성하고, 외부 도구를 호출하고, 웹 검색을 수행하는 등 다양한 작업을 수행할 수 있습니다. 이러한 에이전트를 오케스트레이션함으로써 강력한 워크플로우를 생성할 수 있습니다.

일반적인 설정에는 다음이 포함될 수 있습니다:
- 작업 위임을 위한 **관리자 에이전트**
- 코드 실행을 위한 **코드 인터프리터 에이전트**
- 정보 검색을 위한 **웹 검색 에이전트**

아래 다이어그램은 **관리자 에이전트**가 **코드 인터프리터 도구**와 **웹 검색 에이전트**를 조정하는 간단한 다중 에이전트 아키텍처를 보여줍니다. 웹 검색 에이전트는 `DuckDuckGoSearchTool` 및 `VisitWebpageTool`과 같은 도구를 활용하여 관련 정보를 수집합니다.

<img src="https://mermaid.ink/img/pako:eNp1kc1qhTAQRl9FUiQb8wIpdNO76eKubrmFks1oRg3VSYgjpYjv3lFL_2hnMWQOJwn5sqgmelRWleUSKLAtFs09jqhtoWuYUFfFAa6QA9QDTnpzamheuhxn8pt40-6l13UtS0ddhtQXj6dbR4XUGQg6zEYasTF393KjeSDGnDJKNxzj8I_7hLW5IOSmP9CH9hv_NL-d94d4DVNg84p1EnK4qlIj5hGClySWbadT-6OdsrL02MI8sFOOVkciw8zx8kaNspxnrJQE0fXKtjBMMs3JA-MpgOQwftIE9Bzj14w-cMznI_39E9Z3p0uFoA?type=png" style='background: white;'>

## 다중 에이전트 시스템 작동 방식

다중 에이전트 시스템은 **오케스트레이터 에이전트**의 조정 하에 여러 전문 에이전트가 함께 작동하는 것으로 구성됩니다. 이 접근 방식은 고유한 역할을 가진 에이전트들 사이에 작업을 분산함으로써 복잡한 워크플로우를 가능하게 합니다.

예를 들어, **다중 에이전트 RAG 시스템**은 다음을 통합할 수 있습니다:
- 인터넷 검색을 위한 **웹 에이전트**
- 지식 베이스에서 정보를 가져오는 **검색기 에이전트**
- 시각 자료 생성을 위한 **이미지 생성 에이전트**

이러한 모든 에이전트는 작업 위임 및 상호 작용을 관리하는 오케스트레이터 하에서 작동합니다.

## 다중 에이전트 계층 구조로 복잡한 작업 해결하기

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/multiagent_notebook.ipynb" target="_blank">이 노트북</a>의 코드를 따라할 수 있습니다.
</Tip>

리셉션이 다가오고 있습니다! 당신의 도움으로 알프레드는 이제 준비를 거의 마쳤습니다.

하지만 이제 문제가 생겼습니다: 배트모빌이 사라졌습니다. 알프레드는 대체품을 찾아야 하고, 그것도 빨리 찾아야 합니다.

다행히 브루스 웨인의 삶에 대한 몇몇 전기 영화가 제작되었으므로, 알프레드는 영화 세트 중 하나에 남겨진 자동차를 얻어 현대 표준에 맞게 재설계할 수 있을 것입니다. 여기에는 물론 완전 자율 주행 옵션도 포함될 것입니다.

하지만 이것은 전 세계의 수많은 촬영 장소 중 어디든 될 수 있습니다.

그래서 알프레드는 당신의 도움이 필요합니다. 이 작업을 해결할 수 있는 에이전트를 구축해 주시겠습니까?

> 👉 전 세계의 모든 배트맨 촬영 장소를 찾고, 그곳으로 보트를 통해 이동하는 시간을 계산하고, 보트 이동 시간에 따라 색상이 달라지는 지도로 나타내십시오. 또한 동일한 보트 이동 시간을 가진 일부 슈퍼카 공장도 나타내십시오.

이것을 만들어 봅시다!

이 예시에는 몇 가지 추가 패키지가 필요하므로 먼저 설치해 봅시다:

```bash
pip install 'smolagents[litellm]' plotly geopandas shapely kaleido -q
```

### 먼저 화물기 이동 시간을 얻기 위한 도구를 만듭니다.

```python
import math
from typing import Optional, Tuple

from smolagents import tool


@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # 화물기의 평균 속도
) -> float:
    """
    대원 거리(great-circle distance)를 사용하여 지구상의 두 지점 사이의 화물기 이동 시간을 계산합니다.

    Args:
        origin_coords: 시작 지점의 (위도, 경도) 튜플
        destination_coords: 목적지의 (위도, 경도) 튜플
        cruising_speed_kmh: 선택 사항인 순항 속도(km/h) (일반적인 화물기의 경우 기본값 750km/h)

    Returns:
        float: 예상 이동 시간(시간)

    Example:
        >>> # 시카고 (북위 41.8781°, 서경 87.6298°)에서 시드니 (남위 33.8688°, 동경 151.2093°)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # 좌표 추출
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # 지구 반지름(킬로미터)
    EARTH_RADIUS_KM = 6371.0

    # 하버사인 공식을 사용하여 대원 거리 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # 비직선 경로 및 항공 교통 관제를 고려하여 10% 추가
    actual_distance = distance * 1.1

    # 비행 시간 계산
    # 이착륙 절차에 1시간 추가
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # 결과 형식 지정
    return round(flight_time, 2)


print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))
```

### 에이전트 설정

모델 제공업체로는 Hugging Face Hub의 새로운 [추론 제공업체](https://huggingface.co/blog/inference-providers) 중 하나인 Together AI를 사용합니다!

GoogleSearchTool은 [Serper API](https://serper.dev)를 사용하여 웹을 검색하므로, `SERPAPI_API_KEY` 환경 변수를 설정하고 `provider="serpapi"`를 전달하거나 `SERPER_API_KEY`를 설정하고 `provider=serper`를 전달해야 합니다.

Serp API 제공업체를 설정하지 않은 경우 `DuckDuckGoSearchTool`을 사용할 수 있지만, 속도 제한이 있다는 점에 유의하십시오.

```python
import os
from PIL import Image
from smolagents import CodeAgent, GoogleSearchTool, InferenceClientModel, VisitWebpageTool

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")
```

간단한 보고서를 제공하기 위한 기준선으로 간단한 에이전트를 생성하는 것으로 시작할 수 있습니다.

```python
task = """전 세계의 모든 배트맨 촬영 장소를 찾고, 이곳(고담, 북위 40.7128°, 서경 74.0060°)으로 화물기를 통해 이동하는 시간을 계산하여 pandas 데이터프레임으로 반환해 주세요.
또한 동일한 화물기 이동 시간을 가진 일부 슈퍼카 공장도 알려주세요."""
```

```python
agent = CodeAgent(
    model=model,
    tools=[GoogleSearchTool("serper"), VisitWebpageTool(), calculate_cargo_travel_time],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)
```

```python
result = agent.run(task)
```

```python
result
```

우리 경우, 다음과 같은 출력을 생성합니다:

```python
|  | Location                                             | Travel Time to Gotham (hours) |
|--|------------------------------------------------------|------------------------------|
| 0  | Necropolis Cemetery, Glasgow, Scotland, UK         | 8.60                         |
| 1  | St. George's Hall, Liverpool, England, UK         | 8.81                         |
| 2  | Two Temple Place, London, England, UK             | 9.17                         |
| 3  | Wollaton Hall, Nottingham, England, UK           | 9.00                         |
| 4  | Knebworth House, Knebworth, Hertfordshire, UK    | 9.15                         |
| 5  | Acton Lane Power Station, Acton Lane, Acton, UK  | 9.16                         |
| 6  | Queensboro Bridge, New York City, USA            | 1.01                         |
| 7  | Wall Street, New York City, USA                  | 1.00                         |
| 8  | Mehrangarh Fort, Jodhpur, Rajasthan, India       | 18.34                        |
| 9  | Turda Gorge, Turda, Romania                      | 11.89                        |
| 10 | Chicago, USA                                     | 2.68                         |
| 11 | Hong Kong, China                                 | 19.99                        |
| 12 | Cardington Studios, Northamptonshire, UK        | 9.10                         |
| 13 | Warner Bros. Leavesden Studios, Hertfordshire, UK | 9.13                         |
| 14 | Westwood, Los Angeles, CA, USA                  | 6.79                         |
| 15 | Woking, UK (McLaren)                             | 9.13                         |
```

전용 계획 단계를 추가하고 더 많은 프롬프트를 추가하여 이를 약간 개선할 수 있습니다.

계획 단계는 에이전트가 미리 생각하고 다음 단계를 계획할 수 있도록 하며, 이는 더 복잡한 작업에 유용할 수 있습니다.

```python
agent.planning_interval = 4

detailed_report = agent.run(f"""
당신은 전문 분석가입니다. 많은 웹사이트를 방문한 후 포괄적인 보고서를 작성합니다.
for 루프에서 한 번에 여러 쿼리를 검색하는 것을 주저하지 마십시오.
찾은 각 데이터 포인트에 대해 원본 URL을 방문하여 숫자를 확인하십시오.

{task}
""")

print(detailed_report)
```

```python
detailed_report
```

우리 경우, 다음과 같은 출력을 생성합니다:

```python
|  | Location                                         | Travel Time (hours) |
|--|--------------------------------------------------|---------------------|
| 0  | Bridge of Sighs, Glasgow Necropolis, Glasgow, UK | 8.6                 |
| 1  | Wishart Street, Glasgow, Scotland, UK         | 8.6                 |
```


이러한 빠른 변경 덕분에, 에이전트에 자세한 프롬프트를 제공하고 계획 기능을 부여함으로써 훨씬 더 간결한 보고서를 얻을 수 있었습니다!

모델의 컨텍스트 창이 빠르게 채워지고 있습니다. 따라서 **에이전트에게 자세한 검색 결과를 다른 것과 결합하도록 요청하면 속도가 느려지고 토큰 및 비용이 빠르게 증가할 것입니다**.

➡️ 시스템 구조를 개선해야 합니다.

### ✌️ 두 에이전트 간에 작업 분할

다중 에이전트 구조는 서로 다른 하위 작업 간에 메모리를 분리할 수 있도록 하며, 두 가지 큰 이점이 있습니다:
- 각 에이전트는 핵심 작업에 더 집중하여 성능이 향상됩니다.
- 메모리를 분리하면 각 단계에서 입력 토큰 수가 줄어들어 지연 시간과 비용이 절감됩니다.

전용 웹 검색 에이전트를 다른 에이전트가 관리하는 팀을 만들어 봅시다.

관리자 에이전트는 최종 보고서를 작성하기 위한 플로팅 기능을 가져야 합니다. 따라서 `plotly` 및 공간 플로팅을 위한 `geopandas` + `shapely`를 포함한 추가 임포트에 대한 액세스를 제공합시다.

```python
model = InferenceClientModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=8096
)

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="정보를 찾기 위해 웹을 탐색합니다",
    verbosity_level=0,
    max_steps=10,
)
```

관리자 에이전트는 상당한 정신적 노력을 기울여야 할 것입니다.

따라서 더 강력한 모델인 [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)을 제공하고, `planning_interval`을 추가합니다.

```python
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel


def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "saved_map.png에 플롯을 저장했는지 확인하십시오!"
    image = Image.open(filepath)
    prompt = (
        f"다음은 사용자가 제공한 작업과 에이전트 단계입니다: {agent_memory.get_succinct_steps()}. 이제 만들어진 플롯입니다."
        "추론 프로세스와 플롯이 올바른지 확인하십시오: 주어진 작업을 올바르게 답변합니까?"
        "먼저 예/아니오 이유를 나열한 다음 최종 결정: 만족스러우면 대문자로 PASS, 그렇지 않으면 FAIL을 작성하십시오."
        "가혹하게 판단하지 마십시오: 플롯이 대부분 작업을 해결한다면 통과해야 합니다."
        "통과하려면 px.scatter_map을 사용하여 플롯을 만들어야 하며 다른 방법(scatter_map이 더 보기 좋음)은 사용하지 않아야 합니다."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("피드백: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


manager_agent = CodeAgent(
    model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)
```

이 팀이 어떻게 생겼는지 살펴봅시다:

```python
manager_agent.visualize()
```

이것은 다음과 같은 것을 생성하여 에이전트와 사용된 도구 간의 구조와 관계를 이해하는 데 도움이 됩니다:

```python
CodeAgent | deepseek-ai/DeepSeek-R1
├── ✅ 승인된 임포트: ['geopandas', 'plotly', 'shapely', 'json', 'pandas', 'numpy']
├── 🛠️ 도구:
│   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│   ┃ 이름                        ┃ 설명                                  ┃ 인수                                  ┃
│   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   │ calculate_cargo_travel_time │ 대원 거리를 사용하여 지구상의 두 지점 │ origin_coords (`array`): 시작 지점의  │
│   │                             │ 사이의 화물기 이동 시간을 계산합니다. │ (위도, 경도) 튜플                     │
│   │                             │                                       │ destination_coords (`array`): 목적지의│
│   │                             │                                       │ (위도, 경도) 튜플                     │
│   │                             │                                       │ cruising_speed_kmh (`number`):        │
│   │                             │                                       │ 선택 사항인 순항 속도(km/h)           │
│   │                             │                                       │ (일반적인 화물기의 경우 기본값 750km/h) │
│   │ final_answer                │ 주어진 문제에 대한 최종 답변을 제공합니다. │ answer (`any`): 문제에 대한 최종 답변 │
│   └─────────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘
└── 🤖 관리되는 에이전트:
    └── web_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
        ├── ✅ 승인된 임포트: []
        ├── 📝 설명: 정보를 찾기 위해 웹을 탐색합니다.
        └── 🛠️ 도구:
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ 이름                        ┃ 설명                              ┃ 인수                              ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ web_search                  │ 쿼리에 대한 구글 웹 검색을 수행한 │ query (`string`): 수행할 검색 쿼리 │
            │                             │ 다음 상위 검색 결과 문자열을 반환합니다. │ filter_year (`integer`):          │
            │                             │                                   │ 선택적으로 결과를 특정 연도로 제한합니다. │
            │ visit_webpage               │ 주어진 URL의 웹 페이지를 방문하고 │ url (`string`): 방문할 웹 페이지의 URL │
            │                             │ 내용을 마크다운 문자열로 읽습니다. │                                   │
            │                             │ 웹 페이지를 탐색하는 데 사용합니다. │                                   │
            │ calculate_cargo_travel_time │ 대원 거리를 사용하여 지구상의 두 지점 │ origin_coords (`array`): 시작 지점의  │
            │                             │ 사이의 화물기 이동 시간을 계산합니다. │ (위도, 경도) 튜플                     │
            │                             │                                   │ destination_coords (`array`):     │
            │                             │                                   │ 목적지의 (위도, 경도) 튜플            │
            │                             │                                   │ cruising_speed_kmh (`number`):    │
            │                             │                                   │ 선택 사항인 순항 속도(km/h)           │
            │                             │                                   │ (일반적인 화물기의 경우 기본값 750km/h) │
            │ final_answer                │ 주어진 문제에 대한 최종 답변을 제공합니다. │ answer (`any`): 문제에 대한 최종 답변 │
            └─────────────────────────────┴───────────────────────────────────┴───────────────────────────────────┘
```

```python
manager_agent.run("""
전 세계의 모든 배트맨 촬영 장소를 찾고, 이곳(고담, 북위 40.7128°, 서경 74.0060°)으로 화물기를 통해 이동하는 시간을 계산해 주세요.
또한 동일한 화물기 이동 시간을 가진 일부 슈퍼카 공장도 알려주세요. 총 6개 이상의 지점이 필요합니다.
이를 세계 공간 지도로 나타내고, 위치는 이동 시간에 따라 색상이 달라지는 산점도로 나타내고, saved_map.png로 저장해 주세요!

다음은 지도를 플로팅하고 반환하는 방법의 예시입니다:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

코드를 사용하여 문자열을 처리하려고 하지 마십시오: 읽을 문자열이 있으면 그냥 인쇄하면 됩니다.
""")
```

당신의 실행에서는 어떻게 진행되었는지 모르겠지만, 제 실행에서는 관리자 에이전트가 `1. 배트맨 촬영 장소 검색`, `2. 슈퍼카 공장 찾기`로 웹 에이전트에 주어진 작업을 능숙하게 분할한 다음, 목록을 집계하고 지도를 플로팅했습니다.

에이전트 상태에서 직접 검사하여 지도가 어떻게 생겼는지 살펴봅시다:

```python
manager_agent.python_executor.state["fig"]
```

이것은 지도를 출력합니다:

![Multiagent system example output map](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/smolagents/output_map.png)

## 자료

- [다중 에이전트 시스템](https://huggingface.co/docs/smolagents/main/en/examples/multiagents) – 다중 에이전트 시스템 개요.
- [에이전트 RAG란?](https://weaviate.io/blog/what-is-agentic-rag) – 에이전트 RAG 소개.
- [다중 에이전트 RAG 시스템 🤖🤝🤖 레시피](https://huggingface.co/learn/cookbook/multiagent_rag_system) – 다중 에이전트 RAG 시스템 구축을 위한 단계별 가이드.