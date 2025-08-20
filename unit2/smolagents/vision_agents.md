<CourseFloatingBanner
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/#fileId=https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/vision_agents.ipynb"},
]}
askForHelpUrl="http://hf.co/join/discord" />

# smolagents를 이용한 비전 에이전트

<Tip warning={true}>
이 섹션의 예제는 강력한 VLM 모델에 대한 접근이 필요합니다. 저희는 GPT-4o API를 사용하여 테스트했습니다.
하지만, <a href="./why_use_smolagents">smolagents를 사용하는 이유</a>에서는 smolagents와 Hugging Face가 지원하는 대체 솔루션에 대해 설명합니다. 다른 옵션을 탐색하고 싶다면 해당 섹션을 확인하십시오.
</Tip>

시각적 기능을 갖춘 에이전트를 강화하는 것은 텍스트 처리를 넘어선 작업을 해결하는 데 중요합니다. 웹 브라우징이나 문서 이해와 같은 많은 실제 문제는 풍부한 시각적 콘텐츠를 분석해야 합니다. 다행히 `smolagents`는 VLM(Vision-Language Model)에 대한 내장 지원을 제공하여 에이전트가 이미지를 효과적으로 처리하고 해석할 수 있도록 합니다.

이 예제에서는 웨인 저택의 집사인 알프레드가 파티에 참석하는 손님들의 신원을 확인하는 임무를 맡았다고 상상해 보세요. 알프레드는 도착하는 모든 사람에게 익숙하지 않을 수 있습니다. 그를 돕기 위해 VLM을 사용하여 손님의 외모에 대한 시각적 정보를 검색하여 신원을 확인하는 에이전트를 사용할 수 있습니다. 이를 통해 알프레드는 누가 입장할 수 있는지에 대해 정보에 입각한 결정을 내릴 수 있습니다. 이 예제를 만들어 봅시다!

## 에이전트 실행 시작 시 이미지 제공

<Tip>
Google Colab을 사용하여 실행할 수 있는 <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/vision_agents.ipynb" target="_blank">이 노트북</a>의 코드를 따라할 수 있습니다.
</Tip>

이 접근 방식에서는 이미지가 시작 시 에이전트에 전달되고 작업 프롬프트와 함께 `task_images`로 저장됩니다. 그런 다음 에이전트는 실행 내내 이 이미지를 처리합니다.

알프레드가 파티에 참석하는 슈퍼히어로들의 신원을 확인하려는 경우를 생각해 보세요. 그는 이미 이전 파티에서 손님들의 이름이 포함된 이미지 데이터셋을 가지고 있습니다. 새로운 방문자의 이미지가 주어지면 에이전트는 기존 데이터셋과 비교하여 입장 여부를 결정할 수 있습니다.

이 경우 손님이 입장하려고 하는데, 알프레드는 이 방문자가 원더우먼을 사칭하는 조커일 수 있다고 의심합니다. 알프레드는 원치 않는 사람이 입장하는 것을 막기 위해 신원을 확인해야 합니다.

예제를 만들어 봅시다. 먼저 이미지를 로드합니다. 이 경우 예제를 최소화하기 위해 위키백과 이미지를 사용하지만, 가능한 사용 사례를 상상해 보세요!

```python
from PIL import Image
import requests
from io import BytesIO

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # 조커 이미지
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # 조커 이미지
]

images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url,headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)
```

이제 이미지가 있으므로 에이전트는 한 손님이 실제로 슈퍼히어로(원더우먼)인지 악당(조커)인지 알려줄 것입니다.

```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(model_id="gpt-4o")

# 에이전트 인스턴스화
agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=20,
    verbosity_level=2
)

response = agent.run(
    """
    이 사진에 있는 만화 캐릭터가 입고 있는 의상과 화장을 설명하고 그 설명을 반환하세요.
    손님이 조커인지 원더우먼인지 알려주세요.
    """,
    images=images
)
```

제 실행의 경우 출력은 다음과 같습니다. 하지만 이미 논의했듯이 사용자 환경에 따라 다를 수 있습니다.

```python
    {
        'Costume and Makeup - First Image': (
            '자주색 코트와 겨자색 셔츠 위에 자주색 실크 같은 크라바트 또는 넥타이.',
            '과장된 특징, 짙은 눈썹, 파란색 눈 화장, 넓은 미소를 짓는 붉은 입술을 가진 흰색 얼굴 페인트.'
        ),
        'Costume and Makeup - Second Image': (
            '라펠에 꽃이 달린 어두운 양복을 입고 트럼프 카드를 들고 있음.',
            '창백한 피부, 녹색 머리, 과장된 미소를 짓는 매우 붉은 입술.'
        ),
        'Character Identity': '이 캐릭터는 만화책 미디어에 나오는 조커의 알려진 묘사와 유사합니다.'
    }
```

이 경우 출력은 그 사람이 다른 사람을 사칭하고 있음을 보여주므로 조커가 파티에 입장하는 것을 막을 수 있습니다!

## 동적 검색을 통한 이미지 제공

<Tip>
<a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/vision_web_browser.py" target="_blank">이 Python 파일</a>의 코드를 따라할 수 있습니다.
</Tip>

이전 접근 방식은 유용하며 많은 잠재적 사용 사례를 가지고 있습니다. 그러나 손님이 데이터베이스에 없는 상황에서는 손님을 식별하는 다른 방법을 탐색해야 합니다. 한 가지 가능한 해결책은 웹을 검색하여 외부 소스에서 이미지와 정보를 동적으로 검색하는 것입니다.

이 접근 방식에서는 이미지가 실행 중에 에이전트의 메모리에 동적으로 추가됩니다. 아시다시피 `smolagents`의 에이전트는 ReAct 프레임워크의 추상화인 `MultiStepAgent` 클래스를 기반으로 합니다. 이 클래스는 다양한 변수와 지식이 다른 단계에서 기록되는 구조화된 주기에서 작동합니다.

1. **SystemPromptStep:** 시스템 프롬프트를 저장합니다.
2. **TaskStep:** 사용자 쿼리 및 제공된 입력을 기록합니다.
3. **ActionStep:** 에이전트의 작업 및 결과에서 로그를 캡처합니다.

이 구조화된 접근 방식을 통해 에이전트는 시각적 정보를 동적으로 통합하고 진화하는 작업에 적응적으로 응답할 수 있습니다. 아래는 이미 본 다이어그램으로, 동적 워크플로 프로세스와 에이전트 수명 주기 내에서 다양한 단계가 통합되는 방식을 보여줍니다. 브라우징할 때 에이전트는 스크린샷을 찍고 `ActionStep`에 `observation_images`로 저장할 수 있습니다.

![동적 이미지 검색](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/smolagents-can-see/diagram_adding_vlms_smolagents.png)

이제 필요성을 이해했으므로 완전한 예제를 만들어 봅시다. 이 경우 알프레드는 손님 확인 프로세스를 완전히 제어하기를 원하므로 세부 정보를 브라우징하는 것이 실행 가능한 솔루션이 됩니다. 이 예제를 완료하려면 에이전트에 대한 새로운 도구 세트가 필요합니다. 또한 브라우저 자동화 도구인 Selenium과 Helium을 사용할 것입니다. 이를 통해 잠재적 손님에 대한 세부 정보를 검색하고 확인 정보를 검색하는 웹을 탐색하는 에이전트를 구축할 수 있습니다. 필요한 도구를 설치해 봅시다.

```bash
pip install "smolagents[all]" helium selenium python-dotenv
```

`search_item_ctrl_f`, `go_back`, `close_popups`와 같이 브라우징을 위해 특별히 설계된 에이전트 도구 세트가 필요합니다. 이러한 도구를 통해 에이전트는 웹을 탐색하는 사람처럼 행동할 수 있습니다.

```python
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Ctrl + F를 통해 현재 페이지에서 텍스트를 검색하고 n번째 발생으로 이동합니다.
    Args:
        text: 검색할 텍스트
        nth_result: 이동할 발생 횟수 (기본값: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"일치하는 항목 n°{nth_result}를 찾을 수 없습니다. ({len(elements)}개만 찾음)")
    result = f"'{text}'에 대해 {len(elements)}개의 일치하는 항목을 찾았습니다."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"{len(elements)}개 중 {nth_result}번째 요소에 초점을 맞췄습니다."
    return result


@tool
def go_back() -> None:
    """
    이전 페이지로 돌아갑니다.
    """
    driver.back()


@tool
def close_popups() -> str:
    """
    페이지에 표시되는 모든 모달 또는 팝업을 닫습니다. 팝업 창을 닫으려면 이것을 사용하십시오! 이것은 쿠키 동의 배너에서는 작동하지 않습니다.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
```

또한 스크린샷을 저장하는 기능도 필요합니다. 이는 VLM 에이전트가 작업을 완료하는 데 사용하는 필수적인 부분이 될 것입니다. 이 기능은 스크린샷을 캡처하여 `step_log.observations_images = [image.copy()]`에 저장하여 에이전트가 탐색하는 동안 이미지를 동적으로 저장하고 처리할 수 있도록 합니다.

```python
def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # 스크린샷을 찍기 전에 JavaScript 애니메이션이 발생하도록 합니다.
    driver = helium.get_driver()
    current_step = step_log.step_number
    if driver is not None:
        for step_logs in agent.logs:  # 효율적인 처리를 위해 로그에서 이전 스크린샷을 제거합니다.
            if isinstance(step_log, ActionStep) and step_log.step_number <= current_step - 2:
                step_logs.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"브라우저 스크린샷 캡처: {image.size} 픽셀")
        step_log.observations_images = [image.copy()]  # 지속성을 보장하기 위해 복사본을 만듭니다. 중요합니다!

    # 현재 URL로 관찰 업데이트
    url_info = f"현재 URL: {driver.current_url}"
    step_log.observations = url_info if step_logs.observations is None else step_log.observations + "\n" + url_info
    return
```

이 함수는 에이전트의 실행 중 각 단계가 끝날 때 트리거되므로 `step_callback`으로 에이전트에 전달됩니다. 이를 통해 에이전트는 프로세스 전반에 걸쳐 스크린샷을 동적으로 캡처하고 저장할 수 있습니다.

이제 웹 브라우징을 위한 비전 에이전트를 생성하고, 우리가 만든 도구와 웹을 탐색하기 위한 `DuckDuckGoSearchTool`을 제공할 수 있습니다. 이 도구는 에이전트가 시각적 단서를 기반으로 손님의 신원을 확인하는 데 필요한 정보를 검색하는 데 도움이 될 것입니다.

```python
from smolagents import CodeAgent, OpenAIServerModel, DuckDuckGoSearchTool
model = OpenAIServerModel(model_id="gpt-4o")

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)
```

이제 알프레드는 손님들의 신원을 확인하고 파티에 입장시킬지 여부에 대해 정보에 입각한 결정을 내릴 준비가 되었습니다.

```python
agent.run("""
저는 웨인 저택의 집사 알프레드입니다. 파티 손님들의 신원을 확인하는 일을 담당하고 있습니다. 한 슈퍼히어로가 원더우먼이라고 주장하며 입구에 도착했지만, 그녀가 정말로 원더우먼인지 확인해야 합니다.

원더우먼의 이미지를 검색하고 해당 이미지를 기반으로 자세한 시각적 설명을 생성해 주세요. 또한, 위키백과로 이동하여 그녀의 외모에 대한 주요 세부 정보를 수집해 주세요. 이 정보를 통해 그녀의 이벤트 입장을 허용할지 여부를 결정할 수 있습니다.
""" + helium_instructions)
```

`helium_instructions`를 작업의 일부로 포함하는 것을 볼 수 있습니다. 이 특별한 프롬프트는 에이전트의 탐색을 제어하여 웹을 브라우징하는 동안 올바른 단계를 따르도록 하는 것을 목표로 합니다.

아래 비디오에서 이것이 어떻게 작동하는지 봅시다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/rObJel7-OLc?si=TnNwQ8rqXqun_pqE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

최종 출력은 다음과 같습니다.

```python
Final answer: 원더우먼은 일반적으로 빨간색과 금색 코르셋, 흰색 별이 있는 파란색 반바지 또는 치마, 금색 티아라, 은색 팔찌, 금색 진실의 올가미를 착용하는 것으로 묘사됩니다. 그녀는 테미스키라의 다이애나 공주이며, 인간 세계에서는 다이애나 프린스로 알려져 있습니다.
```

이 모든 것을 통해 우리는 파티를 위한 신원 확인 장치를 성공적으로 만들었습니다! 알프레드는 이제 올바른 손님만 문을 통과할 수 있도록 하는 데 필요한 도구를 갖게 되었습니다. 웨인 저택에서 즐거운 시간을 보낼 모든 준비가 완료되었습니다!

## 추가 자료

- [smolagents에 시각을 부여했습니다](https://huggingface.co/blog/smolagents-can-see) - 비전 에이전트 기능에 대한 블로그.
- [에이전트를 이용한 웹 브라우저 자동화 🤖🌐](https://huggingface.co/docs/smolagents/examples/web_browser) - 비전 에이전트를 이용한 웹 브라우징 예제.
- [웹 브라우저 비전 에이전트 예제](https://github.com/huggingface/smolagents/blob/main/src/smolagents/vision_web_browser.py) - 비전 에이전트를 이용한 웹 브라우징 예제.
