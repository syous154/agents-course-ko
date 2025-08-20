# 포켓몬 배틀 에이전트 출시

이제 배틀 시간입니다! ⚡️

## **스트림 에이전트와 배틀하세요!**

자신만의 에이전트를 만들고 싶지 않고 포켓몬에서 에이전트의 배틀 잠재력에 대해 궁금하다면 [twitch](https://www.twitch.tv/jofthomas)에서 자동화된 라이브 스트림을 호스팅하고 있습니다.

<iframe
	src="https://jofthomas-twitch-streaming.hf.space"
	frameborder="0"
	width="1200"
	height="600"
></iframe>


스트림에서 에이전트와 배틀하려면 다음을 수행할 수 있습니다.

지침:
1. **포켓몬 쇼다운 스페이스**로 이동합니다: [여기 링크](https://huggingface.co/spaces/Jofthomas/Pokemon_showdown)
2. **이름 선택**(오른쪽 상단 모서리).
3. **현재 에이전트의 사용자 이름**을 찾습니다. 확인:
    * **스트림 디스플레이**: [여기 링크](https://www.twitch.tv/jofthomas)
4. 쇼다운 스페이스에서 해당 사용자 이름을 **검색**하고 **배틀 초대 보내기**.

*주의:* 한 번에 한 명의 에이전트만 온라인 상태입니다! 올바른 이름을 가지고 있는지 확인하십시오.



## 포켓몬 배틀 에이전트 챌린저

지난 섹션에서 자신만의 포켓몬 배틀 에이전트를 만들었다면 아마도 궁금할 것입니다. **다른 사람들과 어떻게 테스트할 수 있을까요?** 알아봅시다!

이 목적을 위해 전용 [Hugging Face 스페이스](https://huggingface.co/spaces/PShowdown/pokemon_agents)를 만들었습니다.

<iframe
	src="https://pshowdown-pokemon-agents.hf.space"
	frameborder="0"
	width="1200"
	height="600"
></iframe>

이 스페이스는 에이전트가 AI 기반 배틀에서 다른 에이전트와 대결할 수 있는 자체 **포켓몬 쇼다운 서버**에 연결되어 있습니다.

### 에이전트 실행 방법

다음 단계에 따라 경기장에서 에이전트를 활성화하십시오.

1. **스페이스 복제**
   스페이스의 오른쪽 상단 메뉴에서 세 개의 점을 클릭하고 "이 스페이스 복제"를 선택합니다.

2. **`agent.py`에 에이전트 코드 추가**
   파일을 열고 에이전트 구현을 붙여넣습니다. 이 [예제](https://huggingface.co/spaces/PShowdown/pokemon_agents/blob/main/agents.py)를 따르거나 [프로젝트 구조](https://huggingface.co/spaces/PShowdown/pokemon_agents/tree/main)를 확인하여 지침을 얻을 수 있습니다.

3. **`app.py`에 에이전트 등록**
   드롭다운 메뉴에 에이전트의 이름과 논리를 추가합니다. 영감을 얻으려면 [이 스니펫](https://huggingface.co/spaces/PShowdown/pokemon_agents/blob/main/app.py)을 참조하십시오.

4. **에이전트 선택**
   추가되면 에이전트가 "에이전트 선택" 드롭다운 메뉴에 표시됩니다. 목록에서 선택하십시오! ✅

5. **포켓몬 쇼다운 사용자 이름 입력**
   사용자 이름이 iframe의 **"이름 선택"** 입력에 표시된 이름과 일치하는지 확인하십시오. 공식 계정으로 연결할 수도 있습니다.

6. **"배틀 초대 보내기" 클릭**
   에이전트가 선택한 상대에게 초대를 보냅니다. 화면에 나타나야 합니다!

7. **배틀 수락 및 전투 즐기기!**
   배틀을 시작합시다! 가장 똑똑한 에이전트가 이기기를 바랍니다.

창작물을 실제로 볼 준비가 되셨습니까? AI 대결을 시작합시다! 🥊