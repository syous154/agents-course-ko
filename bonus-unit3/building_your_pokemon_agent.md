# 나만의 포켓몬 배틀 에이전트 만들기

이제 게임에서 에이전트 AI의 잠재력과 한계를 탐색했으니 직접 실습해 볼 차례입니다. 이 섹션에서는 과정 전체에서 배운 모든 것을 사용하여 **포켓몬 스타일 턴제 전투에서 싸울 나만의 AI 에이전트를 구축**합니다.

시스템을 네 가지 주요 구성 요소로 나누겠습니다.

- **Poke-env:** 규칙 기반 또는 강화 학습 포켓몬 봇을 훈련하도록 설계된 Python 라이브러리입니다.

- **Pokémon Showdown:** 에이전트가 싸울 온라인 배틀 시뮬레이터입니다.

- **LLMAgentBase:** LLM을 Poke-env 배틀 환경과 연결하기 위해 구축한 사용자 지정 Python 클래스입니다.

- **TemplateAgent:** 나만의 고유한 배틀 에이전트를 만들기 위해 완성할 스타터 템플릿입니다.

이러한 각 구성 요소를 자세히 살펴보겠습니다.

## 🧠 Poke-env

![배틀 gif](https://github.com/hsahovic/poke-env/raw/master/rl-gif.gif)

[Poke-env](https://github.com/hsahovic/poke-env)는 원래 [Haris Sahovic](https://huggingface.co/hsahovic)이 강화 학습 봇을 훈련하기 위해 구축한 Python 인터페이스이지만, 우리는 이를 에이전트 AI에 맞게 용도를 변경했습니다.
이를 통해 에이전트는 간단한 API를 통해 Pokémon Showdown과 상호 작용할 수 있습니다.

에이전트가 상속할 `Player` 클래스를 제공하여 그래픽 인터페이스와 통신하는 데 필요한 모든 것을 다룹니다.

**설명서**: [poke-env.readthedocs.io](https://poke-env.readthedocs.io/en/stable/)
**리포지토리**: [github.com/hsahovic/poke-env](https://github.com/hsahovic/poke-env)

## ⚔️ Pokémon Showdown

[Pokémon Showdown](https://pokemonshowdown.com/)은 에이전트가 라이브 포켓몬 배틀을 플레이할 [오픈 소스](https://github.com/smogon/Pokemon-Showdown) 배틀 시뮬레이터입니다.
실시간으로 배틀을 시뮬레이션하고 표시하는 전체 인터페이스를 제공합니다. 우리의 챌린지에서 봇은 인간 플레이어처럼 행동하여 턴마다 기술을 선택합니다.

모든 참가자가 배틀에 사용할 서버를 배포했습니다. 누가 최고의 AI 배틀 에이전트를 만드는지 봅시다!

**리포지토리**: [github.com/smogon/Pokemon-Showdown](https://github.com/smogon/Pokemon-Showdown)
**웹사이트**: [pokemonshowdown.com](https://pokemonshowdown.com/)

## 🔌 LLMAgentBase

`LLMAgentBase`는 **Poke-env**의 `Player` 클래스를 확장하는 Python 클래스입니다.
**LLM**과 **포켓몬 배틀 시뮬레이터** 사이의 다리 역할을 하여 입/출력 형식 지정 및 배틀 컨텍스트 유지를 처리합니다.

이 기본 에이전트는 환경과 상호 작용하기 위한 도구 세트(`STANDARD_TOOL_SCHEMA`에 정의됨)를 제공하며 다음을 포함합니다.

- `choose_move`: 배틀 중 공격을 선택하기 위해
- `choose_switch`: 포켓몬을 교체하기 위해

LLM은 이러한 도구를 사용하여 경기 중에 결정을 내려야 합니다.

### 🧠 핵심 논리

- `choose_move(battle: Battle)`: 매 턴 호출되는 기본 메서드입니다. `Battle` 개체를 가져와 LLM의 출력을 기반으로 작업 문자열을 반환합니다.

### 🔧 주요 내부 메서드

- `_format_battle_state(battle)`: 현재 배틀 상태를 문자열로 변환하여 LLM으로 보내기에 적합하게 만듭니다.

- `_find_move_by_name(battle, move_name)`: `choose_move`를 호출하는 LLM 응답에 사용되는 이름으로 기술을 찾습니다.

- `_find_pokemon_by_name(battle, pokemon_name)`: LLM의 교체 명령에 따라 교체할 특정 포켓몬을 찾습니다.

- `_get_llm_decision(battle_state)`: 이 메서드는 기본 클래스에서 추상적입니다. LLM을 쿼리하고 응답을 구문 분석하는 방법을 정의하는 자신만의 에이전트(다음 섹션 참조)에서 구현해야 합니다.

다음은 해당 의사 결정이 어떻게 작동하는지 보여주는 발췌문입니다.


```python
STANDARD_TOOL_SCHEMA = {
    "choose_move": {
        ...
    },
    "choose_switch": {
        ...
    },
}

class LLMAgentBase(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_tools = STANDARD_TOOL_SCHEMA
        self.battle_history = []

    def _format_battle_state(self, battle: Battle) -> str:
        active_pkmn = battle.active_pokemon
        active_pkmn_info = f"당신의 활성 포켓몬: {active_pkmn.species} " \
                           f"(타입: {'/'.join(map(str, active_pkmn.types))}) " \
                           f"HP: {active_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"상태: {active_pkmn.status.name if active_pkmn.status else '없음'} " \
                           f"부스트: {active_pkmn.boosts}"

        opponent_pkmn = battle.opponent_active_pokemon
        opp_info_str = "알 수 없음"
        if opponent_pkmn:
            opp_info_str = f"{opponent_pkmn.species} " \
                           f"(타입: {'/'.join(map(str, opponent_pkmn.types))}) " \
                           f"HP: {opponent_pkmn.current_hp_fraction * 100:.1f}% " \
                           f"상태: {opponent_pkmn.status.name if opponent_pkmn.status else '없음'} " \
                           f"부스트: {opponent_pkmn.boosts}"
        opponent_pkmn_info = f"상대의 활성 포켓몬: {opp_info_str}"

        available_moves_info = "사용 가능한 기술:\n"
        if battle.available_moves:
            available_moves_info += "\n".join(
                [f"- {move.id} (타입: {move.type}, BP: {move.base_power}, 정확도: {move.accuracy}, PP: {move.current_pp}/{move.max_pp}, 카테고리: {move.category.name})"
                 for move in battle.available_moves]
            )
        else:
             available_moves_info += "- 없음 (교체 또는 발버둥쳐야 함)"

        available_switches_info = "사용 가능한 교체:\n"
        if battle.available_switches:
              available_switches_info += "\n".join(
                  [f"- {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%, 상태: {pkmn.status.name if pkmn.status else '없음'})")"
                   for pkmn in battle.available_switches]
              )
        else:
            available_switches_info += "- 없음"

        state_str = f"{active_pkmn_info}\n"
                    f"{opponent_pkmn_info}\n\n"
                    f"{available_moves_info}\n\n"
                    f"{available_switches_info}\n\n"
                    f"날씨: {battle.weather}\n"
                    f"지형: {battle.fields}\n"
                    f"당신의 필드 상태: {battle.side_conditions}\n"
                    f"상대의 필드 상태: {battle.opponent_side_conditions}"
        return state_str.strip()

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized_name = normalize_name(move_name)
        # 정확한 ID 일치 우선
        for move in battle.available_moves:
            if move.id == normalized_name:
                return move
        # 대체: 표시 이름 확인 (신뢰성 낮음)
        for move in battle.available_moves:
            if move.name.lower() == move_name.lower():
                print(f"경고: ID '{move.id}' 대신 표시 이름 '{move.name}'으로 기술 일치. 입력은 '{move_name}'이었습니다.")
                return move
        return None

    def _find_pokemon_by_name(self, battle: Battle, pokemon_name: str) -> Optional[Pokemon]:
        normalized_name = normalize_name(pokemon_name)
        for pkmn in battle.available_switches:
            # 비교를 위해 종 이름 정규화
            if normalize_name(pkmn.species) == normalized_name:
                return pkmn
        return None

    async def choose_move(self, battle: Battle) -> str:
        battle_state_str = self._format_battle_state(battle)
        decision_result = await self._get_llm_decision(battle_state_str)
        print(decision_result)
        decision = decision_result.get("decision")
        error_message = decision_result.get("error")
        action_taken = False
        fallback_reason = ""

        if decision:
            function_name = decision.get("name")
            args = decision.get("arguments", {})
            if function_name == "choose_move":
                move_name = args.get("move_name")
                if move_name:
                    chosen_move = self._find_move_by_name(battle, move_name)
                    if chosen_move and chosen_move in battle.available_moves:
                        action_taken = True
                        chat_msg = f"AI 결정: 기술 '{chosen_move.id}' 사용."
                        print(chat_msg)
                        return self.create_order(chosen_move)
                    else:
                        fallback_reason = f"LLM이 사용할 수 없거나 잘못된 기술 '{move_name}'을 선택했습니다."
                else:
                     fallback_reason = "LLM 'choose_move'가 'move_name' 없이 호출되었습니다."
            elif function_name == "choose_switch":
                pokemon_name = args.get("pokemon_name")
                if pokemon_name:
                    chosen_switch = self._find_pokemon_by_name(battle, pokemon_name)
                    if chosen_switch and chosen_switch in battle.available_switches:
                        action_taken = True
                        chat_msg = f"AI 결정: '{chosen_switch.species}'(으)로 교체."
                        print(chat_msg)
                        return self.create_order(chosen_switch)
                    else:
                        fallback_reason = f"LLM이 사용할 수 없거나 잘못된 교체 '{pokemon_name}'을 선택했습니다."
                else:
                    fallback_reason = "LLM 'choose_switch'가 'pokemon_name' 없이 호출되었습니다."
            else:
                fallback_reason = f"LLM이 알 수 없는 함수 '{function_name}'을 호출했습니다."

        if not action_taken:
            if not fallback_reason:
                 if error_message:
                     fallback_reason = f"API 오류: {error_message}"
                 elif decision is None:
                      fallback_reason = "LLM이 유효한 함수 호출을 제공하지 않았습니다."
                 else:
                      fallback_reason = "LLM 결정 처리 중 알 수 없는 오류."

            print(f"경고: {fallback_reason} 임의의 작업 선택.")

            if battle.available_moves or battle.available_switches:
                 return self.choose_random_move(battle)
            else:
                 print("AI 대체: 사용 가능한 기술이나 교체가 없습니다. 발버둥/기본 사용.")
                 return self.choose_default_move(battle)

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        raise NotImplementedError("하위 클래스는 _get_llm_decision을 구현해야 합니다.")
```

**전체 소스 코드**: [agents.py](https://huggingface.co/spaces/Jofthomas/twitch_streaming/blob/main/agents.py)

## 🧪 TemplateAgent

이제 재미있는 부분입니다! LLMAgentBase를 기반으로 자신만의 에이전트를 구현하고, 자신만의 전략으로 순위표에 오를 시간입니다.

이 템플릿에서 시작하여 자신만의 논리를 구축하게 됩니다. 또한 **OpenAI**, **Mistral** 및 **Gemini** 모델을 사용하는 세 가지 [완전한 예제](https://huggingface.co/spaces/Jofthomas/twitch_streaming/blob/main/agents.py)를 제공하여 안내합니다.

다음은 템플릿의 단순화된 버전입니다.

```python
class TemplateAgent(LLMAgentBase):
    """결정을 위해 Template AI API를 사용합니다."""
    def __init__(self, api_key: str = None, model: str = "model-name", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.template_client = TemplateModelProvider(api_key=...)
        self.template_tools = list(self.standard_tools.values())

    async def _get_llm_decision(self, battle_state: str) -> Dict[str, Any]:
        """상태를 LLM으로 보내고 함수 호출 결정을 다시 받습니다."""
        system_prompt = (
            "당신은..."
        )
        user_prompt = f"..."

        try:
            response = await self.template_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            message = response.choices[0].message
            
            return {"decision": {"name": function_name, "arguments": arguments}}

        except Exception as e:
            print(f"호출 중 예기치 않은 오류: {e}")
            return {"error": f"예기치 않은 오류: {e}"}
```

이 코드는 즉시 실행되지 않으며, 사용자 지정 논리를 위한 청사진입니다.

모든 조각이 준비되었으므로 이제 경쟁력 있는 에이전트를 구축할 차례입니다. 다음 섹션에서는 에이전트를 서버에 배포하고 실시간으로 다른 사람들과 배틀하는 방법을 보여줍니다.

배틀을 시작합시다! 🔥
