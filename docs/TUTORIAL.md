# taskpilot — shuru se seekho (add aur subtract ke saath)

Is tutorial ke saare output asli hain. Inhe ek fresh install pe chala ke copy kiya gaya hai.
Poora chalne wala code: [`examples/10_calculator_tutorial.py`](../examples/10_calculator_tutorial.py)

## 0. Install aur entry point

```bash
pip install taskpilot
```

Sirf teen cheezein yaad rakhni hain:

| Kya | Kaise | Matlab |
|---|---|---|
| Workflow banao | `flow = Flow("naam")` | ek khaali workflow |
| Step jodo | `@flow.step` function ke upar | yeh function ek step hai |
| Chalao | `result = flow.run(a=10, b=4)` | inputs do, result lo |

**Ek hi niyam:** function ke parameter ka naam agar kisi step ka naam hai, toh use us step ka result milta hai. Warna woh ek *input* hai, jo `flow.run(...)` se aata hai.

## 1. Pehla workflow: add + subtract (`examples/calc.py`)

```python
from taskpilot import Flow

flow = Flow("calculator")

@flow.step
def add(a, b):
    return a + b

@flow.step
def subtract(a, b):
    return a - b

result = flow.run(a=10, b=4)
print(result.output)        # {'add': 14, 'subtract': 6}
```

`add` aur `subtract` ek doosre pe depend nahi karte, isliye dono **saath-saath** chalte hain. Dono "aakhri" steps hain, isliye `output` dict hai. Ek step ka result chahiye toh `result["add"]` likho, jo `14` deta hai.

Terminal se bhi chala sakte ho:
```bash
taskpilot run calc.py a=10 b=4
taskpilot show calc.py        # flow ka outline
taskpilot graph calc.py       # diagram (Mermaid)
```

## 2. Har scenario

| # | Scenario | Code ka khaas hissa | Output |
|---|---|---|---|
| 1 | Ek step | `def add(a, b)` | `14` |
| 2 | Parallel | `add(a, b)` + `subtract(a, b)` | `{'add': 14, 'subtract': 6}` |
| 3 | Ek ke baad ek `(a+b)-c` | `def subtract(add, c)` | `11` |
| 4 | Result dekhna | `result.ok`, `result.values`, `result.summary()` | table |
| 5 | Operation chuno | `@flow.step(when=lambda op: op == "add")` | `op="subtract"` → `6` |
| 5b | Ready-made | `Router(lambda op: op, {"add": ..., "subtract": ...})` | `6` |
| 6 | Bahut saare jode | `@flow.step(map="pair")` | `[3, 7, 11]` → total `21` |
| 7 | Input bhool gaye | `flow.run(a=10)` | `Missing input 'b' (needed by 'add')...` |
| 7 | Galat type | `b="4"` | `StepError` (step `add`, `TypeError`) |
| 7 | Fail pe default | `fallback=0` | `0` |
| 7 | Network jaisa error | `retry=Retry(3)` | `14`, 2 attempts |
| 8 | Loop | `flow.loop("subtract", "check", until=lambda check: check < 0, feed={"last": "check"})` | `[70, 40, 10, -20]` |
| 9 | Insaan se approval | `ctx.wait_for("ok", question=...)` → `flow.resume(path, ok=True)` | `saved 1100` |
| 10 | Async | `async def a(x)` + `await flow.arun(...)` | `15` |
| 11 | Runtime pe kitne kaam | `@flow.step(dynamic=True)` + `ctx.map(...)` | `[3, 7, 30]` |
| 12 | AI agent khud chune | `Agent(llm, tools=[add_numbers, subtract_numbers])` | `125+75-40` → `160` (asli gpt-oss:120b) |
| 13 | Workflow ek tool | `Agent(llm, tools=[chain.as_tool()])` | `11` |
| 14 | Dekho kya hua | `flow.run(..., verbose=True)` | live trace |

### 3. Ek ke baad ek
```python
@flow.step
def add(a, b):
    return a + b

@flow.step
def subtract(add, c):       # 'add' = add step ka result
    return add - c

flow.run(a=10, b=4, c=3).output     # 11
```

### 5. Operation chuno
```python
@flow.step(when=lambda op: op == "add")
def add(a, b): return a + b

@flow.step(when=lambda op: op == "subtract")
def subtract(a, b): return a - b

@flow.step
def answer(add=None, subtract=None):     # jo branch chali, uska result
    return add if add is not None else subtract
```
Jo branch skip hoti hai, uske parameter mein default (`None`) aata hai.

### 6. Bahut saare jode
```python
@flow.step(map="pair")
def add(pair):
    return pair[0] + pair[1]

@flow.step
def total(add):
    return sum(add)

flow.run(pair=[(1, 2), (3, 4), (5, 6)])   # add = [3, 7, 11], output = 21
```

### 8. Loop
```python
@flow.step
def subtract(start, step, last=None):
    return (last if last is not None else start) - step

@flow.step
def check(subtract):
    return subtract

flow.loop("subtract", "check", until=lambda check: check < 0,
          feed={"last": "check"}, max_iterations=10)
flow.run(start=100, step=30)   # rounds: 70, 40, 10, -20
```

### 9. Insaan se approval
```python
@flow.step(when=lambda add: add > 1000)
def approval(add, ctx):
    return ctx.wait_for("ok", question=f"Result {add} hai. Save karein?")

@flow.step
def save(add, approval=True):
    return f"saved {add}" if approval else "not saved"

r = flow.run(a=900, b=200)       # r.status == "paused"
r.save("run.json")               # file mein, baad mein ya doosre process mein
flow.resume("run.json", ok=True).output   # 'saved 1100'
```

### 12. Agent (AI khud tay kare)
```python
from taskpilot import Agent, Ollama, tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

agent = Agent(Ollama("gpt-oss:120b"), tools=[add_numbers, subtract_numbers])
agent.run("What is 125 plus 75, minus 40?").output      # '160'
```
OpenAI chahiye toh: `OpenAICompatible("gpt-4o-mini")`. Key `OPENAI_API_KEY` se aati hai.

## Multiple agents ke saath workflow

Poora code, jo asli `gpt-oss:120b` pe chalaya gaya: [`examples/11_multi_agent_tutorial.py`](../examples/11_multi_agent_tutorial.py)

Teen agents hain, har ek ek kaam mein accha:
```python
adder      = Agent(llm, tools=[add_numbers],      name="adder",      system="You add numbers. Use your tool. Number only.")
subtractor = Agent(llm, tools=[subtract_numbers], name="subtractor", system="You subtract numbers. Use your tool. Number only.")
checker    = Agent(llm, name="checker", system="Check arithmetic. Reply 'correct' or 'wrong: <answer>'.")
```

| # | Tarika | Kaun tay karta hai | Code | Asli output |
|---|---|---|---|---|
| 1 | **Ek ke baad ek** | aap | `flow.add(adder.as_step("total", "Add {a} and {b}."))`<br>`flow.add(subtractor.as_step("final", "Subtract {c} from {total}."))` | `(125+75)-40` → `200` → `160` |
| 2 | **Saath-saath** | aap | do `as_step`, dono sirf `{a}`, `{b}` maangte hain | `{'sum': '620', 'difference': '380'}`, ~2s mein dono |
| 3 | **Router** | aapka rule | `Router(classify, {"add": adder.as_step(...), "subtract": subtractor.as_step(...)})` | "19 plus 23" → adder → `42` |
| 4 | **Supervisor** | boss agent | `Supervisor(llm, [adder, subtractor])` | boss ne khud `adder`, phir `subtractor` ko bheja → `160` |
| 5 | **Handoff** | agents aapas mein | `Handoff([reception, adder, subtractor])` | `reception -> subtractor` → `53` |
| 6 | **Solve + check loop** | checker | `ReviewLoop(solve, check, until=...)` | `999`, checker: `correct` |
| 7 | **Beech mein insaan** | aap + insaan | step mein `ctx.wait_for("go")`, phir `flow.resume(r, go=True)` | ruka "Total 1200 hai", resume → `1000` (adder dobara nahi chala) |
| 8 | **Baat-cheet** | baari-baari | `GroupChat([adder, subtractor, checker], until=checker_agrees, finishers=["checker"])` | adder `75` → subtractor `65` → checker `correct` |

**Agents ke beech data kaise jaata hai?** Workflow (1, 2, 3, 7) mein prompt ke `{naam}` se: `{total}` matlab `total` step ka jawab. Supervisor, Handoff aur GroupChat (4, 5, 8) mein agents ek hi conversation share karte hain.

**Kaunsa chunein?**
- Steps pakke hain → **1 / 2** (sabse sasta, har baar same).
- Sawaal ki type pe depend → **3 Router** (aapka rule) ya **5 Handoff** (AI ka faisla).
- Kaam ke tukde pehle se pata nahi → **4 Supervisor**.
- Quality zaroori hai → **6 ReviewLoop**.
- Paisa, delete jaise kaam → **7**, insaan ki approval ke saath.
- Ideas/discussion → **8 GroupChat**. `finishers` se tay karo ki kaun baat khatam kar sakta hai.

Sab mix bhi ho sakte hain: ek Supervisor kisi Handoff team ko worker bana sakta hai, aur koi bhi team `team.as_step()` se workflow ka step ban sakti hai.

## Business user: bina code ke (AutoFlow)

**Developer ek baar** tools likhta hai ([`examples/office_tools.py`](../examples/office_tools.py)):
```python
from taskpilot import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    ...

tools = [add_numbers, subtract_numbers, send_email]
approve = ["send_email"]        # email se pehle hamesha insaan se poochho
```

**Business user** sirf yeh chalata hai aur browser kholta hai:
```bash
taskpilot serve office_tools.py        # http://127.0.0.1:8000
```
Wahan woh likhta hai: *"Is mahine ki sales 125000 aur 75000 thi, returns 40000. Net sales nikaalo aur boss@company.com ko email karo"*

1. **Make a plan** dabaya → AI ne 3 steps ka plan dikhaya (jodo → ghatao → email). Email pe "asks you first" ka badge tha.
2. **Run this plan** → jodna aur ghatana turant hua (200000, 160000). Email pe page ne poocha: *"Allow send_email(... 160000 ...)?"*
3. **Approve** → email gaya.

Terminal pasand ho toh: `taskpilot ask office_tools.py "..."`. Har mahine wahi kaam karna ho toh `--save-plan monthly.json`, phir `--plan monthly.json` se **bina AI ke** dobara chalao.

### Web page ke 5 tab (sab bina code)

| Tab | Kya kar sakte ho |
|---|---|
| **Run** | request likho, phir "How should it be done?" chuno. **AI decides** apni salah wajah ke saath dikhata hai ("Use this" / "Use and run"). Ya khud type chuno: Step plan, One agent, One after another, All at once, Pick the right agent, Do and check, Each item separately, Boss delegates, Pass to specialist, Team discussion. Agents ko us order mein click karo jismein unhe kaam karna hai. |
| **Agents** | code wale agents dekho; apne agent banao, badlo ya hatao: naam, instructions, aur kaunse tools use kar sakta hai |
| **Tools** | system kya-kya kar sakta hai; "asks first" wale tools approval maangte hain |
| **Saved** | kisi bhi run ko naam dekar save karo, aur ek click mein dobara chalao |
| **History** | pichhle runs dekho aur kholo |

**Asli run (gpt-oss:120b):** request thi *"Order A7 ke customer ko thank-you email likho aur bhejo, bhejne se pehle reviewer check kare"*.
1. AI ki salah aayi: **Do and check, support → reviewer**, is wajah ke saath: *"support email likhega, reviewer check karega ki polite aur 80 words se kam hai, phir bheja jaayega"*.
2. Support ne draft likha (is waqt uske paas `send_email` tha hi nahi), aur reviewer ne **OK** kaha.
3. Ab support ne email bhejne ki koshish ki, toh page pe **Approval needed** card aaya.
4. **Approve** dabaya, tab email gaya ("Action done").

Agent ke andar bhi approval wala tool apne aap nahi chalta: call ruk jaati hai aur card dikhta hai.

**Safety:** AI sirf plan banata hai. Plan check hota hai: tool asli hai, zaroori arguments hain, aur sirf pichhle steps ka result use hua hai. Galti ho toh AI se theek karwaya jaata hai. Approval wale tools AI ki marzi se nahi, engine ke niyam se rukte hain.

## Kitna bharosa? (Reliability)

Poori jaankari, naape hue numbers ke saath: [RELIABILITY.md](RELIABILITY.md). Chhota saar:

- Koi AI 100% sahi nahi hota. Hamara vaada yeh hai: **galat jawab sahi ki tarah bahar nahi jaayega, aur system fatega nahi.** Pakka na ho toh poochhega.
- Model ko jaan-boojh ke kharab kiya (25% numbers badle, 12% outage). **Guardrails ke saath 69 runs mein 0 galat jawab aur 0 crash.** Bina guardrails 11 galat jawab aur 10 crash the.
- Sabse kaam ki seekh: **jaanch alag model se karwao.** Ek model apni galti khud nahi pakadta.

```python
auto  = AutoFlow(llm, tools, review=[Ollama("gemma4:31b"), Ollama("nemotron-3-super")], escalate=strong_llm)
agent = Agent(llm, tools=[get_order], verify=Ollama("nemotron-3-super"), escalate=strong_llm)
llm   = FallbackLLM([primary_llm, backup_llm])                     # model down? backup chalega
memory = ConversationMemory(llm, max_tokens=4000)                  # lambi baat-cheet ka auto-compact
```

Workspace (web page) mein plan review aur answer verify **default chalu** hain.

## Doosre platform ke saath (MCP, REST, A2A)

Maan lo kisi company ka apna AI platform hai, aur uske paas apni **tool service** aur **agent service** hai. taskpilot dono se seedha jud jaata hai, aur khud bhi unke liye service ban sakta hai:

```python
from taskpilot import A2AAgent, Agent, MCPClient, Supervisor, openapi_tools

crm   = MCPClient.http("https://tools.company.com/mcp", headers={"Authorization": "Bearer ..."})
tools = crm.tools() + openapi_tools("https://api.company.com/openapi.json")   # MCP tools + REST API tools
reporter = A2AAgent("https://agents.company.com/reporter")                    # unka agent (A2A protocol)

team = Supervisor(llm, [Agent(llm, tools=tools, name="support"), reporter], verify=checker_llm)
```

Ulta bhi chalta hai: hamare tools, workflows aur agents doosron ko de do:

```bash
taskpilot mcp tools.py                     # IDE ya koi bhi MCP host use kar le
taskpilot serve tools.py --token SECRET    # web page + REST API (/v1) + MCP (/mcp), ek hi port pe
```

- Approval wale tools (jaise `send_email`) MCP pe apne aap nahi diye jaate. Person ki "haan" web page ya REST API se aati hai.
- Ek live test mein remote agent ne sirf "all good" bola, aur boss AI ne apni taraf se revenue ke numbers gadh diye. `verify=` lagane pe yeh 3 mein se 3 baar pakda gaya. Remote jawabon pe hamesha verify lagao.
- Poori jaankari: [INTEGRATIONS.md](INTEGRATIONS.md). Chalta hua example: `examples/12_connect_platforms.py`.

## Kab kya?
- **Workflow (functions):** jab aapko pata hai ki kya karna hai. Tez, hamesha same chalta hai, aur test kiya ja sakta hai.
- **Agent:** jab user kuch bhi likh sakta hai aur AI ko tay karna hai ki kaunsa tool chalana hai.
- **Dono:** AI faisla le, lekin pakke kaam (paisa, database) workflow ke rules se hon.
