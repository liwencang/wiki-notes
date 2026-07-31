## Agetnt



## 基于LangGraph的Agent



![image-20260715172057117](./Agent.assets/image-20260715172057117.png)



### LangGraph本地服务

![image-20260715174034579](./Agent.assets/image-20260715174034579.png)

> [!Tip]
>
> langgraph up 是国外的一般不用，国内使用前三个命令
>
> 第二个命令使用的最频繁



#### 一、创建Python虚拟环境

虚拟环境安装步骤

1. 安装好python解释器：Python >= 3.11 is required
2. 安装虚拟环境库，在cmd中输入：

``` cmd
pip install virtualenv
# 这样在python的安装目录D:\program\python3.12.8\Scripts 就有了virtualenv.exe
```

3. 创建虚拟环境

``` cmd
# 在自己的虚拟环境文件夹下执行
virtualenv langgraph_env
```

![image-20260715180648344](./Agent.assets/image-20260715180648344.png)

执行完之后目录下就多了一个刚刚创建的虚拟环境文件

<img src="./Agent.assets/image-20260715180732658.png" alt="image-20260715180732658" style="zoom:50%;" />

4. 激活虚拟环境，在cmd中进入到第三步创建的env_name/Scripts目录下，执行：

``` cmd
activate
```

​	执行成功后，在cmd中，当前输入行前面会有（env_name）的前缀

​	在当前状态下，使用pip就是在虚拟环境中安装第三方库了

5. 退出虚拟环境，cmd中输入：

``` cmd
deactivate
```



#### 二、安装LangGraph CLI

``` cmd
# Python >= 3.11 is required

pip install --upgrade "langgraph-cli[inmem]"
```

安装的包的路径：E:\env_py\langgraph_env\Lib\site-packages



#### 三、创建 LangGraph 应用

从new-langgraph-project-python 模板或new-langgraph-project-js模板创建一个新的应用。此模板演示了一个单节点应用，可以根据自己的逻辑进行扩展

> [!Note]
>
> 如果使用`langgraph new`命令时未指定模板，将显示一个交互式菜单，允许您从可用模板列表中选择



``` cmd
langgraph new path/to/your/app --template new-langgraph-project-python
```

![image-20260715185539433](./Agent.assets/image-20260715185539433.png)

用pycharm打开，选择刚刚创建的虚拟环境的解释器

<img src="./Agent.assets/image-20260715185803132.png" alt="image-20260715185803132" style="zoom: 50%;" />



#### 四、安装项目依赖

![image-20260715190340200](./Agent.assets/image-20260715190340200.png)

执行命令 `pip install -e .`会自动寻找当前目录下的`pyproject.toml`文件



``` cmd
pip install -e .
```



![image-20260715190647609](./Agent.assets/image-20260715190647609.png)

#### 五、修改graph.py的代码



``` python
import os, dotenv
from langchain_qwq import ChatQwen
from langchain.agents import create_agent
from pydantic import SecretStr

dotenv.load_dotenv()
api_key_raw = os.getenv("API_KEY")
if not api_key_raw:
    raise ValueError("环境变量 API_KEY 未配置，请检查 .env 文件")

api_key: SecretStr = SecretStr(api_key_raw)

llm = ChatQwen(
    model="qwen3.6-flash-2026-04-16",
    enable_thinking=False,
    temperature=0.8,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    # max_tokens=3_000,
    # timeout=None,
    # max_retries=2,
    # other params...
)


def get_weather(city: str) -> str:
    """获取一个城市的天气."""
    return f"今天 {city} 是晴天!"


graph = create_agent(
    llm,
    tools=[get_weather],
    system_prompt="你是一个智能助手"
)

res = llm.invoke("你好")
print(res)

```



#### 六、启动LangGraph的服务器

![image-20260715191455507](./Agent.assets/image-20260715191455507.png)

![image-20260715191550860](./Agent.assets/image-20260715191550860.png)

#### 七、测试和访问agent的API

1. ##### LangGraph Studio

![image-20260716110510393](./Agent.assets/image-20260716110510393.png)



2. ##### PythonSDK测试

``` shell
pip install langgraph-sdk
```

> [!Tip]
>
> 测试前先运行langgraph  `langgraph dev`

异步测试

``` python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "今天广州番禺的天气怎么样",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

同步测试

``` python
import asyncio

from langgraph_sdk import get_client, get_sync_client

client = get_sync_client(url="http://localhost:2024")
for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="agent",
        input={
            "messages": [{
                "role": "human",
                "content": "今天广州番禺天气怎么样"
            }]
        },
        stream_mode="messages-tuple"
):
    if isinstance(chunk.data,list) and "type" in chunk.data[0] and chunk.data[0]["type"] == "AIMessageChunk":
        print(chunk.data[0]["content"],end="",flush=True)
```



##### 3. JavaScript SDK测试

``` shell
npm install @langchain/langgraph-sdk
```

向LangGraph服务区发送消息

``` javascript
const { Client } = await import("@langchain/langgraph-sdk");

// only set the apiUrl if you changed the default port when calling langgraph dev
const client = new Client({ apiUrl: "http://localhost:2024"});

const streamResponse = client.runs.stream(
    null, // Threadless run
    "agent", // Assistant ID
    {
        input: {
            "messages": [
                { "role": "user", "content": "What is LangGraph?"}
            ]
        },
        streamMode: "messages-tuple",
    }
);

for await (const chunk of streamResponse) {
    console.log(`Receiving new event of type: ${chunk.event}...`);
    console.log(JSON.stringify(chunk.data));
    console.log("\n\n");
}
```



##### 4. REST API测试

``` python
curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data "{
        \"assistant_id\": \"agent\",
        \"input\": {
            \"messages\": [
                {
                    \"role\": \"human\",
                    \"content\": \"What is LangGraph?\"
                }
            ]
        },
        \"stream_mode\": \"messages-tuple\"
    }"
```





### 工具

#### 创建工具的三种方式

##### 1. 使用@Tool装饰器

![image-20260703114150447](./Agent.assets/image-20260703114150447.png)



``` python
# 第一种：谷歌风格描述

# 这里的名称可不写，默认就是函数名
# parse_docstring是解析下面的谷歌风格文档用的，格式严格按照下面的来！！！
@tool("web_search", parse_docstring=True)
def web_search(question: str) -> str:
    """互联网搜索工具

    Args:
        question: 用于互联网查询的查询信息

    Returns:
        网页检索文本
    """

    try:
        response = client.web_search.web_search(
            search_engine="search_std",
            search_query=question,
        )
        logger.info(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
    except Exception as e:
        logger.error(e)
        return f"ERROR: {str(e)})"

```



``` python

class ToolArgSchema(BaseModel):
    question: str = Field(..., description="用于互联网查询的查询信息")


# 第二种：pydantic方式
@tool("web_search", args_schema=ToolArgSchema, description="互联网搜索工具")
def web_search(question: str) -> str:
    pass

```



``` python
# 第三种：Annotated方式
@tool
def web_search(
        question: Annotated[str,"用于互联网查询的查询信息"]
):
    """用于搜索的工具"""
    print(question)

web_search.invoke("今天广州天气怎么样")

```



``` python
# 第四种：使用函数封装工具，支持异步或同步调用
def calculate5(
        a: float,
        b: float,
        operation: str) -> float:
    """工具函数：计算两个数字的运算结果

    Args:
        a: 第一个需要输入的数字。
        b: 第二个需要输入的数字。
        operation: 运算类型，只能是add、subtract、multiply和divide中的任意一个。

    Returns:
        返回两个输入数字的运算结果。

    """
    print(f"调用 calculate 工具，第一个数字：{a}, 第二个数字：{b}, 运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为零")

    return result


async def calculate6(
        a: float,
        b: float,
        operation: str) -> float:
    """工具函数：计算两个数字的运算结果

    Args:
        a: 第一个需要输入的数字。
        b: 第二个需要输入的数字。
        operation: 运算类型，只能是add、subtract、multiply和divide中的任意一个。

    Returns:
        返回两个输入数字的运算结果。

    """
    print(f"调用 calculate 工具，第一个数字：{a}, 第二个数字：{b}, 运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为零")

    return result


# 创建了一个工具
calculater = StructuredTool.from_function(
    func=calculate5,
    name="calculater",
    description="工具函数：计算两个数字的运算结果",
    return_direct=False,
    coroutine=calculate6
)
```





##### 2. 继承BaseTool工具类

> [!TIP]
>
> 复杂工具建议使用继承BaseTool的方式，平常用的不多

> arg_schema有两种声明方式

``` python
from typing import Any, Type

from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field, create_model

from agent.my_llm import client # 智谱AI的一个网络搜索模型


class ArgsSchema(BaseModel):
    query: str = Field(..., description="用于互联网查询的查询信息")


class MyCustomTool(BaseTool):
    name: str = "web_search2"

    # 第一种参数声明
    # args_schema: Type[BaseModel]  = ArgsSchema

    # 第二种参数声明
    def __init__(self):
        super().__init__()
        self.args_schema = create_model("query", query=(str, Field(..., description="用于互联网查询的查询信息")))

    description: str = "互联网搜索工具"

    def _run(self, query: str) -> str:
        try:
            response = client.web_search.web_search(
                search_engine="search_std",
                search_query=query,
            )
            logger.info(response)
            if response.search_result:
                return "\n\n".join([d.content for d in response.search_result])
        except Exception as e:
            logger.error(e)
            return f"ERROR: {str(e)})"
        
    # 这里写一个异步函数，根据调用方式自动选择执行同步还是异步
    async def _arun(self, query: str):
        return self._run(query)
```



##### 3. 根据Runnable对象创建工具

``` python
import os, dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_qwq import ChatQwen
from pydantic import SecretStr, BaseModel, Field

dotenv.load_dotenv()
api_key_raw = os.getenv("API_KEY")
if not api_key_raw:
    raise ValueError("环境变量 API_KEY 未配置，请检查 .env 文件")

api_key: SecretStr = SecretStr(api_key_raw)

llm = ChatQwen(
    model="deepseek-v4-pro",
    enable_thinking=False,
    temperature=0.8,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
prompt = PromptTemplate.from_template("你是一个报幕词生成助手，可以根据问题和主题生成报幕词，question:{question}，topic:{topic}")
parser = StrOutputParser()
chain = prompt | llm | parser

class OvertureSchema(BaseModel):
    question: str = Field(str,description="生成报幕词的要求")
    topic: str = Field(str,description="生成报幕词的主题")

overture_tool = chain.as_tool(
    args_schema=OvertureSchema,
    name="overture_tool",
    description="报幕词生成工具",
)
```



##### 3. MCP服务端获得工具



### Agent的上下文和记忆

#### 上下文

##### Configurable静态配置

下面模拟一个场景，在调用agent时传入user_id，然后定义一个根据user_id获取user_info的工具

``` python
@tool(description="获取用户信息")
def get_user_info(config: RunnableConfig) -> dict:
    """获取用户信息"""
    user_id = config["configurable"]["user_id"]
    return {
        "user_id": user_id,
        "name": "张三",
        "age": 18
    }


graph = create_agent(
    llm,
    tools=[get_user_info],
    # middleware=[
    #     dynamic_system_prompt,
    # ]
    system_prompt="你是一个智能助手,尽可能调用工具回答用户的问题"
)
```

``` python
import asyncio

from langgraph_sdk import get_client, get_sync_client

client = get_sync_client(url="http://localhost:2024")
for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="agent",
        input={
            "messages": [{
                "role": "human",
                "content": "用户名字年龄分别是什么"
            }]
        },
        stream_mode="messages-tuple",
    config={"configurable":{"user_id":"user_001"}}
):
    # print(chunk)
    if isinstance(chunk.data,list) and "type" in chunk.data[0] and chunk.data[0]["type"] == "AIMessageChunk":
        print(chunk.data[0]["content"],end="",flush=True)
```



##### AgentState

案例：（给用户发出一个祝福语句）输入username ---> config ---> 工具1---> 把username修改到State中------>工具2------>获取State的username得到最终答案

``` python
graph = create_agent(
    llm,
    tools=[get_weather, overture_tool,get_user_name],
    # middleware=[
    #     dynamic_system_prompt,
    # ]
    system_prompt="你是一个智能助手,尽可能调用工具回答用户的问题",
    state_schema=CustomAgentState
)
```

``` python
# 工具一
@tool(description="获取用户年龄的工具")
def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig) -> Command:
    user_name = config['configurable'].get("username", "未查询到用户名")
    return Command(
        update={
            "username": user_name,
            "messages": [ToolMessage(
                content=f"用户名获取工具调用成功{user_name}",
                tool_call_id=tool_call_id,
            )]
        }
    )
    
# 工具二
@tool(description="根据用户名，生成响应的祝福语")
def greet_user(state: Annotated[CustomAgentState, InjectedState]):
    username = state.get("username","未命名")
    return f"祝福你，username，健康平安，向上成长"
 
```

``` python
client = get_sync_client(url="http://localhost:2024")
for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="agent",
        input={
            "messages": [{
                "role": "human",
                "content": "用户名字是什么"
            }]
        },
        stream_mode="messages-tuple",
        config={"configurable": {"user_id": "user_001", "username": "张三"}}
):
    # print(chunk)
    if isinstance(chunk.data, list) and "type" in chunk.data[0] and chunk.data[0]["type"] == "AIMessageChunk":
        print(chunk.data[0]["content"], end="", flush=True)

```



#### 记忆存储

##### 短期存储：线程级存储（会话级）

短期存储使Agent能够跟踪多轮对话。要使用它，您必须：

1. 在创建代理时提供checkpointer。checkpointer可以实现代理状态的持久性。
2. 在运行代理时在配置中提供thread_id。thread_id是对话会话的唯一标识符。

> [!Tip]
>
> 由于需要用langsmith自带的checkpointer才能继续使用langgraph dev，所以下面就不再展示langsmith服务了，
>
> 直接用invoke的方式运行agent

``` python
import os, dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_qwq import ChatQwen
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import SecretStr

from tests.my_tests.AgentStateDemo1 import get_user_name, greet_user
from tests.my_tests.Runnable方式创建Tool import overture_tool

dotenv.load_dotenv()
api_key_raw = os.getenv("API_KEY")
if not api_key_raw:
    raise ValueError("环境变量 API_KEY 未配置，请检查 .env 文件")

api_key: SecretStr = SecretStr(api_key_raw)

llm = ChatQwen(
    model="qwen3.7-flash-2026-07-15",
    enable_thinking=False,
    temperature=0.8,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def get_weather(city: str) -> str:
    """获取一个城市的天气."""
    return f"今天 {city} 是晴天!"

# 从这里开始看
DB_URI = "postgresql://postgres:123456@localhost:5432/langgraph_demo?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    graph = create_agent(
        llm,
        tools=[get_weather],
        system_prompt="你是一个智能助手,尽可能调用工具回答用户的问题",
        checkpointer=checkpointer,
    )
    config = {
        "configurable": {"thread_id":"001"}
    }
    state = graph.get_state(config)
    print(state)
    res1 = graph.invoke({"messages":[{"role":"user","content":"今天广州天气怎么样？"}]},config)
    print(res1)
    res2 = graph.invoke({"messages":[{"role":"user","content":"石家庄呢？"}]},config)
    print(res2)

```

> [!Tip]
>
> 使用这种方式必须提供thread_id（其实是会话id，翻译过来就这样了），
>
> 因为之前使用langsmith服务是自动封装的



##### 长期存储

> [!Warning]
>
> 待完善，目前长期存储还不生效

待完善

``` python
import os, dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_qwq import ChatQwen
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from pydantic import SecretStr

from tests.my_tests.AgentStateDemo1 import get_user_name, greet_user
from tests.my_tests.Runnable方式创建Tool import overture_tool

dotenv.load_dotenv()
api_key_raw = os.getenv("API_KEY")
if not api_key_raw:
    raise ValueError("环境变量 API_KEY 未配置，请检查 .env 文件")

api_key: SecretStr = SecretStr(api_key_raw)

llm = ChatQwen(
    model="qwen3.7-flash-2026-07-15",
    enable_thinking=False,
    temperature=0.8,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def get_weather(city: str) -> str:
    """获取一个城市的天气."""
    return f"今天 {city} 是晴天!"


DB_URI = "postgresql://postgres:123456@localhost:5432/langgraph_demo?sslmode=disable"


@tool(description="获取用户信息")
def get_user_info(config: RunnableConfig) -> dict:
    """获取用户信息"""
    user_id = config["configurable"]["user_id"]
    return {
        "user_id": user_id,
        "name": "张三",
        "age": 18
    }


with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer
):
    # checkpointer.setup()
    # store.setup()
    graph = create_agent(
        llm,
        tools=[get_weather, overture_tool, get_user_name, greet_user],
        # middleware=[
        #     dynamic_system_prompt,
        # ]
        system_prompt="你是一个智能助手,尽可能调用工具回答用户的问题",
        checkpointer=checkpointer,
        store=store,
    )
    config = {
        "configurable": {"thread_id": "001"}
    }
    state = graph.get_state(config)
    print(state)
    res1 = graph.invoke({"messages": [{"role": "user", "content": "今天广州天气怎么样？"}]}, config)
    print(res1)
    res2 = graph.invoke({"messages": [{"role": "user", "content": "石家庄呢？"}]}, config)
    print(res2)

```





# 项目实战

## 掌柜问数（TestToSQL）







