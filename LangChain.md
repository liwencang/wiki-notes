## LangChain



### 模型选择

优先选择langchain官网集成的模块，v1.0+  (控制起来更加的细粒度)

如果没用则选择openai的 ChatOpenAI  但是如果是深度思考模型可能没有输出深度思考的内容



### 格式化输出

```python
llm.with_structured_output(Movie)
```

类似这种pydantic的默认的格式化输出**深度思考模型和千问**之类的模型都不能用，会报错，chatgpt，deepseek的非深度思考可用

