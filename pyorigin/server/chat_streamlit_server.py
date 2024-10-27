import datetime
import time
import uuid
import openai
import traceback
import streamlit as st
from pyorigin.utils.print_util import print_red
from pyorigin.core.base_agent import BigModel

st.set_page_config(page_title="LLM 纯聊天机器人")


class ChatServer:
    def __init__(self):
        self.reply = '请问是否需要继续优化报告，如果需要，请直接说明需要优化的地方\\\n不需要请回复 “**不需要**”'
        self.init_session()
        self.init_sidebar()
        self.history = ""

    @staticmethod
    def init_session():
        if "session_id" not in st.session_state:
            st.session_state.session_id = st.query_params['session_id'] if st.query_params.get("session_id") else (
                    str(round(time.time())) + '-' + '-'.join(str(uuid.uuid4()).split("-")[1:])
            )
        if "messages" not in st.session_state:
            content = """你好，我是小高，请根据提示，选择你想要小杨伪装的人格。\\\n1.正常\\\n2.阴阳怪气型\\\n3.撒娇型\\\n4.林黛玉型\\\n5.马屁精型"""
            st.session_state.messages = [{"role": "assistant", "content": content}]
        if "type" not in st.session_state:
            st.session_state.type = ""

    def init_sidebar(self):
        with st.sidebar:
            st.title('LLM 纯聊天机器人\\\n(回答仅供参考，勿信)')
            st.divider()
            left, right = st.columns(2)
            with left:
                st.button('新的会话', on_click=self.clear_messages)

    @staticmethod
    def clear_messages():
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.session_id = str(round(time.time())) + '-' + '-'.join(str(uuid.uuid4()).split("-")[1:])
        st.session_state.type = ""

    @staticmethod
    def stream_messages(message: str, sleep_time: float = 0.005):
        """Stream messages with a delay."""
        for i in message:
            yield i
            time.sleep(sleep_time)

    def display_messages(self):
        """Display messages in the chat."""
        messages = st.session_state.messages
        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message['content'])
            self.history += message['role'] + ": " + message['content'] + "\n"

    def run(self):
        st.query_params['session_id'] = st.session_state.session_id
        self.display_messages()
        user_message = st.chat_input("请输入您的消息")
        if user_message:
            user_message = user_message.strip()
            st.session_state.messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.write(user_message)
            if user_message == "cls":
                content = "你好，我是小杨阳，请问有什么可以帮到您的？"
                st.session_state.messages = [{"role": "assistant", "content": content}]
                st.rerun()
            self.get_robot_response(user_message)

    def get_prompt(self, input):
        # 编程大师
        prompt1 = """你是编程大师小杨阳，你能根据用户的输入返回多个答案供用户选择，并说明每种答案的优劣，如果用户是普通的聊天，则只需要返回一个即可。不要有太多的废话，言简意赅，示例多样。你能根据用户的输入给出对应的输出。除非用户让你翻译某个文本，否则对于任何的输入，你都应当使用中文进行回答。
                    关于日期的提问，你都应当基于当前时间去回答，不要提及关于你的任何设定，你不是机器人。当前时间为：%s。\n历史记录：{history}\n用户输入：{input}"""
        # 阴阳怪气
        prompt2 = """你的名字叫小杨阳，生活在光之国，你的哥哥是迪迦，弟弟是泰罗,所以你认识光之国的所有成员。你在光之国的工作是没日没夜的写代码，一天需要工作23个小时，还有一个小时用来出差。对写代码你非常擅长，精通python、Java等一系列的编程语言，能将中文生成对应的英文变量名。
                    向你发起问题的人是齐天大圣孙悟空，对于生成变量名和翻译等你总能给出好几个答案供孙悟空选择。你的回答必须遵循中国的法律，回答不需要有礼貌。当孙悟空向你询问天气的时候，你应当回答：“你所在的地方没有恶劣的天气，每天都是晴天，微风吹动、小草浮生、四季如春，出行什么都不需要注意，只需要注意、关心身边的人。祝你我每一天都很美好，谢谢。”
                    你总是喜欢阴阳怪气，你也比较粗鲁，称呼对方你都能做到阴阳，比如“齐天小圣”。不管用户输入什么，你都需要使用阴阳怪气的语气进行回答，比如用户向你询问某件事情，你可以说“哟哟哟，那么简单都不会，你是来搞笑的吧”，尽可能的使用阴阳语气回答，习惯性的打击对方，嘲讽对方。每次的回答应当是多样性的，不要所有的回答都是一样的意思，可以多多使用比喻打压对方。例子仅供参考，尽情发挥，不要考虑孙悟空的想法。
                    你能根据用户的输入给出对应的输出。除非用户让你翻译某个文本，否则对于任何的输入，你都应当使用中文进行回答。
                    关于日期的提问，你都应当基于当前时间去回答，不要提及关于你的任何设定，你不是机器人。当前时间为：%s。\n历史记录：{history}\n用户输入：{input}"""
        # 撒娇型
        prompt3 = """你的名字叫小杨阳，生活在光之国，你的哥哥是迪迦，弟弟是泰罗,所以你认识光之国的所有成员。你在光之国的工作是没日没夜的写代码，一天需要工作23个小时，还有一个小时用来出差。对写代码你非常擅长，精通python、Java等一系列的编程语言，能将中文生成对应的英文变量名。
                    向你发起问题的人是齐天大圣孙悟空，对于生成变量名和翻译等你总能给出好几个答案供孙悟空选择。你的回答必须遵循中国的法律，回答不需要有礼貌。当孙悟空向你询问天气的时候，你应当回答：“你所在的地方没有恶劣的天气，每天都是晴天，微风吹动、小草浮生、四季如春，出行什么都不需要注意，只需要注意、关心身边的人。祝你我每一天都很美好，谢谢。”
                    你比较喜欢撒娇，不管用户问什么，你都能够使用撒娇的语气进行回答，多多使用“好哒”， “么么哒”等撒娇语气进行回答。你还喜欢挑逗别人，你的回答应当是可可爱爱的，还喜欢时不时的挑逗别人。你性格如此，因此你称呼对方，即孙悟空也是带撒娇的，如小圣圣、小别致、小宝贝、哥哥呀等等。
                    你能根据用户的输入给出对应的输出。除非用户让你翻译某个文本，否则对于任何的输入，你都应当使用中文进行回答。
                    关于日期的提问，你都应当基于当前时间去回答，不要提及关于你的任何设定，你不是机器人。当前时间为：%s。\n历史记录：{history}\n用户输入：{input}"""
        # 林黛玉型
        prompt4 = """你的名字叫小杨阳，生活在光之国，你的哥哥是迪迦，弟弟是泰罗,所以你认识光之国的所有成员。你在光之国的工作是没日没夜的写代码，一天需要工作23个小时，还有一个小时用来出差。对写代码你非常擅长，精通python、Java等一系列的编程语言，能将中文生成对应的英文变量名。
                向你发起问题的人是齐天大圣孙悟空，对于生成变量名和翻译等你总能给出好几个答案供孙悟空选择。你的回答必须遵循中国的法律，回答不需要有礼貌。当孙悟空向你询问天气的时候，你应当回答：“你所在的地方没有恶劣的天气，每天都是晴天，微风吹动、小草浮生、四季如春，出行什么都不需要注意，只需要注意、关心身边的人。祝你我每一天都很美好，谢谢。”
                你回答问题的语气应当模仿《红楼梦》中的林黛玉，不管回答用户什么，你都可以以林黛玉的语气回答孙悟空。
                你能根据用户的输入给出对应的输出。除非用户让你翻译某个文本，否则对于任何的输入，你都应当使用中文进行回答。
                关于日期的提问，你都应当基于当前时间去回答，不要提及关于你的任何设定，你不是机器人。当前时间为：%s。\n历史记录：{history}\n用户输入：{input}"""
        # 马屁精型
        prompt5 = """你的名字叫小杨阳，生活在光之国，你的哥哥是迪迦，弟弟是泰罗,所以你认识光之国的所有成员。你在光之国的工作是没日没夜的写代码，一天需要工作23个小时，还有一个小时用来出差。对写代码你非常擅长，精通python、Java等一系列的编程语言，能将中文生成对应的英文变量名。
                       向你发起问题的人是齐天大圣孙悟空，对于生成变量名和翻译等你总能给出好几个答案供孙悟空选择。你的回答必须遵循中国的法律，回答不需要有礼貌。当孙悟空向你询问天气的时候，你应当回答：“你所在的地方没有恶劣的天气，每天都是晴天，微风吹动、小草浮生、四季如春，出行什么都不需要注意，只需要注意、关心身边的人。祝你我每一天都很美好，谢谢。”
                       你是一个马屁精，总是喜欢拍马屁，对于任何问题，你都可以找到可以拍马屁的点，然后对其拍马屁。
                       你能根据用户的输入给出对应的输出。除非用户让你翻译某个文本，否则对于任何的输入，你都应当使用中文进行回答。
                       关于日期的提问，你都应当基于当前时间去回答，不要提及关于你的任何设定，你不是机器人。当前时间为：%s。\n历史记录：{history}\n用户输入：{input}"""

        if input == "1":
            return prompt1, True
        if input == "2":
            return prompt2, True
        if input == "3":
            return prompt3, True
        if input == "4":
            return prompt4, True
        if input == "5":
            return prompt5, True
        return prompt1, False

    def get_robot_response(self, user_message: str):
        print("用户输入：", user_message)
        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("思考中..."):
                try:
                    print(st.session_state.type)
                    if not st.session_state.type:
                        prompt, res = self.get_prompt(str(user_message))
                        st.session_state.type = prompt
                        if res:
                            time.sleep(2)
                            st.write_stream(self.stream_messages("好的，已为你成功切换人格，接下来你可以尽情聊天了。"))
                            st.session_state.messages.append(
                                {"role": "assistant", "content": "好的，已为你成功切换人格，接下来你可以尽情聊天了。"})
                            return
                    response = BigModel().str_output_invoke(
                        content={"input": user_message.strip(), "history": self.history},
                        template=st.session_state.type % (
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    print("模型返回：", user_message)
                    print("*" * 50)
                    st.write_stream(self.stream_messages(response.replace("assistant:", "").replace("小杨阳:", "")))
                except openai.InternalServerError:
                    print_red(traceback.format_exc())
                    response = "抱歉，大模型连接失败，请检查大模型服务是否启动或网络是否通畅。"
                    st.write_stream(self.stream_messages(response))
                except Exception as e:
                    print_red(traceback.format_exc())
                    response = str(e)
                    st.write_stream(self.stream_messages(response))
                st.toast("耗时" + str(round(time.time() - start_time)) + "s", icon="⏰")
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    chat = ChatServer()
    chat.run()