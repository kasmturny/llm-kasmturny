<template>
  <div class="chat-container">
    <div class="chat-messages">
      <div v-for="message in messages" :key="message.id" class="message">
        <strong>{{ message.sender }}</strong>: {{ message.text }}
      </div>
    </div>
    <div class="setcontainer">
      <input type="text" v-model="newMessage" @keyup.enter="sendMessage" placeholder="输入消息..." />
      <button @click="sendMessage">发送</button>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      newMessage: '',
      newRespose:'',
      messages: []
    };
  },
  methods: {
    sendMessage() {
      if (this.newMessage.trim() === '') {
        return;
      }
      const messagesend = {
        id: Date.now(),
        sender: 'User',
        text: this.newMessage
      };
      this.messages.push(messagesend);

      this.sendDataToFastApi().then(() => {
        const messageincept = {
        id: Date.now(),
        sender: 'Assiants',
        text: this.newRespose
        };
        this.messages.push(messageincept);
        this.newRespose = '';
        this.newMessage = '';
      });
      // 清空输入框,以及回复
      // 这里可以添加发送消息到服务器的代码
    },
    async sendDataToFastApi() {
      const dataToPost = {
        question: this.newMessage,
      };
      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(dataToPost),
        });
        // 传递dataToPost给服务器，然后等待返回的response

        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }

        // 把返回的数据json化
        const responseData = await response.json();
        this.newRespose = responseData.response;
        console.log(this.newRespose)
        // 假设FastAPI返回的数据结构中有一个message字段
      }
      catch (error) {
        console.error('请求失败:', error);
        this.newRespose = '请求失败，请稍后再试';

      }
    },
  }
};
</script>


<style scoped>

.chat-container {
  height: 80vh;
  display: flex;
  flex-direction: column;
  border: 1px solid #000000;
  border-radius: 10px;
  overflow: hidden;
}

.chat-messages {
  height: 100%;
  padding: 10px;
  background-color: #f5f5f5;
  overflow-y: auto;
  background-image: url('https://sfile.chatglm.cn/pic_cache/66db376e44b2c1ab6f70b921/1fjb6j.png'); /* 添加这行代码 */
  background-size: cover; /* 背景图片覆盖整个容器 */
  background-position: center; /* 背景图片居中显示 */
  background-repeat: no-repeat; /* 背景图片不重复 */
}

.message {
  margin: 5px 0;
  padding: 5px;
  border-radius: 5px;
  max-width: 100%;
  word-wrap: break-word;
}

.message strong {
  font-weight: bold;
}

.message:nth-child(even) {
  background-color: #e9e9e9;
  align-self: flex-start;
}

.message:nth-child(odd) {
  background-color: #d4e7fa;
  align-self: flex-end;
}

.setcontainer {
  width: 100%;
  background-image: url('https://sfile.chatglm.cn/pic_cache/66db376e44b2c1ab6f70b921/1fjb6j.png'); /* 添加这行代码 */
  background-size: cover; /* 背景图片覆盖整个容器 */
  background-position: center; /* 背景图片居中显示 */
  background-repeat: no-repeat; /* 背景图片不重复 */
}

input[type="text"] {
  width: calc(100% - 150px);
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin: 10px 0px 10px 10px; /* 上 右 下 左 */
  float: left;
}

button {
  width: 80px;
  padding: 10px;
  border: none;
  border-radius: 5px;
  background-color: #0084ff;
  color: white;
  cursor: pointer;
  float: right;
  margin: 10px 10px 10px 0px; /* 上 右 下 左 */
}

button:hover {
  background-color: #006ae8;
}
</style>
