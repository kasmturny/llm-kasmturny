<template>
  <div class="chat-container">
    <div class="chat-messages">
      <div v-for="message in messages" :key="message.id" class="message">
        <strong>{{ message.sender }}</strong>: {{ message.text }}
      </div>
    </div>
    <div class="setcontainer">
      <input type="text" v-model="newMessage" @keyup.enter="sendMessage" placeholder="输入消息..." />
      <button @click="newaChat">新对话</button>
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
    newaChat() {
            // 使用 fetch 发送 POST 请求
      fetch('http://localhost:8000/newchat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
          // 如果需要，可以在这里添加其他 headers
        },
        // 如果需要发送数据，可以在这里添加 body
        // body: JSON.stringify({ /* 数据 */ })
      })
          // eslint-disable-next-line no-unused-vars
      .then(response => {
        // 请求成功，但是我们不关心结果
      })
      .catch(error => {
        console.error('Error sending request:', error);
        // 请求失败，可以在这里处理错误
      })
      .finally(() => {
        // 无论成功还是失败，都进行路由跳转
        this.messages = [];
        this.newMessage = '';
        this.newRespose = '';
      });


    },
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
    fetchInitialData() {
      fetch('http://localhost:8000/newchat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          })
              // eslint-disable-next-line no-unused-vars
          .then(response => {
            // 请求成功，处理响应
          })
          .catch(error => {
            console.error('Error sending request:', error);
            // 请求失败，处理错误
          });
    }
  },
  mounted() {
    // 组件挂载后立即获取初始数据
    this.fetchInitialData();
  }
};
</script>


<style scoped>

.chat-container {
  height: 83vh;
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
  display: flex; /* 启用flexbox布局 */
  align-items: center; /* 垂直居中 */
}

      input[type="text"] {
        width: calc(95% - 150px); /* 你可以根据需要调整这个宽度 */
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin: 10px; /* 统一设置所有边的外边距 */
        flex-grow: 1; /* 让输入框占据剩余空间 */
      }

      button {
        padding: 10px;
        border: none;
        border-radius: 5px;
        background-color: #0084ff;
        color: white;
        cursor: pointer;
        margin: 10px; /* 统一设置所有边的外边距 */
      }

      button:hover {
        background-color: #006ae8;
      }

</style>
