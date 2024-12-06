<template>
<div class="chat-container">
    <div class="chat-messages">
    <div v-for="message in messages" :key="message.id" class="message">
        <strong>{{ message.sender }}</strong>: {{ message.text }}
    </div>
    </div>
    <div class="setcontainer">
    <input type="text" v-model="newMessage" @keyup.enter="sendMessage" placeholder="输入消息..." />
    <button @click="newChat">新对话</button>
    <button @click="sendMessage">发送</button>
    </div>
</div>
</template>

<script>
export default {
data() {
    return {
    newMessage: '',
    messages: [],
    messageHistory: [
        { role: "system", content: "你是一个看过很多影视剧的AI助手，请根据用户的提问给出准确的回答。" }
    ],
    apiKey: "sk-t6b7ea9aQo6oVReey1uOj8l3jpCGngmesP62g8uslrAXh0BY",
    customApiUrl: "https://api.chatanywhere.com.cn/v1"
    };
},
methods: {
    async sendRequest() {
    const response = await fetch(`${this.customApiUrl}/chat/completions`, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: this.messageHistory,
        temperature: 0,
        max_tokens: 1000,
        })
    });

    const result = await response.json();

    if (result.choices && result.choices.length > 0) {
        const newMessage = result.choices[0].message;
        this.messageHistory.push(newMessage);
        this.messages.push({
        id: Date.now(),
        sender: 'AI',
        text: newMessage.content
        });
        this.saveHistory();
    } else {
        console.log("No message content received.");
    }
    },
    async sendMessage() {
    if (this.newMessage.trim() === '') {
        return;
    }
    this.messages.push({
        id: Date.now(),
        sender: 'User',
        text: this.newMessage
    });
    this.messageHistory.push({ role: "user", content: this.newMessage });
    this.saveHistory();
    
    await this.sendRequest();
    
    this.newMessage = '';
    },
    newChat() {
    this.messageHistory = [
        { role: "system", content: "你是一个看过很多影视剧的AI助手，请根据用户的提问给出准确的回答。" }
    ];
    this.messages = [];
    this.saveHistory();
    },
    saveHistory() {
    localStorage.setItem('messageHistory', JSON.stringify(this.messageHistory));
    localStorage.setItem('messages', JSON.stringify(this.messages));
    },
    loadHistory() {
    const savedHistory = JSON.parse(localStorage.getItem('messageHistory'));
    const savedMessages = JSON.parse(localStorage.getItem('messages'));
    if (savedHistory) {
        this.messageHistory = savedHistory;
    }
    if (savedMessages) {
        this.messages = savedMessages;
    }
    }
},
created() {
    this.loadHistory();
}
};
</script>

<!-- 样式保持不变 -->




<style scoped>

.chat-container {
height: 60vh;
margin-top: 20px;
margin-bottom: 20px;
padding: 10px;
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