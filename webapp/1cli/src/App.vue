<template>
  <div id="app">
    <!-- 按钮包裹容器 -->
    <div class="button-container">
      <button @click="goToHomePage">首页</button>
      <button @click="goToChatBot">和我交流</button>
    </div>
    <!-- 路由出口 -->
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App',
  methods: {
    goToHomePage() {
      if (this.$route.path !== '/homepage'){
        this.$router.push('/homepage')
      }
    },
    goToChatBot() {
      // 检查当前路由是否已经是 /chatbot
      if (this.$route.path !== '/chatbot') {
        this.$router.push('/chatbot').then(() => {
          // 路由跳转成功后发送 POST 请求
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
        });
      }
    }

  }
}
</script>

<style>
    #app {
      font-family: Avenir, Helvetica, Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      color: #2c3e50;
      margin-top: 0; /* 移除顶部的边距 */
    }

    .button-container {
      text-align: left; /* 按钮容器内的文本左对齐 */
      margin-bottom: 20px; /* 按钮容器底部添加一些空间 */
      border: 2px solid #383333; /* 添加一个蓝色边框 */
      border-radius: 8px; /* 添加圆角 */
      padding: 1px; /* 为容器添加内边距 */
    }


    button {
      display: inline-block; /* 确保按钮在同一行显示 */
      margin: 2px; /* 按钮的外边距 */
      padding: 10px 20px; /* 按钮的内边距 */
      font-size: 18px;
      cursor: pointer;
    }
</style>
