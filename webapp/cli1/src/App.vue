<template>
  <div id="app">
    <!-- 使用Vue Router进行页面跳转 -->
    <button @click="goToHome">Go to Home</button>
    <button @click="goToChatPink">Go to ChatPink</button>
    <!-- 路由出口 -->
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App',
  methods: {
    // 定义跳转到Home页面的方法
    goToHome() {
      this.$router.push('/home');
    },
    // 定义跳转到ChatPink页面的方法
    goToChatPink() {
      this.$router.push('/chatpink');
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
      text-align: left; /* 将文本对齐方式改为左对齐 */
    }

    button {
      display: inline-block; /* 确保按钮在同一行显示 */
      margin: 10px; /* 按钮的外边距 */
      padding: 10px 20px; /* 按钮的内边距 */
      font-size: 16px;
      cursor: pointer;
    }
</style>



