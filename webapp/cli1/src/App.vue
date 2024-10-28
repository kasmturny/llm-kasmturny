<template>
  <div id="app">
    <!-- 按钮包裹容器 -->
    <div class="button-container">
      <button @click="goToHome">首页</button>
      <button @click="goToChatPink">和我交流</button>
    </div>
    <!-- 路由出口 -->
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App',
  methods: {
    goToHome() {
      this.$router.push('/home');
    },
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
