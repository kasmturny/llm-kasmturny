<template>
  <div id="app">
    <!-- 顶部菜单 -->
    <div class="top-nav">
      <div class="nav-item" @click="goToHomePage('/homepage')">首页</div>
      <div class="nav-item" @click="goToChatBot('/chatbot')">和我交流</div>
    </div>
    <div class="fixed-div">
      <img src="./assets/kiana.png" alt="logo" style="width: 100%; height: 100%;opacity: 0.5;">
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
      if (this.$route.path !== '/chatbot') {
        this.$router.push('/chatbot')
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
      margin-top: 0px; /* 移除顶部的边距 */
    }

    .fixed-div {
      position: fixed;       /* 使用fixed定位 */
      top: 80px;             /* 距离页面顶部50px */
      right: 40px;           /* 距离页面右侧20px */
      width: 200px;          /* div的宽度 */
      height: 200px;         /* div的高度 */
      z-index: 1000;         /* 确保div在其他内容之上 */
      overflow: hidden;
    }

    .top-nav {
      display: flex;
      justify-content: space-around;
      background-color: #1f1f1f;
      color: white;
      width: 100%; /* 设置宽度为100% */
      box-sizing: border-box; /* 包含padding和border在内的宽度 */
      border-radius: 10px;
      margin-bottom: 20px; /* 在导航栏和页面内容之间添加间隔 */
      overflow-y: auto;

    }

    .nav-item {
      padding: 20px 20px;
      cursor: pointer;
      text-align: center;
    }

    .nav-item:hover {
      background-color: #555;
    }
</style>
