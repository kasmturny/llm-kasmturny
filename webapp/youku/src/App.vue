<template>
<div>
    <div class="header">
        <img src=https://img.alicdn.com/imgextra/i1/O1CN01180Rqd1u3Lo8PdgSs_!!6000000005981-55-tps-213-72.svg
        alt="网站图标" class="header-icon">
        <input type="text" placeholder="搜索内容" class="search-input">
        <button class="search-button">搜索</button>
        <div class="data-display1">{{ currentDate }}</div>
        <div class="time-display1">{{ currentTime }}</div>
        <div class="button-group">
            <button class="register-button">注册</button>
            <button class="login-button">登录</button>
        </div>
    </div>
    <div class="sidebar">
            <ul>
            <li><router-link to="/">首页</router-link></li>
            <li><a href="#">电视剧</a></li>
            <li><a href="#">电影</a></li>
            <li><a href="#">综艺</a></li>
            <li><a href="#">动漫</a></li>
            <li><a href="#">少儿</a></li>
            <li><a href="#">VIP会员</a></li>
            <li><a href="#">纪录片</a></li>
            <li><a href="#">人文</a></li>
            <li><a href="#">体育</a></li>
            <li><a href="#">游戏</a></li>
            <li><a href="#">新知</a></li>
            <li><a href="#">学堂</a></li>
            <li><a href="#">音乐</a></li>
            <li><a href="#">直播</a></li>
            <li><a href="#">娱乐</a></li>
            <li><a href="#">搞笑</a></li>
            <li><a href="#">生活</a></li>
            <li><a href="#">电竞</a></li>
            <li><a href="#">亲子</a></li>
            <li><a href="#">公益</a></li>
            <li><a href="#">无障碍</a></li>
            <li><a href="#">教育</a></li>
            <li><a href="#">财经</a></li>
            <li><a href="#">淘好片</a></li>
            <!-- 更多导航项 -->
            </ul>
    </div>
    <div class="football">
        <float-ball @toggle-popup="isPopupVisible = !isPopupVisible" />
        <!-- 弹出的组件页面 -->
        <div v-if="isPopupVisible" class="popup">
        <!-- 组件页面的内容 -->
        <div class = "popup-header">你好我叫小酷，是你的AI助手，你可以向我提问</div>
        <youtu-chat />
        <button @click="isPopupVisible = false" class = "popup-close-button">关闭</button>
        </div>
    </div>
    <div class="carousel">
    <div class="container">
        <router-view/>
    </div>
    </div>
</div>
</template>

<script>
import FloatBall from './components/FloatBall.vue';
import YoutuChat from './components/YoutuChat.vue';
export default {
name: 'App',
components: {
FloatBall,
YoutuChat
},
data() {
    return {
    headerIconSrc: 'https://sfile.chatglm.cn/testpath/gen-1732473042580276180_0.png',
    currentTime: '',
    currentDate: '',
    isPopupVisible: false,
    };
},
methods: {
    updateTime() {
        const now = new Date();
        const year = now.getFullYear();
        const month = (now.getMonth() + 1).toString().padStart(2, '0'); // 月份是从0开始的
        const day = now.getDate().toString().padStart(2, '0');
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        const seconds = now.getSeconds().toString().padStart(2, '0');
        // 格式化日期和时间
        this.currentDate = `当前日期：${year}-${month}-${day}`;
        this.currentTime = `当前时间:${hours}:${minutes}:${seconds}`;
    }
},
mounted() {
    this.updateTime();
    setInterval(this.updateTime, 1000);
}
}
</script>

<style scoped>
#app {
font-family: Avenir, Helvetica, Arial, sans-serif;
-webkit-font-smoothing: antialiased;
-moz-osx-font-smoothing: grayscale;
color: #2c3e50;
margin-top: 0;
background-color: #000000;
                                                                                                  }
.header {
position: fixed;
top: 0;
left: 0;
width: 100%;
height: 50px;
background-color: #3a3b3f;
z-index: 1000;
padding: 10px 0;
box-shadow: 0 4px 0px rgba(0, 0, 0, 0.1);
display: flex;
align-items: center;
                                                                                                  }

                .header-icon {
                height: 40px; /* 根据你的图标大小调整 */
                width: auto; /* 保持图标比例 */
                margin-left: 30px; /* 左侧外边距 */
                }

                .search-input {
                flex: 0 0 30%; /* 设置flex属性，使输入框占据父元素的30% */
                padding: 10px;
                margin: 0 30px;
                border: 1px solid #ccc;
                border-radius: 4px;
                }

                .search-button {
    height: 40px;
    padding: 10px 20px;
    border: none;
    background-color: #007bff;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
    overflow: hidden; /* 添加这行代码 */
    text-overflow: ellipsis; /* 添加这行代码 */
    white-space: nowrap; /* 添加这行代码 */
}

.search-button:hover {
    background-color: #0056b3;
}

.data-display1 {
    background-color: #0c4c8a;
    margin-left: 20px;
    color: #fff;
    padding: 10px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    overflow: hidden; /* 添加这行代码 */
    text-overflow: ellipsis; /* 添加这行代码 */
    white-space: nowrap; /* 添加这行代码 */
}

.time-display1 {
    background-color: #0c4c8a;
    color: #fff;
    margin-left: 20px;
    padding: 10px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    overflow: hidden; /* 添加这行代码 */
    text-overflow: ellipsis; /* 添加这行代码 */
    white-space: nowrap; /* 添加这行代码 */
}

.button-group {
    height: 40px;
    margin-left: auto; /* 将按钮组靠右对齐 */
    margin-right: 20px;
    display: flex;
}

.register-button, .login-button, .change-color {
    padding: 10px 20px;
    margin-left: 10px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.register-button {
    background-color: #28a745;
    color: white;
    overflow: hidden; /* 添加这行代码 */
    text-overflow: ellipsis; /* 添加这行代码 */
    white-space: nowrap; /* 添加这行代码 */
}

.register-button:hover {
    background-color: #218838;
}

.login-button {
    background-color: #007bff;
    color: white;
    overflow: hidden; /* 添加这行代码 */
    text-overflow: ellipsis; /* 添加这行代码 */
    white-space: nowrap; /* 添加这行代码 */
}

.login-button:hover {
    background-color: #0056b3;
}

.sidebar {
position: fixed;
left: 0;
top: 70px; /* 根据顶部搜索栏的高度调整 */
width:100px;
text-align: center;
height: 100%;
background-color: transparent; /* 将背景颜色改为透明 */
overflow-y: auto;
box-shadow: 4px 0px 4px rgba(0, 0, 0, 0.1);
z-index: 999;
                                                                                                 }
    
            .sidebar::-webkit-scrollbar {
                display: none;
            }

            .sidebar ul {
            list-style: none;
            padding: 0;
            }

            .sidebar ul li a {
            display: block;
            padding: 15px;
            color: #ffffff;
            text-decoration: none;
            }

            .sidebar ul li a:hover {
            background-color: #ddd;
            }
.float-ball {
position: fixed;
bottom: 20px;
right: 20px;
width: 60px;
height: 60px;
border-radius: 50%;
background-color: #007bff;
color: white;
text-align: center;
line-height: 60px;
font-size: 24px;
cursor: pointer;
box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
z-index: 999;
transition: transform 0.3s ease-in-out;
                                                                                                   }

            .float-ball:hover {
            transform: scale(1.1);}
/* 弹出的组件页面的样式 */
.popup {
position: fixed; /* 使用fixed定位使弹出层相对于视口固定 */
top: 50%; /* 初始位置垂直居中 */
left: 50%; /* 初始位置水平居中 */
transform: translate(-50%, -50%); /* 使用transform调整使其真正居中 */
width: 80%; /* 宽度设置为视口宽度的80%，可以根据需要调整 */
max-width: 1000px; /* 最大宽度限制，防止在宽屏上过大 */
background-color: #ffffff; /* 背景颜色 */
border-radius: 8px; /* 边框圆角 */
box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* 阴影效果 */
z-index: 2000; /* 高层级，确保在悬浮球之上 */
padding: 20px; /* 内边距 */
box-sizing: border-box; /* 盒模型，包括padding和border在内的总宽高 */
                                                                                                 }

                /* 组件页面的内容样式 */
                .popup-header {
                margin-bottom: 20px; /* 与关闭按钮的间距 */
                text-align: center; /* 文本居中 */
                font-size: 18px; /* 字体大小 */
                color: #333333; /* 字体颜色 */
                }

                /* 关闭按钮的样式 */
                .popup-close-button {
                display: block; /* 块级元素，使其宽度充满容器 */
                width: 100%; /* 宽度充满父容器 */
                padding: 10px; /* 内边距 */
                border: none; /* 无边框 */
                background-color: #f44336; /* 背景颜色 */
                color: white; /* 字体颜色 */
                border-radius: 4px; /* 边框圆角 */
                cursor: pointer; /* 鼠标样式为指针 */
                font-size: 16px; /* 字体大小 */
                transition: background-color 0.3s; /* 过渡效果 */
                }

                /* 关闭按钮悬停状态样式 */
                .popup button:hover {
                background-color: #d32f2f; /* 悬停时背景颜色加深 */
                }
.container {
margin-top: 70px;
margin-left: 0;
left: 0;
top: 0;
height: calc(100% - 70px); 
width: 100%;
position: absolute;
background-color: #333;
overflow-y: auto;
background-image: url('../public/css/gg.png'); /* 添加背景图片 */
background-repeat: no-repeat; /* 背景图片不重复 */
background-size: cover; /* 背景图片覆盖整个元素区域 */
background-position: center top; /* 背景图片从顶部居中显示 */
                                                                                             }

                                                                                             .carousel111 {
                            overflow-x: auto;
                            scrollbar-width: thin; /* 设置滚动条的宽度（thin, auto, or none） */
                            scrollbar-color: #888 #000000; /* 设置滚动条滑块和轨道的颜色 */
                          }

</style>
