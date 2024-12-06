<template>
<div class="float-ball" @click="togglePopup">
    <!-- 悬浮球的内容，可以是图标或者文字 -->
    <img src="https://sfile.chatglm.cn/testpath/gen-1732871945193635426_0.png?image_process=format,webp" alt="悬浮球">
</div>
</template>

<script>
export default {
name: 'FloatBall',
methods: {
    togglePopup() {
    this.$emit('toggle-popup');
    }
}
}
</script>

<style scoped>
.float-ball {
position: fixed;
bottom: 20px;
right: 20px;
width: 30px;
height: 30px;
background-color: #3498db;
border-radius: 50%;
cursor: pointer;
display: flex;
align-items: center;
justify-content: center;
box-shadow: 0 2px 5px rgba(0,0,0,0.2);
z-index: 1000;
}

.float-ball img {
width: 40px;
height: 40px;
overflow: hidden;
}
</style>
