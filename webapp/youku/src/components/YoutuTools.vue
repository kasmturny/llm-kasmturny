<template>
<div class="main-content-tools">
    
    <div class="change-style">
        <button @click="toggleStyles">切换样式</button>
    </div>

    <!-- 输入和按钮部分 -->
    <div class="yoututools-mainbox">
        <div class="main-box-head-ico"></div>
        <div class="input-and-button">
            <input type="text" v-model="numberInput" placeholder="输入一个数字" @keyup.enter="calculateSquare" />
            <button @click="calculateSquare">计算平方</button>
        </div>

        <div class="result-box">
            <p v-if="error" style="color: red;">{{ errorMessage }}</p>
            <p v-else-if="squareValue !== null">{{ squareValue }}</p>
            <p v-else>请输入数字并计算平方</p>
        </div>
        
            <!-- 时间监听部分 -->
        <div class="time-monitor">
                <div class="time-display">
                    <p>当前时间: {{ currentTime }}</p>
                <div class="tuzi-image">
                    <img src="https://sfile.chatglm.cn/testpath/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.png?image_process=format,webp" 
                    alt="tuzi" />   
                </div>
                </div>
                <div class="recorded-times">
                    <p>记录的时间:</p>
                    <div class="recorded-times-list">
                        <ul>
                            <li v-for="time in recordedTimes" :key="time">{{ time }}</li>
                        </ul>
                    </div>
                </div>
        </div>
    </div>
</div>
</template>

<script>
export default {
data() {
return {
    numberInput: '',
    squareValue: null,
    error: false,
    errorMessage: '请输入一个有效的数字',
    currentTime: '',
    recordedTimes: [],
    currentStyle: 'yoututools2.css' // 默认样式文件
};
},
methods: {
    toggleStyles() {
      // 切换样式文件
      this.currentStyle = this.currentStyle === 'yoututools1.css' ? 'yoututools2.css' : 'yoututools1.css';
      this.addStyleLink(this.currentStyle);
    },
    addStyleLink(styleName) {
      // 检查是否已存在<link>元素
      let linkElement = document.getElementById('app-style');
      if (linkElement) {
        // 如果存在，则更新href
        linkElement.href = `/css/${styleName}`;
      } else {
        // 如果不存在，则创建<link>元素并添加到<head>
        linkElement = document.createElement('link');
        linkElement.id = 'app-style';
        linkElement.rel = 'stylesheet';
        linkElement.href = `/css/${styleName}`;
        document.head.appendChild(linkElement);
      }
    },
    calculateSquare() {
    // 正则表达式匹配整数或浮点数
    const numberRegex = /^-?\d+(\.\d+)?$/;
    if (numberRegex.test(this.numberInput)) {
        const number = parseFloat(this.numberInput);
        this.error = false;
        this.squareValue = number * number;
    } else {
        this.error = true;
        this.squareValue = null;
    }
},

        updateTime() {
        const now = new Date();
        this.currentTime = now.toLocaleTimeString();
        const seconds = now.getSeconds();
        if ([0,15,45].includes(seconds)) {
            this.recordedTimes.push(this.currentTime);
        }
}
},
mounted() {
    this.currentTime = new Date().toLocaleTimeString();
    this.timer = setInterval(this.updateTime, 1000);
    this.addStyleLink(this.currentStyle);
},
beforeUnmount() {
    clearInterval(this.timer);
    let linkElement = document.getElementById('app-style');
    if (linkElement) {
      document.head.removeChild(linkElement);
    }
}
};
</script>

<style scoped>
</style>

