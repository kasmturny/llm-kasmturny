<template>
<div class="main-content1">
    <!-- 固定左侧的导航栏 -->
            <div class="image-container">
                    <!-- 这里放置图片 -->
                    <img src="../assets/biggest_image/8d889a2da6cc4307b44c7e72c30bb207.webp.webp" 
                    alt="描述性文本">
                    <!-- 文字覆盖在图片上 -->
                    <div class="text-over-image">
                            <h2>这里是优酷</h2>
                            <p>你可以在这里看到很多有趣的视频</p>
                    </div>
                    <button @click="goToVideos" class="jump-button">优酷工具</button>
                    <button @click="toggleStyles" class="jump-button1">切换样式</button>                
            </div>
</div>
<div class="main-content2">
            <div class="home-page">
                        <div class="container-title">
                        <h2>为你推荐</h2>
                        </div>
                        <div class="carousel">
                            <ul class="content-list" ref="contentList">
                                <li v-for="(item, index) in items3" :key="index" class="content-item">
                                    <img :src="item.image" alt="" />
                                    <h3>{{ item.title }}</h3>
                                </li>
                            </ul>
                        </div>

                        <div class="container-title">
                            <h2>热门内容</h2>
                        </div>
                        <div class="carousel">
                            <ul class="content-list" ref="contentList">
                                <li v-for="(item, index) in items4" :key="index" class="content-item">
                                    <img :src="item.image" alt="" />
                                    <h3>{{ item.title }}</h3>
                                </li>
                            </ul>
                        </div>

                        <div class="container-title">
                            <h2>历史播放</h2>
                        </div>
                        <div class="carousel">
                            <ul class="content-list" ref="contentList">
                                <li v-for="(item, index) in items2" :key="index" class="content-item">
                                    <router-link :to="{ path: '/youtuplayer/:videoId', query: { videoId: item.videoId } }">
                                    <img :src="item.image" alt="" />
                                    <h3>{{ item.title }}</h3>
                                    </router-link>
                                </li>
                            </ul>
                        </div>
            </div>
</div>


</template>

<script>
export default {
data() {
    return {
    items2:[
        { image: require('../assets/video1.png'), title: '本地视频1',videoId :require('../assets/video1.mp4')},
        {
        image: require('../assets/item2/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.webp'),
        title: '网络视频1',
        videoId: 'https://cesium.com/public/SandcastleSampleData/big-buck-bunny_trailer.mp4'
        },
        {
        image: require('../assets/item2/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.webp'),
        title: '网络视频2',
        videoId: 'https://www.w3schools.com/html/movie.mp4'
        },
        {
        image: require('../assets/item2/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.webp'),
        title: '网络视频3',
        videoId: 'https://stream7.iqilu.com/10339/upload_transcode/202002/18/20200218114723HDu3hhxqIT.mp4'
        },
        {
        image: require('../assets/item2/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.webp'),
        title: '网络视频4',
        videoId: 'https://stream7.iqilu.com/10339/article/202002/17/778c5884fa97f460dac8d90493c451de.mp4'
        },
        {
        image: require('../assets/item2/call_74c7bd20-aaa9-11ef-9b57-9a755667bb32_0.webp'),
        title: '网络视频5',
        videoId: 'https://stream7.iqilu.com/10339/article/202002/18/02319a81c80afed90d9a2b9dc47f85b9.mp4'
        },
    ],
    items3:[
        { image: require('../assets/item3/1.jpg'), title: '白夜破晓' },
        { image: require('../assets/item3/2.jpg'), title: '鬼服1秒狂暴' },
        { image: require('../assets/item3/3.jpg'), title: '珠帘玉幕' },
        { image: require('../assets/item3/4.jpg'), title: '装备全靠打' },
        { image: require('../assets/item3/5.jpg'), title: '蜀锦人家' },
        { image: require('../assets/item3/6.jpg'), title: '奔跑吧3茶马古道' },
        { image: require('../assets/item3/7.jpg'), title: '山海伏魔录' },
        { image: require('../assets/item3/8.jpg'), title: '百家新说' },
        { image: require('../assets/item3/9.jpg'), title: '学习时刻' },
        { image: require('../assets/item3/10.jpg'), title: '道士月灵' },
    ],
    items4:[
        { image: require('../assets/item4/11.jpg'), title: '名侦探柯南' },
        { image: require('../assets/item4/12.jpg'), title: '汪汪队立大功' },
        { image: require('../assets/item4/13.jpg'), title: '故乡的泥土' },
        { image: require('../assets/item4/14.jpg'), title: '珠帘玉幕' },
        { image: require('../assets/item4/15.jpg'), title: '白夜破晓' },
        { image: require('../assets/item4/16.jpg'), title: '超级宝贝' },
        { image: require('../assets/item4/17.jpg'), title: '我们恋爱吧' },
        { image: require('../assets/item4/18.jpg'), title: '沉香如梦夜不寒' },
    ],
    currentStyle: 'youtuhome2.css' // 默认样式文件
    };
},
methods: {
    goToVideos() {
        this.$router.push('/yoututools');
    },
    toggleStyles() {
      // 切换样式文件
      this.currentStyle = this.currentStyle === 'youtuhome1.css' ? 'youtuhome2.css' : 'youtuhome1.css';
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
},
beforeUnmount() {
    // 组件销毁前移除<link>元素
    let linkElement = document.getElementById('app-style');
    if (linkElement) {
      document.head.removeChild(linkElement);
    }
},
mounted() {
    this.addStyleLink(this.currentStyle);
},
}
</script>

<style scoped>

</style>
