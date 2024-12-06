<template>
    <div class="video-player">
        <video ref="videoPlayer" :src="videoSrc" controls @click="togglePlay">
            您的浏览器不支持视频播放，请升级浏览器。
        </video>
        <div class="controls">
            <button @click="togglePlay">{{ isPlaying ? '暂停' : '播放' }}</button>
            <button @click="toggleFullScreen">全屏</button>
            <input type="range" min="0" max="1" step="0.1" v-model="volume" @input="setVolume" />
        </div>
    </div>
    </template>
    
    <script>
    export default {
        name: 'YoutuPlayer',
        data() {
            return {
                videoSrc: '', // 初始化为空，稍后根据参数设置
                isPlaying: false,
                volume: 1
            };
        },
        mounted() {
            // 当组件挂载后，从路由参数中获取videoId并设置视频源
            this.videoSrc = this.$route.query.videoId;
            console.log(this.videoSrc);
        },
        methods: {
            togglePlay() {
                if (this.$refs.videoPlayer.paused) {
                    this.$refs.videoPlayer.play();
                    this.isPlaying = true;
                } else {
                    this.$refs.videoPlayer.pause();
                    this.isPlaying = false;
                }
            },
            toggleFullScreen() {
                const videoElement = this.$refs.videoPlayer;
                if (videoElement.requestFullscreen) {
                    videoElement.requestFullscreen();
                } else if (videoElement.mozRequestFullScreen) { /* Firefox */
                    videoElement.mozRequestFullScreen();
                } else if (videoElement.webkitRequestFullscreen) { /* Chrome, Safari & Opera */
                    videoElement.webkitRequestFullscreen();
                } else if (videoElement.msRequestFullscreen) { /* IE/Edge */
                    videoElement.msRequestFullscreen();
                }
            },
            setVolume() {
                this.$refs.videoPlayer.volume = this.volume;
            }
        }
    };
    </script>
    
    <!-- 样式保持不变 -->
    

<style scoped>
.video-player {
width: 1024px; /* 视频宽度 */
height: 768px;
margin: 20px auto; /* 居中显示 */
}
video {
width: 100%;
height: auto;
}
.controls {
display: flex;
justify-content: space-between;
align-items: center;
padding: 10px;
}
button {
cursor: pointer;
padding: 5px 10px;
background-color: #ddd;
border: none;
border-radius: 5px;
}
input[type="range"] {
vertical-align: middle;
}
</style>
  