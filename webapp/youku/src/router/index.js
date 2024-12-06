import { createRouter, createWebHistory } from 'vue-router';
import YoutuTools from '../components/YoutuTools.vue'; // 假设这是你对应的组件
import YoutuHome from '../components/YoutuHome.vue'; // 假设这是你对应的组件
import YoutuPlayer from '../components/YoutuPlayer.vue'; // 假设这是你对应的组件

const routes = [
  {
    path: '/',
    redirect: '/youtuhome'
  },
  {
    path: '/youtuhome',
    name: 'YoutuHome',
    component: YoutuHome
  },
  {
    path: '/yoututools',
    name: 'YoutuTools',
    component: YoutuTools
  },
  {
    path: '/youtuplayer/:videoId',
    name: 'YoutuPlayer',
    component: YoutuPlayer
  },
  // ...其他路由
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
});

export default router;