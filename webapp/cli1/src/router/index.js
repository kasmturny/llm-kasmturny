// router/index.js
import { createRouter, createWebHistory } from 'vue-router';
import Home from '../components/Home.vue';
import ChatPink from '../components/ChatPink.vue';

const routes = [
    {
      path: '/',
      redirect: '/home'
    },
    {
    path: '/chatpink',
    name: 'chatpink',
    component: ChatPink
    },
    {
    path: '/home',
    name: 'home',
    component: Home
    }
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
});

export default router;
