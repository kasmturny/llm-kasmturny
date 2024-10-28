// router/index.js
import { createRouter, createWebHistory } from 'vue-router';
import Home from '../components/Home.vue';
import ChatBot from '../components/ChatBot.vue';

const routes = [
    {
      path: '/',
      redirect: '/home'
    },
    {
    path: '/chatbot',
    name: 'chatbot',
    component: ChatBot
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
