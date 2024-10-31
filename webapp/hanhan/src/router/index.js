// router/index.js
import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '../components/HomePage.vue';
import ChatBot from '../components/ChatBot.vue';

const routes = [
    {
      path: '/',
      redirect: '/homepage'
    },
    {
    path: '/chatbot',
    name: 'chatbot',
    component: ChatBot
    },
    {
    path: '/homepage',
    name: 'homepage',
    component: HomePage
    }
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
});

export default router;
