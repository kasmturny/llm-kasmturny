<template>
  <div class="hello">

    <h3>VUE学习</h3>
    <div class="container">
      Vue (发音为 /vjuː/，类似 view) 是一款用于构建用户界面的渐进式 JavaScript 框架。以下是VUE学习的关键点：
      <ul>
        <li>Vue实例与数据绑定</li>
        <li>指令：v-model、v-for、v-if、v-bind、v-on</li>
        <li>计算属性和侦听器</li>
        <li>组件的定义与复用</li>
        <li>生命周期钩子</li>
      </ul>
    </div>

    <h3>JAVA学习</h3>
    <div class="container">
      Java是一种广泛使用的编程语言，特别适用于企业级应用。以下是JAVA学习的关键点：
      <ul>
        <li>Java基础知识：数据类型、运算符、控制结构</li>
        <li>面向对象编程：类、对象、继承、多态</li>
        <li>集合框架：List、Set、Map</li>
        <li>异常处理：try、catch、finally</li>
        <li>Java I/O与文件操作</li>
      </ul>
    </div>

    <h3>PYTHON学习</h3>
    <div class="container">
      Python是一种解释型、面向对象、动态数据类型的高级编程语言。以下是PYTHON学习的关键点：
      <ul>
        <li>Python基础：语法、数据类型、控制流</li>
        <li>函数与模块</li>
        <li>面向对象编程：类与对象</li>
        <li>高级特性：列表推导式、生成器、装饰器</li>
        <li>标准库与第三方库的使用</li>
      </ul>
    </div>

  </div>
</template>

<script>
export default {
  name: 'HomePage'
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 10px 0 0;
}
.hello{
  margin: 1px 0; /* 添加一些外边距 */
  padding: 10px; /* 添加内边距 */
  border: 2px solid #ccc; /* 可选：添加边框 */
}
.container {
  margin: 10px 0; /* 添加一些外边距 */
  padding: 10px; /* 添加内边距 */
  border: 2px solid #ccc; /* 可选：添加边框 */
}
</style>
