<%--
  Created by IntelliJ IDEA.
  User: wzzsa
  Date: 2024/10/22
  Time: 下午2:28
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Title</title>
</head>
<body>
<div align="center">
    <p>添加学生</p>
    <form action="/add" method="post">
        id：<input type="text" name="id"><br/>
        姓名：<input type="text" name="name"><br/>
        年龄：<input type="text" name="age"><br/>
        <input type="submit" value="注册学生">
    </form>
    <br/><br/>
    <p>查询学生</p>
    <form action="/query" method="get">
        学生id：<input type="text" name="stuid"><br/>
        <input type="submit" value="查询学生">
    </form>
</div>
</body>
</html>
