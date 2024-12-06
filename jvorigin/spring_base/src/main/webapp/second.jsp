<%--
  Created by IntelliJ IDEA.
  User: wzzsa
  Date: 2024/10/22
  Time: 下午2:28
  To change this template use File | Settings | File Templates.
--%>
<%@ page import="org.example.entity.Student" %>
<%@ page contentType="text/html;charset=utf-8" language="java" %>
<html>
<head>
    <title>$</title>
</head>
<body>
<%
    Student student= (Student) request.getAttribute("stu");
%>
查询的结果：<%=student%>
</body>
</html>
