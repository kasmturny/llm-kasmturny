package org.example.controller;

import org.example.entity.Student;
import org.example.service.StudentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.context.WebApplicationContext;
import org.springframework.web.context.support.WebApplicationContextUtils;

import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;


public class AddStudentServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("utf-8");
        response.setContentType("text/html;charset=utf-8");
        String strName=request.getParameter("name");
        String strAge=request.getParameter("age");
        String strId=request.getParameter("id");

        //获取全局作用域对象，确保spring容器对象只创建一次
        ServletContext servletContext=getServletContext();
        //使用spring提供的工具方法，获取容器对象
        WebApplicationContext ctx= WebApplicationContextUtils.getRequiredWebApplicationContext(servletContext);

        StudentService service= (StudentService) ctx.getBean("studentService");

        Student student=new Student();
        student.setId(Integer.valueOf(strId));
        student.setName(strName);
        student.setAge(Integer.valueOf(strAge));

        service.add_data(student);

        //给用户显示请求的处理结果
        request.getRequestDispatcher("/show.jsp").forward(request,response);
    }
}
