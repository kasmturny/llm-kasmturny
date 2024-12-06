package org.example.controller;

import org.example.entity.Student;
import org.example.service.StudentService;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.context.WebApplicationContext;
import org.springframework.web.context.support.WebApplicationContextUtils;

import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 *
 */

public class QueryStudentServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("utf-8");
        response.setContentType("text/html;charset=utf-8");
        String stuId=request.getParameter("stuid");

        //获取全局作用域对象，确保spring容器对象只创建一次
        ServletContext servletContext=getServletContext();
        //使用spring提供的工具方法，获取容器对象
        WebApplicationContext ctx= WebApplicationContextUtils.getRequiredWebApplicationContext(servletContext);

        StudentService service= (StudentService) ctx.getBean("studentService");

        Student student=service.searchById(Integer.parseInt(stuId));
        System.out.println("student对象===" + student);

        request.setAttribute("stu",student);
        request.getRequestDispatcher("/second.jsp").forward(request,response);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }
}